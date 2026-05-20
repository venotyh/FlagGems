import logging
import math

import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    x += r
    # write back to residual
    tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    # write back to input
    tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_loop_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Pass 1: fuse x+r into residual, accumulate (x+r)^2 for variance
    # Pass 2: read from residual (x+r), normalize, write to input
    # Avoids the 37.5% masked-load waste when N=2560 but BLOCK_SIZE=4096
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)

    # Pass 1: compute s = x + r, write s to residual, accumulate s^2
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(input_ptr + pid * N + n_offsets).to(tl.float32)
        r = tl.load(residual_ptr + pid * N + n_offsets).to(tl.float32)
        s = x + r
        tl.store(residual_ptr + pid * N + n_offsets, s)
        acc += s * s

    # last chunk with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(input_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(residual_ptr + pid * N + n_offsets, s, mask=mask)
    acc += s * s

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Pass 2: read residual (= x+r), normalize, write to input
    # Reverse order for L2 cache reuse of the tail chunk just written
    prev_multiple = prev_multiple_of(N, TILE_N)

    # tail chunk first (reverse step 0), with mask
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        s = tl.load(
            residual_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (s * rrms * w).to(cdtype)
        tl.store(input_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        s = tl.load(
            residual_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        y = (s * rrms * w).to(cdtype)
        tl.store(input_ptr + pid * N + n_offsets, y)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(x.device):
        if N <= 2048:
            BLOCK_SIZE = triton.next_power_of_2(N)
            fused_add_rms_norm_kernel[M,](
                x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        else:
            fused_add_rms_norm_loop_kernel[M,](x, residual, weight, N, eps)
    return x, residual
