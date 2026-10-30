"""INT8 (W8A8-dynamic) GEMM kernels. All matmuls run on native int8 tensor cores
with int32 accumulation; scales are applied in the epilogue.

Activations carry one scale per 64-element group along K, so each int8 dot
product is dequantised with its own scale before being accumulated in fp32.
"""
import torch
import triton
import triton.language as tl

from .common import GROUP, GROUP_C, round_i8, silu, swizzle
from .w4 import gemm_w4, mlp_gate_up_quant_w4, proj3_w4


def _is_w4(b):
    """W4 checkpoints store uint8 packed nibbles; W8A16 stores int8 + 1-D scales."""
    return b.dtype == torch.uint8


@triton.jit
def gemm_i8_kernel(A, AS, B, BS, C, M, N, K, NG,
                   stride_cm, RES, stride_rm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, ADD_RES: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    m_mask = offs_m < M
    n_mask = offs_n < N

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        acc = tl.dot(a, b, out_dtype=tl.int32)
        ag = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        out += acc.to(tl.float32) * ag[:, None]
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
        offs_k += BLOCK_K
        kb += 1

    bs = tl.load(BS + offs_n, mask=n_mask, other=0.0)
    out = out * bs[None, :]
    if ADD_RES:
        out += tl.load(RES + offs_m[:, None] * stride_rm + offs_n[None, :],
                       mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def gemm_i8(a_q, a_s, b, b_s, out, residual=None, block_m=None, block_n=None, block_k=64,
            group_m=8):
    """out[M,N] (bf16) = (a_q * a_s) @ (b * b_s), optional residual add.

    `b` is int8 [K,N] with one scale per output row (W8A16 checkpoints), or uint8
    [N,K//2] packed int4 with per-group scales (W4A16 checkpoints), in which case
    the nibbles are expanded to int8 inside the dot loop.

    a_s holds one scale per GROUP of K (shape [M, K//GROUP])."""
    if _is_w4(b):
        return gemm_w4(a_q, a_s, b, b_s, out, residual=residual, block_m=block_m,
                       block_n=block_n, block_k=block_k, group_m=group_m)
    M, K = a_q.shape
    N = b.shape[1]
    if block_m is None:
        block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    if block_n is None:
        # Single-token decode is latency bound: keep tiles narrow so the 16 SMs
        # get several blocks each.  The huge vocab projection stays wide.
        block_n = (64 if N <= 4096 else 256) if M <= 16 else 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    stride_cm = out.stride(0)
    res = residual if residual is not None else out
    stages = 3 if block_n * block_k <= 16384 else 2
    gemm_i8_kernel[grid](
        a_q, a_s, b, b_s, out, M, N, K, a_s.shape[1],
        stride_cm, res, residual.stride(0) if residual is not None else 0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        ADD_RES=residual is not None,
        num_warps=4 if block_m <= 16 else 8, num_stages=stages,
    )
    return out


@triton.jit
def mlp_gate_up_kernel(A, AS, BG, BSG, BU, BSU, C, M, N, K, NG,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP_M: tl.constexpr):
    """h = silu(gate(x)) * up(x), stored bf16 [M,N]."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    ag = A + offs_m[:, None] * K + offs_k[None, :]
    bg = BG + offs_k[:, None] * N + offs_n[None, :]
    bu = BU + offs_k[:, None] * N + offs_n[None, :]
    g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(ag, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        b1 = tl.load(bg, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        b2 = tl.load(bu, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        acc_g = tl.dot(a, b1, out_dtype=tl.int32)
        acc_u = tl.dot(a, b2, out_dtype=tl.int32)
        as_ = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        g += acc_g.to(tl.float32) * as_[:, None]
        u += acc_u.to(tl.float32) * as_[:, None]
        ag += BLOCK_K
        bg += BLOCK_K * N
        bu += BLOCK_K * N
        offs_k += BLOCK_K
        kb += 1

    sg = tl.load(BSG + offs_n, mask=n_mask, other=0.0)
    su = tl.load(BSU + offs_n, mask=n_mask, other=0.0)
    g = g * sg[None, :]
    u = u * su[None, :]
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], (silu(g) * u).to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def mlp_gate_up(a_q, a_s, wg, wsg, wu, wsu, out, block_k=64):
    M, K = a_q.shape
    N = wg.shape[1]
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    mlp_gate_up_kernel[grid](
        a_q, a_s, wg, wsg, wu, wsu, out, M, N, K, a_s.shape[1],
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out


@triton.jit
def mlp_gate_up_quant_kernel(A, AS, BG, BSG, BU, BSU, Q, QS, M, N, K, NG,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                             NG_B: tl.constexpr):
    """h = silu(gate(x)) * up(x), written straight out as int8 + per-group scales.

    BLOCK_N must be a multiple of GROUP so each tile covers whole scale groups.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    ag = A + offs_m[:, None] * K + offs_k[None, :]
    bg = BG + offs_k[:, None] * N + offs_n[None, :]
    bu = BU + offs_k[:, None] * N + offs_n[None, :]
    g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(ag, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        b1 = tl.load(bg, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        b2 = tl.load(bu, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        acc_g = tl.dot(a, b1, out_dtype=tl.int32)
        acc_u = tl.dot(a, b2, out_dtype=tl.int32)
        as_ = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        g += acc_g.to(tl.float32) * as_[:, None]
        u += acc_u.to(tl.float32) * as_[:, None]
        ag += BLOCK_K
        bg += BLOCK_K * N
        bu += BLOCK_K * N
        offs_k += BLOCK_K
        kb += 1

    sg = tl.load(BSG + offs_n, mask=n_mask, other=0.0)
    su = tl.load(BSU + offs_n, mask=n_mask, other=0.0)
    h = silu(g * sg[None, :]) * (u * su[None, :])
    # Round through bf16 first so the int8 grid matches the reference dataflow
    # (activations are bf16 between layers).
    h = h.to(tl.bfloat16).to(tl.float32)

    h3 = tl.reshape(h, (BLOCK_M, NG_B, GROUP_C))
    amax = tl.max(tl.abs(h3), axis=2)
    sc = tl.maximum(amax / 127.0, 1e-8)
    q3 = round_i8(h3 / sc[:, :, None])
    tl.store(Q + offs_m[:, None] * N + offs_n[None, :],
             tl.reshape(q3, (BLOCK_M, BLOCK_N)),
             mask=m_mask[:, None] & n_mask[None, :])
    og = pid_n * (BLOCK_N // GROUP_C) + tl.arange(0, NG_B)
    tl.store(QS + offs_m[:, None] * (N // GROUP_C) + og[None, :], sc,
             mask=m_mask[:, None])


def mlp_gate_up_quant(a_q, a_s, wg, wsg, wu, wsu, out_q, out_s, block_k=64):
    """Fused gate/up MLP + int8 quantisation of the activation (the down_proj input)."""
    if _is_w4(wg):
        return mlp_gate_up_quant_w4(a_q, a_s, wg, wsg, wu, wsu, out_q, out_s,
                                    block_k=block_k)
    M, K = a_q.shape
    N = wg.shape[1]
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    mlp_gate_up_quant_kernel[grid](
        a_q, a_s, wg, wsg, wu, wsu, out_q, out_s, M, N, K, a_s.shape[1],
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        NG_B=block_n // GROUP,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out_q, out_s


@triton.jit
def proj3_kernel(A, AS, B1, S1, B2, S2, B3, S3, C, M, N1, N2, N3, K, NG, stride_c,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr):
    """Three (optionally) chained projections sharing the same A, written contiguously
    into C[:, 0:N1], C[:, N1:N1+N2], C[:, N1+N2:N1+N2+N3] (row stride N1+N2+N3)."""
    pid = tl.program_id(0)
    NT = N1 + N2 + N3
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(NT, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]

    if pid_n * BLOCK_N < N1:
        B = B1
        BS = S1
        NOFF = 0
        NN = N1
    elif pid_n * BLOCK_N < N1 + N2:
        B = B2
        BS = S2
        NOFF = N1
        NN = N2
    else:
        B = B3
        BS = S3
        NOFF = N1 + N2
        NN = N3

    col = offs_n - NOFF
    n_mask = (col >= 0) & (col < NN)
    b_ptrs = B + offs_k[:, None] * NN + col[None, :]
    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & n_mask[None, :], other=0)
        acc = tl.dot(a, b, out_dtype=tl.int32)
        as_ = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        out += acc.to(tl.float32) * as_[:, None]
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * NN
        offs_k += BLOCK_K
        kb += 1

    bs = tl.load(BS + col, mask=n_mask, other=0.0)
    out = out * bs[None, :]
    tl.store(C + offs_m[:, None] * stride_c + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def proj3(a_q, a_s, w1, s1, w2, s2, w3, s3, out, block_k=64):
    if _is_w4(w1):
        return proj3_w4(a_q, a_s, w1, s1, w2, s2, w3, s3, out, block_k=block_k)
    M, K = a_q.shape
    n1 = w1.shape[1] if w1 is not None else 0
    n2 = w2.shape[1] if w2 is not None else 0
    n3 = w3.shape[1] if w3 is not None else 0
    if w2 is None:
        w2, s2 = w1, s1
    if w3 is None:
        w3, s3 = w1, s1
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    grid = (triton.cdiv(M, block_m) * triton.cdiv(n1 + n2 + n3, block_n),)
    proj3_kernel[grid](
        a_q, a_s, w1, s1, w2, s2, w3, s3, out, M, n1, n2, n3, K, a_s.shape[1],
        out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out
