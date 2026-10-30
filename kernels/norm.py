"""Normalisation / quantisation kernels.

Activations are quantised to int8 with one scale per 64-element group
(`GROUP`), not one scale per row.  A single outlier therefore only costs
precision inside its own group, which keeps the INT8 GEMM faithful to the
bf16 reference even on layers with massive activations.
"""
import torch
import triton
import triton.language as tl

from .common import GROUP, GROUP_C, round_i8, silu, softplus


@triton.jit
def _quant_groups(y, Q, S, row, K, offs, mask, BLOCK_K: tl.constexpr,
                  NG: tl.constexpr, NG_REAL):
    """Quantise a padded row `y` (BLOCK_K values) into NG groups of GROUP."""
    y2 = tl.reshape(y, (NG, GROUP_C))
    amax = tl.max(tl.abs(y2), axis=1)
    s = tl.maximum(amax / 127.0, 1e-8)
    q2 = round_i8(y2 / s[:, None])
    tl.store(Q + row * K + offs, tl.reshape(q2, (BLOCK_K,)), mask=mask)
    gidx = tl.arange(0, NG)
    tl.store(S + row * NG_REAL + gidx, s, mask=gidx < NG_REAL)


@triton.jit
def rmsnorm_quant_kernel(X, W, Q, S, M, K, eps, stride_xm, NG_REAL,
                         ONE_PLUS: tl.constexpr, BLOCK_K: tl.constexpr,
                         NG: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    if ONE_PLUS:
        w = w + 1.0
    var = tl.sum(x * x, axis=0) / K
    y = x * tl.rsqrt(var + eps) * w
    _quant_groups(y, Q, S, row, K, offs, mask, BLOCK_K, NG, NG_REAL)


def _ng(k):
    return k // GROUP


def rmsnorm_quant(x, w, q, s, eps=1e-6, one_plus=True):
    """x: bf16 [M,K] (row stride may differ), w: [K], q: int8 [M,K], s: fp32 [M,K/64]."""
    M, K = q.shape
    BLOCK_K = max(256, triton.next_power_of_2(K))
    rmsnorm_quant_kernel[(M,)](
        x, w, q, s, M, K, eps, x.stride(0), _ng(K),
        ONE_PLUS=one_plus, BLOCK_K=BLOCK_K, NG=BLOCK_K // GROUP,
        num_warps=4 if BLOCK_K <= 1024 else 8,
    )
    return q, s


@triton.jit
def rmsnorm_bf16_kernel(X, W, Y, M, K, eps, stride_xm,
                        ONE_PLUS: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    if ONE_PLUS:
        w = w + 1.0
    var = tl.sum(x * x, axis=0) / K
    y = x * tl.rsqrt(var + eps) * w
    tl.store(Y + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def quant_row_kernel(X, Q, S, M, K, stride_xm, NG_REAL,
                     BLOCK_K: tl.constexpr, NG: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    _quant_groups(x, Q, S, row, K, offs, mask, BLOCK_K, NG, NG_REAL)


def quant_row(x, q, s):
    """x bf16 [M,K] -> int8 q [M,K], per-64-group scales s [M,K/64]."""
    M, K = q.shape
    BLOCK_K = max(256, triton.next_power_of_2(K))
    quant_row_kernel[(M,)](x, q, s, M, K, x.stride(0), _ng(K),
                           BLOCK_K=BLOCK_K, NG=BLOCK_K // GROUP,
                           num_warps=4 if BLOCK_K <= 1024 else 8)
    return q, s


@triton.jit
def norm_gated_quant_kernel(V, Z, W, Q, S, T, NG_REAL, HD: tl.constexpr,
                            NV: tl.constexpr, eps, BLOCK: tl.constexpr,
                            NG_H: tl.constexpr, stride_z):
    """Per (token, head) gated RMSNorm of the delta-rule output, quantised to int8.

    Each program owns one whole head (HD contiguous dims = NG_H groups), so it can
    emit both the int8 activations and their per-group scales without a second pass.
    """
    pid = tl.program_id(0)
    row = pid // NV
    head = pid % NV
    offs = tl.arange(0, BLOCK)
    mask = offs < HD
    base = (row * NV + head) * HD
    x = tl.load(V + base + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + row * stride_z + head * HD + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HD
    y = x * tl.rsqrt(var + eps) * w * silu(z)
    # Match the reference bf16 activation precision before quantising.
    y = y.to(tl.bfloat16).to(tl.float32)

    y2 = tl.reshape(y, (NG_H, GROUP_C))
    amax = tl.max(tl.abs(y2), axis=1)
    s = tl.maximum(amax / 127.0, 1e-8)
    q2 = round_i8(y2 / s[:, None])
    tl.store(Q + base + offs, tl.reshape(q2, (BLOCK,)), mask=mask)
    g = tl.arange(0, NG_H)
    tl.store(S + row * NG_REAL + head * NG_H + g, s)


def norm_gated_quant(v, z, w, q, s, eps=1e-6):
    """v: [T, NV*HD] bf16, z: [T, NV*HD] gate, w: [HD].
    Writes q int8 [T, NV*HD] and s fp32 [T, (NV*HD)//GROUP]."""
    T, N = v.shape
    HD = w.numel()
    NV = N // HD
    BLOCK = max(128, triton.next_power_of_2(HD))
    norm_gated_quant_kernel[(T * NV,)](v, z, w, q, s, T, _ng(N), HD, NV, eps,
                                       BLOCK=BLOCK, NG_H=BLOCK // GROUP,
                                       stride_z=z.stride(0), num_warps=4)
    return q, s


@triton.jit
def norm_gated_bf16_kernel(V, Z, W, Y, T, HD: tl.constexpr, NV: tl.constexpr,
                           eps, BLOCK: tl.constexpr, stride_z):
    """Per (token, head) gated RMSNorm of the delta-rule output (weight used directly)."""
    pid = tl.program_id(0)
    row = pid // NV
    head = pid % NV
    offs = tl.arange(0, BLOCK)
    mask = offs < HD
    base = (row * NV + head) * HD
    x = tl.load(V + base + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + row * stride_z + head * HD + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HD
    y = x * tl.rsqrt(var + eps) * w * silu(z)
    tl.store(Y + base + offs, y.to(tl.bfloat16), mask=mask)


def norm_gated(v, z, w, out, eps=1e-6):
    T, N = out.shape
    HD = w.numel()
    NV = N // HD
    BLOCK = max(128, triton.next_power_of_2(HD))
    norm_gated_bf16_kernel[(T * NV,)](v, z, w, out, T, HD, NV, eps, BLOCK=BLOCK,
                                      stride_z=z.stride(0), num_warps=4)
    return out


@triton.jit
def gate_mul_quant_kernel(A, G, Q, S, M, N, NG_REAL, HD: tl.constexpr,
                          BLOCK: tl.constexpr, NG: tl.constexpr, stride_g):
    """out = A * sigmoid(G) with G laid out as [M, heads, 2*HD] (gate in 2nd half of each head)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    gcol = (offs // HD) * (2 * HD) + HD + (offs % HD)
    g = tl.load(G + row * stride_g + gcol, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(g)
    _quant_groups(y, Q, S, row, N, offs, mask, BLOCK, NG, NG_REAL)


def gate_mul_quant(a, g, q, s, hd=256):
    M, N = a.shape
    HD = hd
    BLOCK = max(256, triton.next_power_of_2(N))
    gate_mul_quant_kernel[(M,)](a, g, q, s, M, N, _ng(N), HD, BLOCK=BLOCK,
                                NG=BLOCK // GROUP, stride_g=g.stride(0), num_warps=8)
    return q, s


@triton.jit
def silu_mul_quant_kernel(G, U, Q, S, M, N, NG_REAL,
                          BLOCK: tl.constexpr, NG: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(G + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    y = silu(g) * u
    _quant_groups(y, Q, S, row, N, offs, mask, BLOCK, NG, NG_REAL)


def silu_mul_quant(g, u, q, s):
    M, N = q.shape
    BLOCK = max(256, triton.next_power_of_2(N))
    silu_mul_quant_kernel[(M,)](g, u, q, s, M, N, _ng(N), BLOCK=BLOCK,
                                NG=BLOCK // GROUP, num_warps=8)
    return q, s


@triton.jit
def gate_proj_kernel(A, AS, WB, WA, BETA, GATE, M, K, NG_REAL, NB: tl.constexpr,
                     DT_BIAS, A_LOG, BLOCK_K: tl.constexpr):
    """beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias), from int8 activations."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    a = tl.load(A + row * K + offs, mask=mask, other=0).to(tl.float32)
    s = tl.load(AS + row * NG_REAL + offs // GROUP_C, mask=mask, other=1.0)
    x = a * s
    nb = tl.arange(0, NB)
    wb = tl.load(WB + nb[:, None] * K + offs[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    wa = tl.load(WA + nb[:, None] * K + offs[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    b = tl.sum(wb * x[None, :], axis=1)
    av = tl.sum(wa * x[None, :], axis=1)
    dtb = tl.load(DT_BIAS + nb).to(tl.float32)
    alog = tl.load(A_LOG + nb).to(tl.float32)
    beta = tl.sigmoid(b)
    g = -tl.exp(alog) * softplus(av + dtb)
    tl.store(BETA + row * NB + nb, beta)
    tl.store(GATE + row * NB + nb, g)


def gate_proj(a_q, a_s, wb, wa, beta, gate, dt_bias, a_log):
    M, K = a_q.shape
    NB = wb.shape[0]
    BLOCK_K = max(256, triton.next_power_of_2(K))
    gate_proj_kernel[(M,)](a_q, a_s, wb, wa, beta, gate, M, K, _ng(K), NB,
                           dt_bias, a_log, BLOCK_K=BLOCK_K, num_warps=4)
    return beta, gate
