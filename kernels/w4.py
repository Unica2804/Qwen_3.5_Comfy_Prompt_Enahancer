"""W4A16 GEMM kernels: int4 weights unpacked to int8 *inside* the dot loop.

The checkpoint stores weights as `uint8[N, K//2]` with the even-K element in the
low nibble and the odd-K element in the high nibble, plus one bf16 scale per
`GROUP_W`-element group along K (`bf16[N, K//WG]`).  A stored value is
`signed = nibble - 8`, `value = signed * scale`.

At load time we transpose both tensors to a K-major layout -- `[K//2, N]` and
`[K//WG, N]` -- so that one K-block of a tile is a fully coalesced run along N.
That layout also removes all shuffles from the inner loop: the even/odd K split
of the activation is free (`reshape` + `tl.split`, since the low nibble *is* the
even K), and the two halves are summed with two plain int8 dots.  Splitting the
K sum in two does not change the result, because int32 accumulation is exact and
both halves share the same activation scale and weight scale group.

We never materialise a dequantised weight.  Each K-block loads the packed
nibbles (BLOCK_K/2 bytes per column, half of what int8 would need), expands them
in registers, and runs `tl.dot(int8, int8) -> int32` on the int8 tensor cores.
Both the activation scale and the weight group scale are applied to the fp32
accumulator per K-block, so the 4-bit grid of the checkpoint is preserved
exactly: this path adds no rounding beyond the checkpoint's own.

Keeping the weights packed matters here: a 2B model holds ~1.3 G int4 weights,
i.e. 685 MB packed versus 1.37 GB if expanded to int8.
"""
import triton
import triton.language as tl

from .common import GROUP, GROUP_C, round_i8, silu, swizzle

# int4 values live in [-8, 7] once the nibble is biased by 8.
# Triton kernels can only read globals declared as tl.constexpr.
W4_ZERO_C = tl.constexpr(8)


@triton.jit
def _w4_dot(a, lo, hi, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """int32 sum over one K-block: even-K half against `lo`, odd-K against `hi`."""
    BK2: tl.constexpr = BLOCK_K // 2
    ae, ao = tl.split(tl.reshape(a, (BLOCK_M, BK2, 2)))
    return tl.dot(ae, lo, out_dtype=tl.int32) + tl.dot(ao, hi, out_dtype=tl.int32)


@triton.jit
def gemm_w4_kernel(A, AS, W, WS, C, M, N, K, NG, WNG,
                   stride_cm, RES, stride_rm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, ADD_RES: tl.constexpr, WKG: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = W + tl.arange(0, BLOCK_K // 2)[:, None] * N + offs_n[None, :]

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        v = tl.load(w_ptrs, mask=n_mask[None, :], other=0).to(tl.int32)
        lo = ((v & 0x0F) - W4_ZERO_C).to(tl.int8)
        hi = (((v >> 4) & 0x0F) - W4_ZERO_C).to(tl.int8)
        acc = _w4_dot(a, lo, hi, BLOCK_M, BLOCK_K)
        asc = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        wsc = tl.load(WS + (kb // WKG) * N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out += acc.to(tl.float32) * (asc[:, None] * wsc[None, :])
        a_ptrs += BLOCK_K
        w_ptrs += (BLOCK_K // 2) * N
        offs_k += BLOCK_K
        kb += 1

    if ADD_RES:
        out += tl.load(RES + offs_m[:, None] * stride_rm + offs_n[None, :],
                       mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def gemm_w4(a_q, a_s, b, b_s, out, residual=None, block_m=None, block_n=None,
            block_k=64, group_m=8, num_warps=None, num_stages=3):
    """out[M,N] (bf16) = (a_q * a_s) @ unpack(b) * b_s, with optional residual add.

    `b` is [K//2, N] uint8 packed nibbles, `b_s` is [K//WG, N] bf16 group scales.
    """
    M, K = a_q.shape
    N = b.shape[1]
    if block_m is None:
        block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    if block_n is None:
        block_n = (64 if N <= 4096 else 256) if M <= 16 else 128
    if num_warps is None:
        num_warps = 4 if block_m <= 16 else 8
    wkg = (K // b_s.shape[0]) // block_k
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    res = residual if residual is not None else out
    gemm_w4_kernel[grid](
        a_q, a_s, b, b_s, out, M, N, K, a_s.shape[1], b_s.shape[0],
        out.stride(0), res, residual.stride(0) if residual is not None else 0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        ADD_RES=residual is not None, WKG=wkg,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def proj3_w4_kernel(A, AS, W1, S1, W2, S2, W3, S3, C, M, N1, N2, N3, K, NG,
                    stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, WKG: tl.constexpr):
    """Three W4 projections sharing one activation tile (q/k/v, or qkv+z)."""
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
        W = W1
        WS = S1
        NOFF = 0
        NN = N1
    elif pid_n * BLOCK_N < N1 + N2:
        W = W2
        WS = S2
        NOFF = N1
        NN = N2
    else:
        W = W3
        WS = S3
        NOFF = N1 + N2
        NN = N3

    col = offs_n - NOFF
    n_mask = (col >= 0) & (col < NN)
    WNG = K // (WKG * BLOCK_K)
    w_ptrs = W + tl.arange(0, BLOCK_K // 2)[:, None] * NN + col[None, :]

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        v = tl.load(w_ptrs, mask=n_mask[None, :], other=0).to(tl.int32)
        lo = ((v & 0x0F) - W4_ZERO_C).to(tl.int8)
        hi = (((v >> 4) & 0x0F) - W4_ZERO_C).to(tl.int8)
        acc = _w4_dot(a, lo, hi, BLOCK_M, BLOCK_K)
        asc = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        wsc = tl.load(WS + (kb // WKG) * NN + col, mask=n_mask, other=0.0).to(tl.float32)
        out += acc.to(tl.float32) * (asc[:, None] * wsc[None, :])
        a_ptrs += BLOCK_K
        w_ptrs += (BLOCK_K // 2) * NN
        offs_k += BLOCK_K
        kb += 1

    tl.store(C + offs_m[:, None] * stride_c + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def proj3_w4(a_q, a_s, w1, s1, w2, s2, w3, s3, out, block_k=64):
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
    wkg = (K // s1.shape[0]) // block_k
    grid = (triton.cdiv(M, block_m) * triton.cdiv(n1 + n2 + n3, block_n),)
    proj3_w4_kernel[grid](
        a_q, a_s, w1, s1, w2, s2, w3, s3, out, M, n1, n2, n3, K, a_s.shape[1],
        out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8, WKG=wkg,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out


@triton.jit
def mlp_gate_up_quant_w4_kernel(A, AS, WG, SG, WU, SU, Q, QS, M, N, K, NG, WNG,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                                NG_B: tl.constexpr, WKG: tl.constexpr):
    """h = silu(gate(x)) * up(x) with int4 weights, emitted as int8 + group scales."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    jj = tl.arange(0, BLOCK_K // 2)
    wg_ptrs = WG + jj[:, None] * N + offs_n[None, :]
    wu_ptrs = WU + jj[:, None] * N + offs_n[None, :]

    g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    kb = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        vg = tl.load(wg_ptrs, mask=n_mask[None, :], other=0).to(tl.int32)
        vu = tl.load(wu_ptrs, mask=n_mask[None, :], other=0).to(tl.int32)
        lg = ((vg & 0x0F) - W4_ZERO_C).to(tl.int8)
        hg = (((vg >> 4) & 0x0F) - W4_ZERO_C).to(tl.int8)
        lu = ((vu & 0x0F) - W4_ZERO_C).to(tl.int8)
        hu = (((vu >> 4) & 0x0F) - W4_ZERO_C).to(tl.int8)
        asc = tl.load(AS + offs_m * NG + kb, mask=m_mask, other=0.0)
        scg = tl.load(SG + (kb // WKG) * N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        scu = tl.load(SU + (kb // WKG) * N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        g += _w4_dot(a, lg, hg, BLOCK_M, BLOCK_K).to(tl.float32) * (asc[:, None] * scg[None, :])
        u += _w4_dot(a, lu, hu, BLOCK_M, BLOCK_K).to(tl.float32) * (asc[:, None] * scu[None, :])
        a_ptrs += BLOCK_K
        wg_ptrs += (BLOCK_K // 2) * N
        wu_ptrs += (BLOCK_K // 2) * N
        offs_k += BLOCK_K
        kb += 1

    h = silu(g) * u
    # Match the bf16 activation precision of the reference dataflow.
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


def mlp_gate_up_quant_w4(a_q, a_s, wg, sg, wu, su, out_q, out_s, block_k=64):
    M, K = a_q.shape
    N = wg.shape[1]
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    wkg = (K // sg.shape[0]) // block_k
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    mlp_gate_up_quant_w4_kernel[grid](
        a_q, a_s, wg, sg, wu, su, out_q, out_s, M, N, K, a_s.shape[1], sg.shape[0],
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        NG_B=block_n // GROUP, WKG=wkg,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out_q, out_s
