"""Gated delta-net (linear attention) kernels: causal depthwise conv, chunked prefill
delta rule and single-token recurrent decode."""
import torch
import triton
import triton.language as tl

from .common import silu


@triton.jit
def conv_silu_kernel(X, W, STATE, Y, T, C, KS: tl.constexpr, stride_x,
                     BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    offs_t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = offs_c < C
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for j in tl.static_range(KS):
        src = offs_t - (KS - 1) + j
        xa = tl.load(X + src[:, None] * stride_x + offs_c[None, :],
                     mask=(src >= 0)[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        xs = tl.load(STATE + (KS - 1 + src)[:, None] * C + offs_c[None, :],
                     mask=(src < 0)[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        x = tl.where((src >= 0)[:, None], xa, xs)
        w = tl.load(W + offs_c * KS + j, mask=cmask, other=0.0).to(tl.float32)
        acc += x * w[None, :]
    tl.store(Y + offs_t[:, None] * C + offs_c[None, :], silu(acc).to(tl.bfloat16),
             mask=(offs_t < T)[:, None] & cmask[None, :])


def conv_silu(x, w, state, y, T):
    C = x.shape[1]
    conv_silu_kernel[(triton.cdiv(T, 16), triton.cdiv(C, 128))](
        x, w, state, y, T, C, KS=4, stride_x=x.stride(0), BLOCK_T=16, BLOCK_C=128, num_warps=4)
    return y


@triton.jit
def conv_state_update_kernel(X, STATE, T, C, KS: tl.constexpr, stride_x,
                             BLOCK_C: tl.constexpr):
    """State holds the raw (pre-conv) last KS-1 input rows."""
    pc = tl.program_id(0)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = offs_c < C
    for j in tl.static_range(KS - 1):
        # new state row j = input row T-(KS-1)+j  (may come from the old state if T is short)
        src = T - (KS - 1) + j
        xa = tl.load(X + src * stride_x + offs_c, mask=cmask & (src >= 0), other=0.0)
        xs = tl.load(STATE + (src + KS - 1) * C + offs_c, mask=cmask & (src < 0), other=0.0)
        v = tl.where(src >= 0, xa, xs)
        tl.store(STATE + j * C + offs_c, v, mask=cmask)


def conv_state_update(x, state, T):
    C = x.shape[1]
    conv_state_update_kernel[(triton.cdiv(C, 256),)](x, state, T, C, KS=4,
                                                     stride_x=x.stride(0), BLOCK_C=256, num_warps=4)


@triton.jit
def _ut_inverse(A, C: tl.constexpr):
    """(I - A)^{-1} for strictly-lower-triangular A via the Neumann product form."""
    eye = tl.where(tl.arange(0, C)[:, None] == tl.arange(0, C)[None, :], 1.0, 0.0)
    Tm = eye + A
    P = A
    for _ in tl.static_range(5):
        P = tl.dot(P, P, input_precision="ieee", out_dtype=tl.float32)
        Tm = Tm + tl.dot(Tm, P, input_precision="ieee", out_dtype=tl.float32)
    return Tm


@triton.jit
def delta_prefill_kernel(QKV, G, BETA, STATE, OUT, T,
                         NH: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                         VS: tl.constexpr, CHUNK: tl.constexpr):
    """grid = (NH, DV // VS).  QKV: [T, 2*NH*DK + NH*DV] bf16, G/BETA: [T, NH] fp32,
    STATE: [NH, DK, DV] fp32, OUT: [T, NH*DV] bf16."""
    h = tl.program_id(0)
    vs = tl.program_id(1)
    CT: tl.constexpr = 2 * NH * DK + NH * DV

    q_base = h * DK
    k_base = NH * DK + h * DK
    v_base = 2 * NH * DK + h * DV + vs * VS

    S = tl.load(STATE + h * DK * DV + tl.arange(0, DK)[:, None] * DV + (vs * VS + tl.arange(0, VS))[None, :])

    offs_c = tl.arange(0, CHUNK)
    offs_k = tl.arange(0, DK)
    offs_v = tl.arange(0, VS)

    for c0 in range(0, T, CHUNK):
        offs_t = c0 + offs_c
        tmask = offs_t < T

        q = tl.load(QKV + offs_t[:, None] * CT + q_base + offs_k[None, :],
                    mask=tmask[:, None], other=0.0).to(tl.float32)
        k = tl.load(QKV + offs_t[:, None] * CT + k_base + offs_k[None, :],
                    mask=tmask[:, None], other=0.0).to(tl.float32)
        v = tl.load(QKV + offs_t[:, None] * CT + v_base + offs_v[None, :],
                    mask=tmask[:, None], other=0.0).to(tl.float32)
        beta = tl.load(BETA + offs_t * NH + h, mask=tmask, other=0.0)
        g = tl.load(G + offs_t * NH + h, mask=tmask, other=0.0)

        # l2 normalise q/k, scale q
        qn = q * tl.rsqrt(tl.sum(q * q, axis=1, keep_dims=True) + 1e-6) * (1.0 / tl.sqrt(float(DK)))
        kn = k * tl.rsqrt(tl.sum(k * k, axis=1, keep_dims=True) + 1e-6)

        gc = tl.cumsum(g, axis=0)
        decay = tl.where(offs_c[:, None] >= offs_c[None, :],
                         tl.exp(gc[:, None] - gc[None, :]), 0.0)

        kb = kn * beta[:, None]
        A = -tl.dot(kb, tl.trans(kn), input_precision="ieee", out_dtype=tl.float32) * decay
        A = tl.where(offs_c[:, None] > offs_c[None, :], A, 0.0)
        Tm = _ut_inverse(A, CHUNK)

        vb = v * beta[:, None]
        value = tl.dot(Tm, vb, input_precision="ieee", out_dtype=tl.float32)
        kcum = tl.dot(Tm, kb * tl.exp(gc)[:, None], input_precision="ieee", out_dtype=tl.float32)

        v_new = value - tl.dot(kcum, S, input_precision="ieee", out_dtype=tl.float32)
        out = tl.dot(qn * tl.exp(gc)[:, None], S, input_precision="ieee", out_dtype=tl.float32)
        out += tl.dot(tl.dot(qn, tl.trans(kn), input_precision="ieee", out_dtype=tl.float32) * decay,
                      v_new, input_precision="ieee", out_dtype=tl.float32)

        g_last = tl.min(gc)
        S = S * tl.exp(g_last) + tl.dot(
            tl.trans(kn * tl.exp(g_last - gc)[:, None]), v_new,
            input_precision="ieee", out_dtype=tl.float32)

        tl.store(OUT + offs_t[:, None] * (NH * DV) + h * DV + vs * VS + offs_v[None, :],
                 out.to(tl.bfloat16), mask=tmask[:, None])

    tl.store(STATE + h * DK * DV + tl.arange(0, DK)[:, None] * DV + (vs * VS + tl.arange(0, VS))[None, :], S)


def delta_prefill(qkv, g, beta, state, out, T, vs=32, chunk=32):
    NH, DK, DV = 16, 128, 128
    delta_prefill_kernel[(NH, DV // vs)](qkv, g, beta, state, out, T,
                                         NH=NH, DK=DK, DV=DV, VS=vs, CHUNK=chunk,
                                         num_warps=4, num_stages=1)
    return out


@triton.jit
def delta_decode_kernel(QKV, G, BETA, STATE, OUT, NH: tl.constexpr, DK: tl.constexpr,
                        DV: tl.constexpr, VS: tl.constexpr):
    """grid = (NH, DV // VS) for a single new token."""
    h = tl.program_id(0)
    vs = tl.program_id(1)
    offs_k = tl.arange(0, DK)
    offs_v = tl.arange(0, VS)

    q = tl.load(QKV + h * DK + offs_k).to(tl.float32)
    k = tl.load(QKV + NH * DK + h * DK + offs_k).to(tl.float32)
    v = tl.load(QKV + 2 * NH * DK + h * DV + vs * VS + offs_v).to(tl.float32)
    beta = tl.load(BETA + h)
    g = tl.load(G + h)

    qn = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * (1.0 / tl.sqrt(float(DK)))
    kn = k * tl.rsqrt(tl.sum(k * k) + 1e-6)

    sp = STATE + h * DK * DV + offs_k[:, None] * DV + (vs * VS + offs_v)[None, :]
    S = tl.load(sp)
    S = S * tl.exp(g)
    kv = tl.sum(kn[:, None] * S, axis=0)
    delta = (v - kv) * beta
    S = S + kn[:, None] * delta[None, :]
    out = tl.sum(qn[:, None] * S, axis=0)
    tl.store(sp, S)
    tl.store(OUT + h * DV + vs * VS + offs_v, out.to(tl.bfloat16))


def delta_decode(qkv, g, beta, state, out, vs=32):
    NH, DK, DV = 16, 128, 128
    delta_decode_kernel[(NH, DV // vs)](qkv, g, beta, state, out,
                                        NH=NH, DK=DK, DV=DV, VS=vs, num_warps=4)
    return out
