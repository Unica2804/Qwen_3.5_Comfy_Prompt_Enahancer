"""Full-attention kernels (partial RoPE, GQA) and KV cache handling."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_store(src, w, dst, pos, invf, RD: tl.constexpr, HD: tl.constexpr,
                     eps, scale, ONE_PLUS: tl.constexpr):
    """Gated-free RMSNorm (1+w) over HD then partial rotate_half RoPE on the first RD dims."""
    D2: tl.constexpr = RD // 2
    d = tl.arange(0, D2)
    xlo = tl.load(src + d).to(tl.float32)
    xhi = tl.load(src + D2 + d).to(tl.float32)
    dr = tl.arange(0, HD)
    rmask = dr < (HD - RD)
    xr = tl.load(src + RD + dr, mask=rmask, other=0.0).to(tl.float32)

    wlo = tl.load(w + d).to(tl.float32)
    whi = tl.load(w + D2 + d).to(tl.float32)
    wr = tl.load(w + RD + dr, mask=rmask, other=0.0).to(tl.float32)
    if ONE_PLUS:
        wlo += 1.0
        whi += 1.0
        wr += 1.0

    var = (tl.sum(xlo * xlo) + tl.sum(xhi * xhi) + tl.sum(xr * xr)) / HD
    rstd = tl.rsqrt(var + eps)
    ylo = xlo * rstd * wlo
    yhi = xhi * rstd * whi
    yr = xr * rstd * wr

    theta = pos * tl.load(invf + d)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(dst + d, ((ylo * cos - yhi * sin) * scale).to(tl.bfloat16))
    tl.store(dst + D2 + d, ((yhi * cos + ylo * sin) * scale).to(tl.bfloat16))
    tl.store(dst + RD + dr, (yr * scale).to(tl.bfloat16), mask=rmask)


@triton.jit
def qk_norm_rope_kernel(PROJ, QN, KN, INVF, QOUT, KVC, T, pos0,
                        NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                        RD: tl.constexpr, eps, stride_pm):
    """grid = (T, NQ + 2*NKV): q heads, then k heads, then v heads."""
    row = tl.program_id(0)
    hi = tl.program_id(1)
    qscale = 1.0 / tl.sqrt(float(HD))

    if hi < NQ:
        _norm_rope_store(PROJ + row * stride_pm + hi * 2 * HD, QN,
                         QOUT + row * (NQ * HD) + hi * HD, pos0 + row, INVF,
                         RD, HD, eps, qscale, True)
    elif hi < NQ + NKV:
        _norm_rope_store(PROJ + row * stride_pm + NQ * 2 * HD + (hi - NQ) * HD, KN,
                         KVC + (pos0 + row) * (2 * NKV * HD) + (hi - NQ) * HD, pos0 + row, INVF,
                         RD, HD, eps, 1.0, True)
    else:
        offs = tl.arange(0, HD)
        src = PROJ + row * stride_pm + NQ * 2 * HD + NKV * HD + (hi - NQ - NKV) * HD
        v = tl.load(src + offs)
        tl.store(KVC + (pos0 + row) * (2 * NKV * HD) + (hi - NQ) * HD + offs, v)


def qk_norm_rope(proj, qn, kn, invf, qout, kvc, pos0, eps=1e-6):
    T = proj.shape[0]
    NQ, NKV, HD = 8, 2, 256
    qk_norm_rope_kernel[(T, NQ + 2 * NKV)](
        proj, qn, kn, invf, qout, kvc, T, pos0,
        NQ=NQ, NKV=NKV, HD=HD, RD=64, eps=eps, stride_pm=proj.stride(0), num_warps=4,
    )


@triton.jit
def attn_prefill_kernel(Q, KVC, OUT, T, pos0, NQ: tl.constexpr, NKV: tl.constexpr,
                        HD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    head = pid % NQ
    mb = pid // NQ
    kvh = head // (NQ // NKV)

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * (NQ * HD) + head * HD + offs_d[None, :],
                mask=(offs_m < T)[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HD), tl.float32)

    end = tl.minimum((mb + 1) * BLOCK_M, T)
    for start_n in range(0, end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        k = tl.load(KVC + (pos0 + offs_n)[:, None] * (2 * NKV * HD) + kvh * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        causal = (offs_m[:, None] >= offs_n[None, :]) & nmask[None, :]
        s = tl.where(causal, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(KVC + (pos0 + offs_n)[:, None] * (2 * NKV * HD) + (NKV + kvh) * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(OUT + offs_m[:, None] * (NQ * HD) + head * HD + offs_d[None, :],
             out.to(tl.bfloat16), mask=(offs_m < T)[:, None])


def attn_prefill(q, kvc, out, T, pos0=0, block_m=32, block_n=32):
    NQ, NKV, HD = 8, 2, 256
    grid = (triton.cdiv(T, block_m) * NQ,)
    attn_prefill_kernel[grid](q, kvc, out, T, pos0, NQ=NQ, NKV=NKV, HD=HD,
                              BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=2)
    return out


@triton.jit
def attn_decode_kernel(Q, KVC, OUT, pos, NQ: tl.constexpr, NKV: tl.constexpr,
                       HD: tl.constexpr, BLOCK_N: tl.constexpr):
    head = tl.program_id(0)
    kvh = head // (NQ // NKV)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + head * HD + offs_d).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HD,), tl.float32)
    n = pos + 1
    for start_n in range(0, n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < n
        k = tl.load(KVC + offs_n[:, None] * (2 * NKV * HD) + kvh * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1)
        s = tl.where(nmask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        v = tl.load(KVC + offs_n[:, None] * (2 * NKV * HD) + (NKV + kvh) * HD + offs_d[None, :],
                    mask=nmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(OUT + head * HD + offs_d, (acc / l_i).to(tl.bfloat16))


def attn_decode(q, kvc, out, pos, block_n=32):
    NQ, NKV, HD = 8, 2, 256
    attn_decode_kernel[(NQ,)](q, kvc, out, pos, NQ=NQ, NKV=NKV, HD=HD,
                              BLOCK_N=block_n, num_warps=4)
    return out
