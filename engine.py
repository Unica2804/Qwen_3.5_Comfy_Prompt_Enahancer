"""Qwen3.5-0.8B INT8 inference engine (prefill + decode) driving the Triton kernels."""
import math
import os
import time
import types

import torch

from .kernels.attention import attn_decode, attn_prefill, qk_norm_rope
from .kernels.common import GROUP
from .kernels.gemm import gemm_i8, mlp_gate_up_quant, proj3
from .kernels.linear_attn import (conv_silu, conv_state_update, delta_decode,
                                  delta_prefill)
from .kernels.norm import (gate_mul_quant, gate_proj, norm_gated_quant,
                           rmsnorm_quant)
from .weights import load_weights


def rope_inv_freq(rope_theta, dim, device):
    return 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))


class Workspace:
    """Scratch buffers sized for `cap` tokens."""

    def __init__(self, cfg, cap, device):
        self.cap = cap
        self._views = {}
        H = cfg["hidden_size"]
        I = cfg["intermediate_size"]
        NH = cfg["linear_num_value_heads"]
        DK = cfg["linear_key_head_dim"]
        DV = cfg["linear_value_head_dim"]
        conv_dim = 2 * NH * DK + NH * DV
        NQ = cfg["num_attention_heads"]
        HD = cfg["head_dim"]
        V = cfg["vocab_size"]
        d = dict(device=device)
        ng = lambda k: k // GROUP
        self.hidden = torch.empty((cap, H), dtype=torch.bfloat16, **d)
        self.xq = torch.empty((cap, H), dtype=torch.int8, **d)
        self.xs = torch.empty((cap, ng(H)), dtype=torch.float32, **d)
        self.proj = torch.empty((cap, max(conv_dim + NH * DV, NQ * HD * 2 + 2 * 2 * HD)),
                                dtype=torch.bfloat16, **d)
        self.mixed = torch.empty((cap, conv_dim), dtype=torch.bfloat16, **d)
        self.dout = torch.empty((cap, NH * DV), dtype=torch.bfloat16, **d)
        self.aq_o = torch.empty((cap, NQ * HD), dtype=torch.int8, **d)
        self.as_o = torch.empty((cap, ng(NQ * HD)), dtype=torch.float32, **d)
        self.aq_d = torch.empty((cap, I), dtype=torch.int8, **d)
        self.as_d = torch.empty((cap, ng(I)), dtype=torch.float32, **d)
        self.q = torch.empty((cap, NQ * HD), dtype=torch.bfloat16, **d)
        self.aout = torch.empty((cap, NQ * HD), dtype=torch.bfloat16, **d)
        self.hq = torch.empty((cap, I), dtype=torch.int8, **d)
        self.hs = torch.empty((cap, ng(I)), dtype=torch.float32, **d)
        self.beta = torch.empty((cap, NH), dtype=torch.float32, **d)
        self.gate = torch.empty((cap, NH), dtype=torch.float32, **d)
        self.logits = torch.empty((1, V), dtype=torch.bfloat16, **d)
        self.xq1 = torch.empty((1, H), dtype=torch.int8, **d)
        self.xs1 = torch.empty((1, ng(H)), dtype=torch.float32, **d)

    def view(self, T):
        """Cached namespace of the first T rows.

        Attribute access on the result must be a plain slot lookup: decode calls
        this hundreds of times per token, so a lazy __getattr__ is far too slow.
        """
        v = self._views.get(T)
        if v is None:
            v = types.SimpleNamespace()
            for name, val in vars(self).items():
                if (isinstance(val, torch.Tensor) and val.dim() >= 1
                        and val.shape[0] == self.cap and name != "logits"):
                    val = val[:T]
                setattr(v, name, val)
            self._views[T] = v
        return v


class Engine:
    def __init__(self, model_dir, device="cuda", max_seq=4096, prefill_chunk=512, log=print):
        self.log = log
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.max_seq = max_seq
        self.prefill_chunk = prefill_chunk
        self.w = load_weights(model_dir, device=self.device, log=log)
        cfg = self.w.cfg
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        self.NL = cfg["num_hidden_layers"]
        self.layer_types = cfg["layer_types"]
        self.n_lin = sum(1 for x in self.layer_types if x == "linear_attention")
        self.NH = cfg["linear_num_value_heads"]
        self.DK = cfg["linear_key_head_dim"]
        self.DV = cfg["linear_value_head_dim"]
        self.VD = self.NH * self.DV
        self.conv_dim = 2 * self.NH * self.DK + self.VD
        self.NQ = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.V = cfg["vocab_size"]
        self.invf = rope_inv_freq(cfg["rope_parameters"]["rope_theta"], 64, self.device)

        self.kv = torch.zeros((self.NL, max_seq, 2 * self.NKV * self.HD),
                              dtype=torch.bfloat16, device=self.device)
        self.dstate = torch.zeros((self.n_lin, self.NH, self.DK, self.DV),
                                  dtype=torch.float32, device=self.device)
        self.cstate = torch.zeros((self.n_lin, 3, self.conv_dim),
                                  dtype=torch.bfloat16, device=self.device)
        self.ws = Workspace(cfg, prefill_chunk, self.device)
        self.pos = 0
        self._mega = None
        self._collect = False
        self.debug = None
        self.log(f"[qwen35] engine ready on {self.device}, "
                 f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # ------------------------------------------------------------------ state
    def ensure_seq(self, n):
        """Grow the KV cache so `n` positions fit (state is preserved)."""
        if n <= self.max_seq:
            return
        new = max(int(n), self.max_seq * 2)
        kv = torch.zeros((self.NL, new, 2 * self.NKV * self.HD),
                         dtype=torch.bfloat16, device=self.kv.device)
        if self.pos > 0:
            kv[:, : self.pos] = self.kv[:, : self.pos]
        self.kv = kv
        self.max_seq = new

    def reset(self):
        self.pos = 0
        self.dstate.zero_()
        self.cstate.zero_()

    def offload(self):
        self.w.offload()
        self.dstate = self.dstate.to("cpu")
        self.cstate = self.cstate.to("cpu")
        self.kv = self.kv.to("cpu")
        self.ws = None
        torch.cuda.empty_cache()

    def reload(self):
        self.w.to(self.device)
        d = self.device
        self.kv = self.kv.to(d)
        self.dstate = self.dstate.to(d)
        self.cstate = self.cstate.to(d)
        self.ws = Workspace(self.cfg, self.prefill_chunk, d)

    @property
    def on_gpu(self):
        return self.w.device.type == "cuda"

    # --------------------------------------------------------------- forward
    def _linear_layer(self, i, li, T, pos0):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["lin_qkv"][li], w["lin_qkv_s"][li],
              w["lin_z"][li], w["lin_z_s"][li], None, None,
              ws.proj, block_k=64)
        gate_proj(ws.xq, ws.xs, w["lin_b"][li], w["lin_a"][li],
                  ws.beta, ws.gate, w["lin_dt"][li], w["lin_alog"][li])
        mixed_in = ws.proj[:, : self.conv_dim]
        conv_silu(mixed_in, w["lin_conv"][li], self.cstate[li], ws.mixed, T)
        conv_state_update(mixed_in, self.cstate[li], T)
        if T == 1:
            delta_decode(ws.mixed[0], ws.gate[0], ws.beta[0], self.dstate[li], ws.dout[0])
        else:
            delta_prefill(ws.mixed, ws.gate, ws.beta, self.dstate[li], ws.dout, T)
        norm_gated_quant(ws.dout, ws.proj[:, self.conv_dim:], w["lin_norm_w"][li],
                         ws.aq_o, ws.as_o, self.eps)
        gemm_i8(ws.aq_o, ws.as_o, w["lin_out"][li], w["lin_out_s"][li],
                ws.hidden, residual=ws.hidden)

    def _attn_layer(self, i, fi, T, pos0):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["attn_q"][fi], w["attn_q_s"][fi],
              w["attn_k"][fi], w["attn_k_s"][fi],
              w["attn_v"][fi], w["attn_v_s"][fi],
              ws.proj, block_k=64)
        qk_norm_rope(ws.proj, w["attn_qn"][fi], w["attn_kn"][fi], self.invf,
                     ws.q, self.kv[i], pos0, self.eps)
        if T == 1:
            attn_decode(ws.q[0], self.kv[i], ws.aout[0], pos0)
        else:
            attn_prefill(ws.q, self.kv[i], ws.aout, T, pos0)
        gate_mul_quant(ws.aout, ws.proj, ws.aq_o, ws.as_o)
        gemm_i8(ws.aq_o, ws.as_o, w["attn_o"][fi], w["attn_o_s"][fi],
                ws.hidden, residual=ws.hidden)

    def _mlp(self, i, T):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["post_norm"][i], ws.xq, ws.xs, self.eps)
        mlp_gate_up_quant(ws.xq, ws.xs, w["mlp_gate"][i], w["mlp_gate_s"][i],
                          w["mlp_up"][i], w["mlp_up_s"][i], ws.hq, ws.hs)
        gemm_i8(ws.hq, ws.hs, w["mlp_down"][i], w["mlp_down_s"][i],
                ws.hidden, residual=ws.hidden)

    def _final_logits(self, T):
        w, ws = self.w, self.ws.view(T)
        last = ws.hidden[T - 1: T]
        rmsnorm_quant(last, w["final_norm"], ws.xq1, ws.xs1, self.eps)
        gemm_i8(ws.xq1, ws.xs1, w["lm_head"], w["lm_head_s"], ws.logits,
                block_n=512, block_k=64)
        return ws.logits[0]

    def logits_all(self, T):
        """Logits for every one of the T rows currently in the workspace (debug/parity)."""
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["final_norm"], ws.xq, ws.xs, self.eps)
        out = torch.empty((T, self.V), dtype=torch.bfloat16, device=self.device)
        gemm_i8(ws.xq, ws.xs, w["lm_head"], w["lm_head_s"], out,
                block_n=512, block_k=64)
        return out

    def forward(self, token_ids, pos0=0):
        """Run `token_ids` (list[int]) starting at absolute position `pos0`; returns logits."""
        T = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        ws = self.ws.view(T)
        torch.index_select(self.w["embed"], 0, ids, out=ws.hidden)
        lin_i = full_i = 0
        self.debug = [] if self._collect else None
        for i in range(self.NL):
            if self.layer_types[i] == "linear_attention":
                self._linear_layer(i, lin_i, T, pos0)
                lin_i += 1
            else:
                self._attn_layer(i, full_i, T, pos0)
                full_i += 1
            self._mlp(i, T)
            if self.debug is not None:
                self.debug.append(ws.hidden[T - 1].float().cpu())
        self.pos = pos0 + T
        return self._final_logits(T)

    # ------------------------------------------------------------ generation
    def prefill(self, token_ids):
        """Run the prompt (chunked) and return the logits for the next token."""
        self.reset()
        logits = None
        T = len(token_ids)
        for c0 in range(0, T, self.prefill_chunk):
            chunk = token_ids[c0: c0 + self.prefill_chunk]
            logits = self.forward(chunk, pos0=c0)
        return logits

    def decode_step(self, token_id):
        return self.forward([token_id], pos0=self.pos)

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=0, top_p=1.0,
                 seed=None, stop_ids=(248046, 248044), on_token=None):
        from .sampler import sample_logits
        gen = None
        if temperature > 1e-5:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed if seed is not None else torch.seed()))
        budget = self.max_seq - 1
        self.ensure_seq(len(prompt_ids) + max_new_tokens + 1)
        budget = self.max_seq - 1
        if len(prompt_ids) + max_new_tokens > budget:
            keep = max(1, budget - max_new_tokens)
            prompt_ids = prompt_ids[-keep:]
        logits = self.prefill(prompt_ids)
        out = []
        for _ in range(max_new_tokens):
            tok = sample_logits(logits, temperature, top_k, top_p, gen)
            if tok in stop_ids:
                break
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if self.pos >= self.max_seq - 1:
                break
            logits = self.decode_step(tok)
        return out
