"""W8A16 checkpoint loading -> stacked INT8 tensors ready for the Triton kernels."""
import glob
import json
import os
import re

import torch
from safetensors import safe_open

from .kernels.common import next_pow2


class QwenWeights:
    """All quantised weights stacked per role (layer index on dim 0)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.t = {}          # name -> tensor
        self.device = torch.device("cpu")
        self._cpu_backup = None

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, k):
        return self.t[k]

    def to(self, device):
        if device == self.device:
            return self
        for k, v in self.t.items():
            self.t[k] = v.to(device, non_blocking=False)
        self.device = device
        return self

    def offload(self):
        """Move every weight tensor to pinned CPU memory, freeing VRAM."""
        if self.device.type == "cpu":
            return
        for k, v in self.t.items():
            try:
                self.t[k] = v.to("cpu", non_blocking=False).pin_memory()
            except Exception:
                self.t[k] = v.to("cpu")
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.t.values())


def _stacked(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)


def _is_qwen35_dir(d):
    cfgp = os.path.join(d, "config.json")
    try:
        cfg = json.load(open(cfgp, encoding="utf-8"))
    except Exception:
        return False
    sub = cfg.get("text_config", cfg)
    mt = str(sub.get("model_type", ""))
    if "qwen3_5" not in mt and "qwen3.5" not in mt:
        return False
    return "hidden_size" in sub and "layer_types" in sub


def discover_models(llm_root):
    """Directories under models/LLM that look like a quantised Qwen3.5 checkpoint."""
    out = []
    if not os.path.isdir(llm_root):
        return out
    for name in sorted(os.listdir(llm_root)):
        d = os.path.join(llm_root, name)
        if not os.path.isdir(d) or name.startswith("."):
            continue
        if not _is_qwen35_dir(d):
            continue
        sf = [p for p in glob.glob(os.path.join(d, "*.safetensors"))]
        if not sf:
            continue
        out.append(name)
    return out


def _pick_ckpt(d):
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    base = os.path.basename(d).lower()
    for f in files:
        if base and base in os.path.basename(f).lower():
            return f
    return files[0]


def load_weights(model_dir, device="cuda", log=print):
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    I = cfg["intermediate_size"]
    NL = cfg["num_hidden_layers"]
    layer_types = cfg["layer_types"]
    n_lin = sum(1 for x in layer_types if x == "linear_attention")
    n_full = NL - n_lin
    NH = cfg["linear_num_value_heads"]
    DK = cfg["linear_key_head_dim"]
    DV = cfg["linear_value_head_dim"]
    KD = cfg["linear_key_dim"] if "linear_key_dim" in cfg else NH * DK
    VD = DV * NH
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    conv_dim = 2 * KD + VD

    dev = torch.device(device)
    w = QwenWeights(cfg)
    t = w.t

    t["embed"] = _stacked((V, H), torch.bfloat16, dev)
    t["final_norm"] = _stacked((H,), torch.bfloat16, dev)
    t["in_norm"] = _stacked((NL, H), torch.bfloat16, dev)
    t["post_norm"] = _stacked((NL, H), torch.bfloat16, dev)

    t["lin_qkv"] = _stacked((n_lin, H, conv_dim), torch.int8, dev)
    t["lin_qkv_s"] = _stacked((n_lin, conv_dim), torch.float32, dev)
    t["lin_z"] = _stacked((n_lin, H, VD), torch.int8, dev)
    t["lin_z_s"] = _stacked((n_lin, VD), torch.float32, dev)
    t["lin_b"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
    t["lin_a"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
    t["lin_conv"] = _stacked((n_lin, conv_dim, cfg["linear_conv_kernel_dim"]), torch.bfloat16, dev)
    t["lin_dt"] = _stacked((n_lin, NH), torch.bfloat16, dev)
    t["lin_alog"] = _stacked((n_lin, NH), torch.bfloat16, dev)
    t["lin_norm_w"] = _stacked((n_lin, DV), torch.bfloat16, dev)
    t["lin_out"] = _stacked((n_lin, VD, H), torch.int8, dev)
    t["lin_out_s"] = _stacked((n_lin, H), torch.float32, dev)

    t["attn_q"] = _stacked((n_full, H, NQ * HD * 2), torch.int8, dev)
    t["attn_q_s"] = _stacked((n_full, NQ * HD * 2), torch.float32, dev)
    t["attn_k"] = _stacked((n_full, H, NKV * HD), torch.int8, dev)
    t["attn_k_s"] = _stacked((n_full, NKV * HD), torch.float32, dev)
    t["attn_v"] = _stacked((n_full, H, NKV * HD), torch.int8, dev)
    t["attn_v_s"] = _stacked((n_full, NKV * HD), torch.float32, dev)
    t["attn_o"] = _stacked((n_full, NQ * HD, H), torch.int8, dev)
    t["attn_o_s"] = _stacked((n_full, H), torch.float32, dev)
    t["attn_qn"] = _stacked((n_full, HD), torch.bfloat16, dev)
    t["attn_kn"] = _stacked((n_full, HD), torch.bfloat16, dev)

    t["mlp_gate"] = _stacked((NL, H, I), torch.int8, dev)
    t["mlp_gate_s"] = _stacked((NL, I), torch.float32, dev)
    t["mlp_up"] = _stacked((NL, H, I), torch.int8, dev)
    t["mlp_up_s"] = _stacked((NL, I), torch.float32, dev)
    t["mlp_down"] = _stacked((NL, I, H), torch.int8, dev)
    t["mlp_down_s"] = _stacked((NL, H), torch.float32, dev)

    ckpt = _pick_ckpt(model_dir)
    log(f"[qwen35] loading {ckpt}")

    with safe_open(ckpt, framework="pt") as f:
        keys = set(f.keys())

        def get(name):
            return f.get_tensor(name)

        def q(name, dst, row=None):
            qw = get(name + ".qweight")
            sc = get(name + ".scale").to(torch.float32)
            if row is None:
                dst.copy_(qw.to(dev))
                return sc.to(dev)
            dst.copy_(qw.to(dev))
            return sc.to(dev)

        lin_i = full_i = 0
        for i in range(NL):
            if layer_types[i] == "linear_attention":
                p = f"model.layers.{i}.linear_attn."
                q(p + "in_proj_qkv", t["lin_qkv"][lin_i])
                t["lin_qkv_s"][lin_i].copy_(get(p + "in_proj_qkv.scale").float().to(dev))
                q(p + "in_proj_z", t["lin_z"][lin_i])
                t["lin_z_s"][lin_i].copy_(get(p + "in_proj_z.scale").float().to(dev))
                q(p + "out_proj", t["lin_out"][lin_i])
                t["lin_out_s"][lin_i].copy_(get(p + "out_proj.scale").float().to(dev))
                t["lin_b"][lin_i].copy_(get(p + "in_proj_b.weight").to(dev))
                t["lin_a"][lin_i].copy_(get(p + "in_proj_a.weight").to(dev))
                t["lin_conv"][lin_i].copy_(get(p + "conv1d.weight").squeeze(1).to(dev))
                t["lin_dt"][lin_i].copy_(get(p + "dt_bias").to(dev))
                t["lin_alog"][lin_i].copy_(get(p + "A_log").to(dev))
                t["lin_norm_w"][lin_i].copy_(get(p + "norm.weight").to(dev))
                lin_i += 1
            else:
                p = f"model.layers.{i}.self_attn."
                q(p + "q_proj", t["attn_q"][full_i])
                t["attn_q_s"][full_i].copy_(get(p + "q_proj.scale").float().to(dev))
                q(p + "k_proj", t["attn_k"][full_i])
                t["attn_k_s"][full_i].copy_(get(p + "k_proj.scale").float().to(dev))
                q(p + "v_proj", t["attn_v"][full_i])
                t["attn_v_s"][full_i].copy_(get(p + "v_proj.scale").float().to(dev))
                q(p + "o_proj", t["attn_o"][full_i])
                t["attn_o_s"][full_i].copy_(get(p + "o_proj.scale").float().to(dev))
                t["attn_qn"][full_i].copy_(get(p + "q_norm.weight").to(dev))
                t["attn_kn"][full_i].copy_(get(p + "k_norm.weight").to(dev))
                full_i += 1

            p = f"model.layers.{i}."
            q(p + "mlp.gate_proj", t["mlp_gate"][i])
            t["mlp_gate_s"][i].copy_(get(p + "mlp.gate_proj.scale").float().to(dev))
            q(p + "mlp.up_proj", t["mlp_up"][i])
            t["mlp_up_s"][i].copy_(get(p + "mlp.up_proj.scale").float().to(dev))
            q(p + "mlp.down_proj", t["mlp_down"][i])
            t["mlp_down_s"][i].copy_(get(p + "mlp.down_proj.scale").float().to(dev))
            t["in_norm"][i].copy_(get(p + "input_layernorm.weight").to(dev))
            t["post_norm"][i].copy_(get(p + "post_attention_layernorm.weight").to(dev))

        t["embed"].copy_(get("model.embed_tokens.weight").to(dev))
        t["final_norm"].copy_(get("model.norm.weight").to(dev))

        # lm_head: int8 with per-vocab-row scale, transposed to [H, V] for the GEMM
        head = get("lm_head.weight") if "lm_head.weight" in keys else get("model.embed_tokens.weight")
        head_q = _stacked((H, V), torch.int8, dev)
        head_s = _stacked((V,), torch.float32, dev)
        step = 16384
        for s0 in range(0, V, step):
            s1 = min(V, s0 + step)
            blk = head[s0:s1].to(dev, non_blocking=False).float()
            amax = blk.abs().amax(dim=1).clamp_min(1e-8)
            sc = amax / 127.0
            qb = torch.clamp(torch.round(blk / sc[:, None]), -127, 127).to(torch.int8)
            head_q[:, s0:s1].copy_(qb.T.contiguous())
            head_s[s0:s1].copy_(sc)
            del blk, qb
        t["lm_head"] = head_q
        t["lm_head_s"] = head_s

    log(f"[qwen35] weights loaded: {w.nbytes()/1e9:.2f} GB on {device}")
    return w
