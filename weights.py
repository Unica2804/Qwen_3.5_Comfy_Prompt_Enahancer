"""Checkpoint loading -> stacked tensors ready for the Triton kernels.

Two quantisation formats are supported:

* **W8A16** — `<proj>.qweight` int8 `[in, out]` + `<proj>.scale` bf16 `[out]`.
  Weights stay int8 and are used directly by the GEMM.
* **W4A16 groupwise** — `<proj>.packed_weight` uint8 `[out, in//2]` (even-K in the
  low nibble, odd-K in the high nibble) + `<proj>.scale` bf16 `[out, in//group]`.
  Weights stay packed at 4 bits and are expanded to int8 inside the dot loop, so
  a 2B model keeps 685 MB of weights instead of 1.37 GB.
"""
import glob
import json
import os

import torch
from safetensors import safe_open


class QwenWeights:
    """All quantised weights stacked per role (layer index on dim 0)."""

    def __init__(self, cfg, quant="w8"):
        self.cfg = cfg
        self.quant = quant
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


def discover_models(roots):
    """Quantised Qwen3.5 checkpoints under `roots` -> [(label, path), ...].

    Labels are the folder name, disambiguated by the parent folder name when two
    roots contain the same name.
    """
    if isinstance(roots, str):
        roots = [roots]
    out, seen = [], {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d) or name.startswith("."):
                continue
            if not _is_qwen35_dir(d):
                continue
            if not glob.glob(os.path.join(d, "*.safetensors")):
                continue
            label = name
            if label in seen:
                label = f"{os.path.basename(os.path.normpath(root))}/{name}"
            seen[label] = d
            out.append((label, d))
    return out


def _pick_ckpt(d):
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    base = os.path.basename(os.path.normpath(d)).lower()
    for f in files:
        if base and base in os.path.basename(f).lower():
            return f
    return files[0]


def _quant_config(model_dir, keys):
    """Return (kind, group_size) for the checkpoint's linear-layer format."""
    qcfg = {}
    p = os.path.join(model_dir, "quantization_config.json")
    if os.path.isfile(p):
        try:
            qcfg = json.load(open(p, encoding="utf-8"))
        except Exception:
            qcfg = {}
    packed = any(k.endswith(".packed_weight") for k in keys)
    if packed:
        return "w4", int(qcfg.get("group_size", 128))
    return "w8", 0


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
    ckpt = _pick_ckpt(model_dir)
    log(f"[qwen35] loading {ckpt}")

    with safe_open(ckpt, framework="pt") as f:
        keys = set(f.keys())
        kind, wgroup = _quant_config(model_dir, keys)
        dev = torch.device(device)
        w = QwenWeights(cfg, quant=kind)
        t = w.t

        # ---- quantised linear layers: (role, out_dim, in_dim, n_layers) ------
        roles = [
            ("lin_qkv", conv_dim, H, n_lin),
            ("lin_z", VD, H, n_lin),
            ("lin_out", H, VD, n_lin),
            ("attn_q", NQ * HD * 2, H, n_full),
            ("attn_k", NKV * HD, H, n_full),
            ("attn_v", NKV * HD, H, n_full),
            ("attn_o", H, NQ * HD, n_full),
            ("mlp_gate", I, H, NL),
            ("mlp_up", I, H, NL),
            ("mlp_down", H, I, NL),
        ]
        if kind == "w4":
            for role, nout, nin, n in roles:
                assert nin % wgroup == 0 and nin % 2 == 0, (role, nin, wgroup)
                # K-major: [K//2, N] nibbles and [K//WG, N] scales, so the GEMM
                # reads each tile as one coalesced run along N.
                t[role] = _stacked((n, nin // 2, nout), torch.uint8, dev)
                t[role + "_s"] = _stacked((n, nin // wgroup, nout), torch.bfloat16, dev)
        else:
            for role, nout, nin, n in roles:
                t[role] = _stacked((n, nin, nout), torch.int8, dev)
                t[role + "_s"] = _stacked((n, nout), torch.float32, dev)

        t["embed"] = _stacked((V, H), torch.bfloat16, dev)
        t["final_norm"] = _stacked((H,), torch.bfloat16, dev)
        t["in_norm"] = _stacked((NL, H), torch.bfloat16, dev)
        t["post_norm"] = _stacked((NL, H), torch.bfloat16, dev)
        t["lin_b"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
        t["lin_a"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
        t["lin_conv"] = _stacked((n_lin, conv_dim, cfg["linear_conv_kernel_dim"]),
                                 torch.bfloat16, dev)
        t["lin_dt"] = _stacked((n_lin, NH), torch.bfloat16, dev)
        t["lin_alog"] = _stacked((n_lin, NH), torch.bfloat16, dev)
        t["lin_norm_w"] = _stacked((n_lin, DV), torch.bfloat16, dev)
        t["attn_qn"] = _stacked((n_full, HD), torch.bfloat16, dev)
        t["attn_kn"] = _stacked((n_full, HD), torch.bfloat16, dev)

        def get(name):
            return f.get_tensor(name)

        def put(role, slot, name):
            """Copy one quantised projection into its stacked destination."""
            if kind == "w4":
                t[role][slot].copy_(get(name + ".packed_weight").t().contiguous().to(dev))
                t[role + "_s"][slot].copy_(get(name + ".scale").t().contiguous().to(dev))
            else:
                t[role][slot].copy_(get(name + ".qweight").to(dev))
                t[role + "_s"][slot].copy_(get(name + ".scale").float().to(dev))

        lin_i = full_i = 0
        for i in range(NL):
            if layer_types[i] == "linear_attention":
                p = f"model.layers.{i}.linear_attn."
                put("lin_qkv", lin_i, p + "in_proj_qkv")
                put("lin_z", lin_i, p + "in_proj_z")
                put("lin_out", lin_i, p + "out_proj")
                t["lin_b"][lin_i].copy_(get(p + "in_proj_b.weight").to(dev))
                t["lin_a"][lin_i].copy_(get(p + "in_proj_a.weight").to(dev))
                t["lin_conv"][lin_i].copy_(get(p + "conv1d.weight").squeeze(1).to(dev))
                t["lin_dt"][lin_i].copy_(get(p + "dt_bias").to(dev))
                t["lin_alog"][lin_i].copy_(get(p + "A_log").to(dev))
                t["lin_norm_w"][lin_i].copy_(get(p + "norm.weight").to(dev))
                lin_i += 1
            else:
                p = f"model.layers.{i}.self_attn."
                put("attn_q", full_i, p + "q_proj")
                put("attn_k", full_i, p + "k_proj")
                put("attn_v", full_i, p + "v_proj")
                put("attn_o", full_i, p + "o_proj")
                t["attn_qn"][full_i].copy_(get(p + "q_norm.weight").to(dev))
                t["attn_kn"][full_i].copy_(get(p + "k_norm.weight").to(dev))
                full_i += 1

            p = f"model.layers.{i}."
            put("mlp_gate", i, p + "mlp.gate_proj")
            put("mlp_up", i, p + "mlp.up_proj")
            put("mlp_down", i, p + "mlp.down_proj")
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

    log(f"[qwen35] weights loaded: {w.nbytes()/1e9:.2f} GB ({kind.upper()}) on {device}")
    return w
