# Qwen3.5 Prompt Enhancer (INT8 Triton)

A ComfyUI custom node that turns a short idea into a detailed text-to-image
prompt, using a local **Qwen3.5** checkpoint driven by a hand-written **INT8
Triton** inference engine.

Everything runs on the embedded Python that ships with ComfyUI — no extra
packages, no server, no API keys.

```
STRING (short idea) ──► [Qwen3.5 Prompt Enhancer] ──► STRING (detailed prompt)
```

## Setup

1. Clone the repo into `ComfyUI/custom_nodes`:

   ```bash
   git clone https://github.com/Unica2804/Qwen_3.5_Comfy_Prompt_Enahancer.git \
     ComfyUI/custom_nodes/qwen35_prompt_enhancer
   ```

2. Put a quantised Qwen3.5 checkpoint in `ComfyUI/models/LLM/<folder>`. The W8A16
   0.8B release on
   [Mlboy23/qwen3.5-0.8b-w8a16](https://huggingface.co/Mlboy23/qwen3.5-0.8b-w8a16) is the smallest starting point; 
   [Mlboy23/Qwen3.5-2B-W4A16](https://huggingface.co/Mlboy23/Qwen3.5-2B-W4A16) is the 2B W4A16 release
   runs through the same code. Checkpoints elsewhere can be added with
   `QWEN35_MODEL_PATHS` or `EXTRA_MODEL_ROOTS` — see
   [Model discovery](#model-discovery).

3. Restart ComfyUI. The node appears as `Qwen3.5 Prompt Enhancer (INT8)` under
   `text/Qwen3.5`.

A checkpoint folder needs `config.json`, the tokenizer (`tokenizer.json` plus
`chat_template.jinja`), one `.safetensors`, and — for W4 checkpoints —
`quantization_config.json`. Each folder is read as a single safetensors file: if
several are present, the one whose name contains the folder name wins,
otherwise the first one alphabetically.

No `pip install` is needed: the engine only uses `torch`, `triton`,
`safetensors`, `tokenizers` and `jinja2`, all of which ComfyUI already ships.

## Node

Category: `text/Qwen3.5`

| Input | Type | Notes |
|---|---|---|
| `text` | STRING (multiline) | The idea / keyword list to expand. Defaults to `a cat sitting on a windowsill`. |
| `model` | combo | Any Qwen3.5 checkpoint found in the scanned folders. |
| `memory_mode` | combo | `offload_after_run` (default, frees VRAM after every run) or `keep_loaded` (resident, faster repeats). |
| `max_tokens` | combo | Generation cap, default `1024`: `512` / `1024` / `2048` / `4096`. |
| `system_prompt` | STRING (multiline) | Defines the enhancer's style; defaults to the built-in prompt template below. |
| `temperature`, `top_p`, `top_k`, `seed` | | Sampling controls, default `0.85` / `0.9` / `20` / `0`. `temperature = 0` is greedy. |

Output: `prompt` — also shown inline on the node.

### Default system prompt

The default (`tokenizer.DEFAULT_SYSTEM`) asks for the expanded prompt as one
block of complete sentences: it keeps every adjective and attribute from the
input, avoids comma-separated tag lists and markdown, elaborates on lighting,
atmosphere, texture, colour, composition and mood, and returns the prompt text
only — no preamble or quotes.

### Model discovery

Checkpoint folders are collected from, in order:

1. `ComfyUI/models/LLM`
2. every path in the `QWEN35_MODEL_PATHS` environment variable (`os.pathsep`
   separated — `;` on Windows, `:` elsewhere; quotes are stripped)
3. `EXTRA_MODEL_ROOTS` in `nodes.py` (currently `D:\models`)

A folder qualifies if its `config.json` (or its nested `text_config`) has a
`qwen3_5*` / `qwen3.5*` `model_type` with `hidden_size` and `layer_types`, and
it contains at least one `.safetensors`. The combo label is the folder name,
prefixed with the parent folder when two roots contain the same name; a model
that is selected by an explicit directory path is also resolved directly. Point
`EXTRA_MODEL_ROOTS` at wherever your checkpoints live.

### Memory behaviour

* `offload_after_run` — weights are copied to **pinned host memory** and VRAM is
  released (`torch.cuda.empty_cache`). A following run reloads from pinned RAM,
  which is far faster than re-reading the safetensors file.
* `keep_loaded` — the engine stays on the GPU between runs.
* Only one engine is kept resident; selecting a different model offloads the
  previous one first. `comfy.model_management.free_memory` is called before a
  load so ComfyUI can evict models as needed, and the load is retried after a
  second, larger eviction if it still runs out of VRAM.
* Note that `keep_loaded` with the 2B model leaves only ~0.8 GB free on a 4 GB
  card, so a diffusion model may not fit alongside it. Prefer
  `offload_after_run` when generating images after enhancing.

## Checkpoints

Two quantisation formats are supported, detected from the tensor names:

| | **W8A16** | **W4A16 groupwise** |
|---|---|---|
| weights | `<proj>.qweight` int8 `[in, out]` | `<proj>.packed_weight` uint8 `[out, in//2]` |
| weight scale | `<proj>.scale` bf16 `[out]` | `<proj>.scale` bf16 `[out, in//group]` |
| unpack | `w * scale` | `(nibble - 8) * scale`, even K in the low nibble |
| example | [Qwen3.5-0.8B W8A16](https://huggingface.co/Mlboy23/qwen3.5-0.8b-w8a16) | [Qwen3.5-2B W4A16](https://huggingface.co/Mlboy23/Qwen3.5-2B-W4A16) |

The format is picked from the tensor names — any `.packed_weight` key means
W4A16, otherwise `.qweight` means W8A16 — so a checkpoint needs no extra flag.
For W4 the group size is read from `quantization_config.json` (`group_size`,
default `128`), and every linear input dimension must be a multiple of it.

Both formats keep the weights in their checkpoint encoding and quantise
**activations** dynamically, so every weight matmul is
`tl.dot(int8, int8) → int32` on native int8 tensor cores; only the epilogue is
float. The W4 path never materialises a dequantised weight.

## Engine

### Activation quantisation

One scale per **64-element group** along the reduction axis (`GROUP = 64`,
matching the GEMM `BLOCK_K`). Each int8 dot is dequantised with its own scale and
accumulated in fp32.

Per-token (single-scale) quantisation was the original design, but a single
outlier then crushes the precision of the other 63 values in the row. On the
0.8B that cost a full point of top-1 agreement. Group-wise scales are strictly
better *and* faster (each `tl.dot` no longer has a loop-carried int32
dependency):

| metric (teacher-forced, 94 positions) | per-token scale | per-64-group scale |
|---|---|---|
| top-1 agreement vs reference | 89.4 % | **94.7 %** |
| mean logit cosine | 0.9896 | **0.9982** |
| mean KL(ref‖engine) | 0.0422 | **0.0093** |
| ref top-1 inside engine top-5 | — | **100 %** |
| decode | 23.1 ms/token | **13.5 ms/token** |

### W4A16: int4 weights, int8 matmul

The 4-bit checkpoint is kept packed in VRAM — 685 MB for the 2B instead of
1.37 GB if the nibbles were expanded to int8, which matters on a 4 GB card. The
nibbles are expanded to int8 **inside** the dot loop, so the matmul still runs on
the int8 tensor cores and the checkpoint's 4-bit grid is preserved exactly: this
path adds no rounding beyond the checkpoint's own.

Two details make it fast:

* **K-major layout.** At load time `packed_weight [N, K//2]` is transposed to
  `[K//2, N]` and the group scales `[N, K//WG]` to `[K//WG, N]`, so one K-block
  of a tile is a fully coalesced run along N. Storing them the checkpoint way and
  transposing in registers cost 40 % of the achievable bandwidth.
* **No shuffles in the inner loop.** The low nibble *is* the even-K weight, so
  the activation's even/odd split is free (`tl.reshape` + `tl.split`) and the
  block is summed with two plain int8 dots. Splitting the K sum in two changes
  nothing numerically: int32 accumulation is exact and both halves share the same
  activation scale and weight scale group.

Measured effect on the decode GEMMs (RTX 2050, ~105 GB/s achievable):

| kernel | before | after |
|---|---|---|
| `mlp_gate_up_quant_w4` | 285 µs | 144 µs |
| `proj3_w4` | 154 µs | 87 µs |
| `gemm_w4` | 114 µs | 46 µs |

### Fused kernels

| kernel | fuses |
|---|---|
| `rmsnorm_quant` | RMSNorm + group quantisation |
| `proj3` / `proj3_w4` | three projections sharing one activation tile (q/k/v, or qkv+z) |
| `mlp_gate_up_quant` / `mlp_gate_up_quant_w4` | gate + up GEMMs, SiLU·mul, and quantisation of the down-proj input |
| `norm_gated_quant` | gated RMSNorm + quantisation of the out-proj input |
| `gate_mul_quant` | attention output gate + quantisation |
| `qk_norm_rope` | q/k RMSNorm + partial RoPE + KV-cache write |
| `delta_prefill` / `delta_decode` | gated delta-rule (linear attention) |
| `gemm_i8` / `gemm_w4` | int8 GEMM with optional fused residual add |

The W4 kernels are selected by dtype (`uint8` weights → W4), so `engine.py`
calls the same functions for both formats.

### Layout notes

* Weights are stacked per role with the layer index on dim 0
  (`lin_qkv[18]`, `mlp_gate[24]`, …). W8 `qweight` is already stored
  `[in_features, out_features]` — exactly the GEMM layout; W4 tensors are
  transposed to K-major as described above.
* `embed_tokens` stays bf16 (exact gather); `lm_head` is quantised to
  `[hidden, vocab]` int8 with a per-vocabulary-row scale, halving logits read
  bandwidth. Both models tie it to `embed_tokens`.
* Attention QK/PV products stay bf16/fp32 — only weight matmuls are int8.
* Everything shape-dependent comes from `config.json` — head counts, head dim,
  rope partial factor, hidden/intermediate size, layer mix — and is passed to
  the kernels as arguments, so nothing is hardcoded to the 0.8B. The same code
  runs the 0.8B (1024/3584) and the 2B (2048/6144).
* RoPE is partial: `head_dim × partial_rotary_factor` dimensions rotate
  (`64` for both models), with `rope_theta` read from `rope_parameters`. Both
  default to `0.25` and `1e7` when the config omits them.
* KV cache `[layers, max_seq, 2·kv_heads·head_dim]` bf16, delta state
  `[18, 16, 128, 128]` fp32, conv state `[18, 3, 6144]` bf16.
  The KV cache grows on demand for `max_tokens`.

## Measured performance

RTX 2050 4 GB (sm_86, 16 SMs, ~105 GB/s), ComfyUI embedded Python 3.13,
torch 2.11+cu130, triton 3.7. 400-token prefill, warm.

| | 0.8B W8A16 | 2B W4A16 |
|---|---|---|
| weights | 1.26 GB | 2.24 GB |
| VRAM (engine loaded) | 1.52 GB | 2.77 GB |
| prefill | 601 tok/s | 544 tok/s |
| decode | 17.6 ms/token (57 tok/s) | 16.1 ms/token (62 tok/s) |
| 200-token prompt, warm | ~2.7 s | ~3.4 s |
| 200-token prompt, cold | ~4–5 s | ~5–6 s |

The 2B costs almost nothing extra: it reads fewer weight bytes per token
(685 MB of packed nibbles + 508 MB of int8 `lm_head`, versus 1.26 GB for the
0.8B) and decode is dispatch-bound anyway.

### Why decode is dispatch-bound, not bandwidth-bound

One 2B decode step is ~255 Triton launches: 16.7 ms of CPU enqueue but only
13.3 ms of GPU work, of which 4.7 ms is the `lm_head` read alone. Decode
throughput is therefore set by launch count (~56 µs each, Python 3.13 on
Windows), not by memory bandwidth. Consequences:

* The workspace view is cached as a plain `SimpleNamespace` (a lazy
  `__getattr__` cost ~4 µs × 896 accesses/token).
* Quantisation passes were fused into their producing kernels, removing 42
  launches.
* Decode tiles stay narrow (`BLOCK_N = 64`) so the 16 SMs get several blocks
  each.

The next step for a large win would be CUDA-graph capture of the decode step
(the per-token position argument is currently a Python int, so this needs the
position passed as a device tensor) or a single full-model decode megakernel.

## Files

```
engine.py        Engine: weights, workspace, prefill/decode/generate, offload
weights.py       safetensors -> stacked tensors, format detection, model discovery
tokenizer.py     chat tokenisation via the checkpoint's own Jinja template
sampler.py       greedy / temperature / top-k / top-p
nodes.py         the ComfyUI node, model cache, VRAM handling
kernels/         Triton kernels (gemm, w4, norm, attention, linear_attn, common)
.dev/            parity tests, benchmarks, profilers (git-ignored, dev only)
```

## Tests

The harness lives in `.dev/` (git-ignored, so only present in a development
checkout — the node itself needs none of it). Run it with the embedded
interpreter:

```powershell
$py = "D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
& $py test_kernels.py      # per-kernel parity vs torch reference (W8 path)
& $py test_fused.py        # fused kernels vs unfused composition (bit-exact)
& $py test_w4.py           # W4 kernels vs torch reference (packed int4)
& $py verify_w4_load.py    # loaded tensors bit-identical to the checkpoint
& $py test_quality.py ref  # dump HF reference logits (once)
& $py test_quality.py check# teacher-forced agreement / cosine / KL
& $py parity_w4.py         # 2B: per-position + per-decode-step parity vs HF
& $py bench_both.py        # prefill + decode throughput, both models
& $py bench_e2b.py [dir]   # per-kernel profile of one decode step
& $py test_node.py         # node end-to-end, both models, both memory modes
& $py test_api.py          # same, through a live ComfyUI /prompt API
```

All kernels match the torch reference to bf16 rounding (`rel_l2 ≈ 1.7e-3`);
`rmsnorm_quant`, `gate_proj`, `conv_state_update` and the delta-rule state are
bit-exact.

The 2B W4 engine was checked against an HF `Qwen3_5ForCausalLM` built by
dequantising the same checkpoint on the CPU: mean logit cosine **0.9986** over
all 42 prompt positions, cosine ≥ 0.9993 on 8 teacher-forced decode steps, and
the **same argmax token at every step**.
