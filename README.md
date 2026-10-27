# Qwen3.5 Prompt Enhancer (INT8 Triton)

A ComfyUI custom node that turns a short idea into a detailed text-to-image
prompt, using a local **Qwen3.5-0.8B** checkpoint driven by a hand-written
**INT8 Triton** inference engine.

Everything runs on the embedded Python that ships with ComfyUI — no extra
packages, no server, no API keys.

```
STRING (short idea) ──► [Qwen3.5 Prompt Enhancer] ──► STRING (detailed prompt)
```

## Usage

1. Clone this repository into ComfyUI's `custom_nodes` directory:

   ```bash
   git clone https://github.com/Unica2804/Qwen_3.5_Comfy_Prompt_Enahancer.git \
     ComfyUI/custom_nodes/qwen35_prompt_enhancer
   ```

2. Download the W8A16 quantized model from
   [Hugging Face](https://huggingface.co/Mlboy23/qwen3.5-0.8b-w8a16) into a
   directory named `qwen` under `ComfyUI/models/LLM`:

   ```bash
   huggingface-cli download Mlboy23/qwen3.5-0.8b-w8a16 \
     --local-dir ComfyUI/models/LLM/qwen
   ```

   Alternatively, download the files from the Hugging Face page and place all
   of them in `ComfyUI/models/LLM/qwen`.

3. Restart ComfyUI. The node is available under the `text` category as
   `Qwen3.5 Prompt Enhancer`.

## Node

Category: `text/Qwen3.5`

| Input | Type | Notes |
|---|---|---|
| `text` | STRING (multiline) | The idea / keyword list to expand. |
| `model` | combo | Any Qwen3.5 checkpoint folder under `ComfyUI/models/LLM`. |
| `memory_mode` | combo | `offload_after_run` (frees VRAM after every run) or `keep_loaded` (resident, faster repeats). |
| `max_tokens` | combo | Generation cap: `512` / `1024` / `2048` / `4096`. |
| `system_prompt` | STRING (multiline) | Defines the enhancer's style. |
| `temperature`, `top_p`, `top_k`, `seed` | | Sampling controls. `temperature = 0` is greedy. |

Output: `prompt` — also shown inline on the node.

### Memory behaviour

* `offload_after_run` — weights are copied to **pinned host memory** and VRAM is
  released (`torch.cuda.empty_cache`). A following run reloads from pinned RAM,
  which is far faster than re-reading the safetensors file.
* `keep_loaded` — the engine stays on the GPU between runs.
* Only one engine is kept resident; selecting a different model offloads the
  previous one first. `comfy.model_management.free_memory` is called before a
  load so ComfyUI can evict models as needed.

## Engine

The model is `Qwen3.5-0.8B` in a **W8A16** checkpoint (int8 weights, per-output-channel
bf16 scales). The engine keeps the weights int8 and quantises **activations**
dynamically, so every weight matmul is `tl.dot(int8, int8) → int32` on native
int8 tensor cores; only the epilogue is float.

### Activation quantisation

One scale per **64-element group** along the reduction axis (`GROUP = 64`,
matching the GEMM `BLOCK_K`). Each int8 dot is dequantised with its own scale and
accumulated in fp32.

Per-token (single-scale) quantisation was the original design, but a single
outlier then crushes the precision of the other 63 values in the row. On this
model that cost a full point of top-1 agreement. Group-wise scales are strictly
better *and* faster (each `tl.dot` no longer has a loop-carried int32
dependency):

| metric (teacher-forced, 94 positions) | per-token scale | per-64-group scale |
|---|---|---|
| top-1 agreement vs reference | 89.4 % | **94.7 %** |
| mean logit cosine | 0.9896 | **0.9982** |
| mean KL(ref‖engine) | 0.0422 | **0.0093** |
| ref top-1 inside engine top-5 | — | **100 %** |
| decode | 23.1 ms/token | **13.5 ms/token** |

### Fused kernels

| kernel | fuses |
|---|---|
| `rmsnorm_quant` | RMSNorm + group quantisation |
| `proj3` | three projections sharing one activation tile (q/k/v, or qkv+z) |
| `mlp_gate_up_quant` | gate + up GEMMs, SiLU·mul, and quantisation of the down-proj input |
| `norm_gated_quant` | gated RMSNorm + quantisation of the out-proj input |
| `gate_mul_quant` | attention output gate + quantisation |
| `qk_norm_rope` | q/k RMSNorm + partial RoPE + KV-cache write |
| `delta_prefill` / `delta_decode` | gated delta-rule (linear attention) |
| `gemm_i8` | int8 GEMM with optional fused residual add |

### Layout notes

* Weights are stacked per role with the layer index on dim 0
  (`lin_qkv[18]`, `mlp_gate[24]`, …), and `qweight` is already stored
  `[in_features, out_features]` — exactly the GEMM layout, so no repacking.
* `embed_tokens` stays bf16 (exact gather); `lm_head` is quantised to
  `[hidden, vocab]` int8 with a per-vocabulary-row scale, halving logits
  read bandwidth.
* Attention QK/PV products stay bf16/fp32 — only weight matmuls are int8.
* KV cache `[layers, max_seq, 2·kv_heads·head_dim]` bf16, delta state
  `[18, 16, 128, 128]` fp32, conv state `[18, 3, 6144]` bf16.
  The KV cache grows on demand for `max_tokens`.

## Measured performance

RTX 2050 4 GB (sm_86, 16 SMs), ComfyUI embedded Python 3.13, torch 2.11+cu130,
triton 3.7.

| | |
|---|---|
| weights | 1.26 GB |
| total VRAM (engine loaded) | 1.51 GB |
| prefill | ~600 tok/s |
| decode | **~74 tok/s** (13.5 ms/token) |
| 200-token prompt, cold start | ~4–5 s |
| 200-token prompt, warm | ~2.7 s |

### Why decode is dispatch-bound, not bandwidth-bound

Profiling one decode step showed **19.5 ms of CPU enqueue time but only 0.05 ms
of GPU work still queued** afterwards, spread over 296 Triton launches
(~18 µs each, Python 3.13 on Windows). Decode throughput is therefore set by
launch count, not by memory bandwidth. Two consequences:

* The workspace view is cached as a plain `SimpleNamespace` (a lazy
  `__getattr__` cost ~4 µs × 896 accesses/token).
* Quantisation passes were fused into their producing kernels, removing 42
  launches.

The remaining ~236 launches are the main limit. The next step for a large win
would be a single full-model decode megakernel (grid = 16 blocks, monotonic
counter grid barrier, ~0.65 µs/barrier) or CUDA-graph capture of the decode
step; both target the same overhead.

## Files

```
engine.py        Engine: weights, workspace, prefill/decode/generate, offload
weights.py       safetensors -> stacked int8 tensors, model discovery
tokenizer.py     chat tokenisation via the checkpoint's own Jinja template
sampler.py       greedy / temperature / top-k / top-p
nodes.py         the ComfyUI node, model cache, VRAM handling
kernels/         Triton kernels (gemm, norm, attention, linear_attn, common)
```

All kernels match the torch reference to bf16 rounding (`rel_l2 ≈ 1.7e-3`);
`rmsnorm_quant`, `gate_proj`, `conv_state_update` and the delta-rule state are
bit-exact.
