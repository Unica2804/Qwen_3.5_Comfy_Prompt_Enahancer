"""ComfyUI node: expand a short idea into a detailed prompt with a local
Qwen3.5-0.8B model running on a fused INT8 Triton kernel engine."""
import os
import threading

import torch

try:
    import comfy.model_management as mm
    import comfy.utils
except Exception:  # running outside ComfyUI (tests)
    mm = None
    comfy = None

from .engine import Engine
from .tokenizer import DEFAULT_SYSTEM, ChatTokenizer
from .weights import discover_models

_HERE = os.path.dirname(os.path.abspath(__file__))
# package -> custom_nodes -> ComfyUI
_COMFY_ROOT = os.path.dirname(os.path.dirname(_HERE))
LLM_ROOT = os.path.join(_COMFY_ROOT, "models", "LLM")
try:  # prefer ComfyUI's own resolution when available
    import folder_paths

    LLM_ROOT = os.path.join(folder_paths.models_dir, "LLM")
except Exception:
    pass

MAX_TOKEN_CHOICES = ["512", "1024", "2048", "4096"]

_cache = {}
_tok_cache = {}
_lock = threading.Lock()


def _model_dirs():
    names = discover_models(LLM_ROOT)
    return names


def _resolve_model(name):
    d = os.path.join(LLM_ROOT, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"model folder not found: {d}")
    return d


def _get_engine(model_dir):
    """Load (or reuse) an engine, keeping at most one on the GPU at a time."""
    with _lock:
        eng = _cache.get(model_dir)
        if eng is not None:
            if not eng.on_gpu:
                eng.reload()
            return eng
        for key, other in list(_cache.items()):
            if key != model_dir:
                other.offload()
                del _cache[key]
        if mm is not None:
            mm.free_memory(2_000_000_000, torch.device("cuda"))
        eng = Engine(model_dir, device="cuda", log=_log)
        _cache[model_dir] = eng
        return eng


def _get_tokenizer(model_dir):
    """Tokenizer is immutable and slow to build (20 MB vocab), so keep one per model."""
    with _lock:
        tok = _tok_cache.get(model_dir)
        if tok is None:
            tok = ChatTokenizer(model_dir)
            _tok_cache[model_dir] = tok
        return tok


def _log(msg):
    print(msg, flush=True)


class Qwen35PromptEnhancer:
    """Short text -> detailed prompt, via a local INT8 Qwen3.5 engine."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _model_dirs()
        if not models:
            models = ["(no models found in models/LLM)"]
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "a cat sitting on a windowsill",
                    "tooltip": "The short idea or keyword list to expand.",
                }),
                "model": (models, {
                    "tooltip": "Checkpoint folder under ComfyUI/models/LLM.",
                }),
                "memory_mode": (["offload_after_run", "keep_loaded"], {
                    "default": "offload_after_run",
                    "tooltip": "offload_after_run frees VRAM after every run; "
                               "keep_loaded keeps the engine resident for faster repeats.",
                }),
                "max_tokens": (MAX_TOKEN_CHOICES, {
                    "default": "1024",
                    "tooltip": "Upper bound on generated tokens.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM,
                    "tooltip": "Instruction that defines the enhancer's style.",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 0, "max": 200, "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "control_after_generate": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "enhance"
    CATEGORY = "text/Qwen3.5"
    OUTPUT_NODE = True

    def enhance(self, text, model, memory_mode, max_tokens, system_prompt,
                temperature=0.7, top_p=0.9, top_k=40, seed=0):
        text = (text or "").strip()
        if not text:
            return {"ui": {"text": [""]}, "result": ("",)}

        model_dir = _resolve_model(model)
        limit = int(max_tokens)
        eng = _get_engine(model_dir)
        tok = _get_tokenizer(model_dir)

        ids = tok.encode_chat(system_prompt or DEFAULT_SYSTEM, text)
        stop = tok.stop_ids

        pbar = comfy.utils.ProgressBar(limit) if comfy is not None else None
        state = {"n": 0}

        def on_token(_tok):
            state["n"] += 1
            if pbar is not None:
                pbar.update(1)

        try:
            with torch.inference_mode():
                out_ids = eng.generate(
                    ids,
                    max_new_tokens=limit,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    seed=int(seed),
                    stop_ids=stop,
                    on_token=on_token,
                )
            result = tok.decode(out_ids).strip()
        finally:
            if memory_mode == "offload_after_run":
                with _lock:
                    eng.offload()

        return {"ui": {"text": [result]}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer": Qwen35PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer": "Qwen3.5 Prompt Enhancer (INT8)",
}
