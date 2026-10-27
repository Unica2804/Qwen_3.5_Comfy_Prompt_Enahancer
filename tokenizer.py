"""Chat tokenisation for the Qwen3.5 prompt-enhancer node.

Uses the `tokenizers` fast tokenizer shipped with the checkpoint plus the model's
own Jinja chat template, so the exact prompt format matches the reference model.
"""
import json
import os

from jinja2 import Template
from tokenizers import Tokenizer

DEFAULT_SYSTEM = (
    "You are a prompt engineer. Rewrite the user's short idea into one rich, "
    "detailed text-to-image prompt. Describe subject, composition, lighting, "
    "colours, mood and style in flowing prose. Reply with the prompt only."
)

# Tokens that end a generation turn.
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


class ChatTokenizer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tmpl_path = os.path.join(model_dir, "chat_template.jinja")
        with open(tmpl_path, encoding="utf-8") as f:
            self.template = Template(f.read())
        self.stop_ids = tuple(
            self.tok.token_to_id(s) for s in STOP_STRINGS
            if self.tok.token_to_id(s) is not None
        )

    def render_prompt(self, system_prompt, user_text, enable_thinking=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def encode_chat(self, system_prompt, user_text, enable_thinking=False):
        text = self.render_prompt(system_prompt, user_text, enable_thinking)
        return self.tok.encode(text).ids

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def decode_stream(self, ids):
        """Decode a token list for progressive display (special tokens stripped)."""
        return self.tok.decode(ids, skip_special_tokens=True)
