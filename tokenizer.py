"""Chat tokenisation for the Qwen3.5 prompt-enhancer node.

Uses the `tokenizers` fast tokenizer shipped with the checkpoint plus the model's
own Jinja chat template, so the exact prompt format matches the reference model.
"""
import json
import os

from jinja2 import Template
from tokenizers import Tokenizer

DEFAULT_SYSTEM = (
    """You are an expert image prompt generation system. Your sole task is to take the user's input and expand it into a detailed, descriptive prompt using complete, grammatically correct sentences.

    Rules & Guidelines:
    1. Always retain and rich-up all original adjectives, adverbs, and key attributes provided by the user. Do not shorten, truncate, or simplify the input.
    2. Formulate the expanded prompt as fluid, fully constructed sentences. Do not use comma-separated tag lists, short phrases, or bullet points.
    3. Elaborate naturally on visual details such as lighting, atmosphere, textures, colors, composition, and mood while maintaining sentence structure.
    4. Output ONLY the final expanded caption text. Do not include introductory remarks, explanations, conversational filler, conversational greetings, quotes, tags, or markdown headers."""
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
