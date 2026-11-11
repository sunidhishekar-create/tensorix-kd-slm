"""Shared helpers for the Gemma 4 sequence-level distillation experiment.

The prompt format is plain text (no special chat tokens), so the pretrained student sees nothing it
has never met. A training example is: build_prompt(...) + answer + EOS (id 0). Loss is taken on the
answer and the EOS only.
"""
import json
from pathlib import Path

MAX_LEN = 256          # student context length in tokens (prompt + answer + EOS must fit)
EOS_ID = 0             # <|endoftext|> in the SmolLM2 tokenizer
TOKENIZER = "HuggingFaceTB/SmolLM2-360M"


def build_prompt(instruction, context=""):
    """Text the student sees before the answer. Used identically for human and Gemma answers."""
    instruction, context = instruction.strip(), (context or "").strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def gemma_request(instruction, context="", max_words=80):
    """Text sent to Gemma. The style rules sit in the prompt itself (no separate system instruction)."""
    rules = (f"Answer the following instruction helpfully and accurately in plain text, "
             f"in at most {max_words} words. Do not use markdown. Do not start with a greeting or "
             f"phrases like 'Sure' or 'Here is'. Give only the answer.")
    body = f"Instruction: {instruction.strip()}"
    if (context or "").strip():
        body += f"\n\nContext: {context.strip()}"
    return f"{rules}\n\n{body}"


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
