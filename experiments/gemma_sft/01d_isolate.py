"""Isolate why requests fail with HTTP 500 even with thinking off. Needs GEMINI_API_KEY.

Run from the repo root after 02_make_prompts.py. Uses the first real prompt from the prompt set.
"""
import os, time
from google import genai
from google.genai import types
from common import gemma_request, read_jsonl

MODEL = "gemma-4-26b-a4b-it"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
first = read_jsonl("data/sft/prompts.jsonl")[0]
simple = gemma_request("What is a credit card?")
real = gemma_request(first["instruction"], first["context"])
minimal = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

def cfg(temperature=None):
    kw = dict(max_output_tokens=1024, thinking_config=minimal)
    if temperature is not None:
        kw["temperature"] = temperature
    return types.GenerateContentConfig(**kw)

tests = [
    ("D: simple prompt, MINIMAL, temperature 0.7", simple, cfg(0.7)),
    ("E: real first prompt, MINIMAL, no temperature", real, cfg()),
    ("F: real first prompt, MINIMAL, temperature 0.7", real, cfg(0.7)),
]
for name, text, config in tests:
    t0 = time.time()
    try:
        r = client.models.generate_content(model=MODEL, contents=text, config=config)
        print(f"{name}\n  OK in {time.time() - t0:.0f}s | finish {r.candidates[0].finish_reason}"
              f"\n  answer: {(r.text or '')[:120]!r}\n")
    except Exception as e:
        print(f"{name}\n  FAILED after {time.time() - t0:.0f}s: {str(e)[:120]}\n")
