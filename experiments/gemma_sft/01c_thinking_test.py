"""Find out which request settings Gemma 4 accepts. Needs GEMINI_API_KEY in the environment.

Sends the same prompt three ways and prints what came back (or the error), so the cause of empty
answers or HTTP 500 errors can be isolated.
"""
import os, time
from google import genai
from google.genai import types
from common import gemma_request

MODEL = "gemma-4-26b-a4b-it"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
text = gemma_request("What is a credit card?")

tests = {
    "A: default thinking, 1024 max tokens": types.GenerateContentConfig(max_output_tokens=1024),
    "B: thinking MINIMAL, 1024 max tokens": types.GenerateContentConfig(
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)),
    "C: default thinking, 4096 max tokens": types.GenerateContentConfig(max_output_tokens=4096),
}
for name, cfg in tests.items():
    t0 = time.time()
    try:
        r = client.models.generate_content(model=MODEL, contents=text, config=cfg)
        um = r.usage_metadata
        print(f"{name}\n  finish: {r.candidates[0].finish_reason} | thinking tokens: "
              f"{getattr(um, 'thoughts_token_count', None)} | {time.time() - t0:.0f}s"
              f"\n  answer: {(r.text or '')[:200]!r}\n")
    except Exception as e:
        print(f"{name}\n  FAILED after {time.time() - t0:.0f}s: {str(e)[:150]}\n")
