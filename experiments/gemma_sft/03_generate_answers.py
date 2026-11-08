"""Generate Gemma 4 answers for the fixed prompt set through the Gemini API.

Resumable: every answer is appended to the output file as soon as it arrives, and prompts that are
already in the file are skipped on the next run. Rerun the same command to continue after a quota
stop, a crash or a closed laptop.

  python experiments/gemma_sft/03_generate_answers.py --limit 5        # first try: 5 prompts
  python experiments/gemma_sft/03_generate_answers.py --rpm 10 --daily_limit 1000

Needs GEMINI_API_KEY in the environment. Never put the key in a file.
"""
import argparse, os, time
from pathlib import Path

from common import append_jsonl, gemma_request, read_jsonl

DEFAULT_MODEL = "gemma-4-26b-a4b-it"     # gemma-4-31b-it returned HTTP 500 during the first test
SLEEP = time.sleep                        # replaced in tests


def make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_config(temperature, thinking, max_tokens=1024):
    """Gemma 4 thinks by default, and thinking tokens count toward max_output_tokens.
    thinking_level MINIMAL switches thinking off (thinking_budget is rejected for Gemma 4)."""
    from google.genai import types
    kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
    if thinking == "minimal":
        if not hasattr(types, "ThinkingLevel"):
            raise SystemExit("google-genai is too old for thinking_level. Run: pip install -U google-genai")
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    return types.GenerateContentConfig(**kwargs)


def generate(client, model, text, config):
    r = client.models.generate_content(model=model, contents=text, config=config)
    reason = None
    try:
        reason = str(r.candidates[0].finish_reason)
    except Exception:
        pass
    thoughts = getattr(getattr(r, "usage_metadata", None), "thoughts_token_count", None)
    return {"answer": (r.text or "").strip(), "finish_reason": reason, "thoughts_tokens": thoughts}


def ask(client, model, text, config):
    """Returns a result dict, "QUOTA" (stop the run), or None (skip this prompt for now)."""
    delays = [5, 15, 45, 120]
    quota_hits = 0
    attempt = 0
    while True:
        try:
            return generate(client, model, text, config)
        except Exception as e:                                   # noqa: BLE001
            code = getattr(e, "code", None)
            msg = str(e)[:150]
            if code in (401, 403):
                raise SystemExit(f"API key problem ({code}): {msg}")
            if code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                quota_hits += 1
                if quota_hits >= 3:
                    return "QUOTA"
                print("  rate limited (429), waiting 60 s", flush=True)
                SLEEP(60)
                continue
            if (code is None or code >= 500) and attempt < len(delays):
                print(f"  server error ({code}), retry in {delays[attempt]} s", flush=True)
                SLEEP(delays[attempt])
                attempt += 1
                continue
            print(f"  giving up on this prompt for now: {msg}", flush=True)
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/sft/prompts.jsonl")
    ap.add_argument("--out", default="data/sft/gemma_answers.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rpm", type=float, default=10, help="requests per minute to stay under")
    ap.add_argument("--daily_limit", type=int, default=1000, help="max successful requests in this run")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_words", type=int, default=80)
    ap.add_argument("--max_output_tokens", type=int, default=1024,
                    help="raise (for example 4096) when --thinking default is used")
    ap.add_argument("--thinking", choices=["minimal", "default"], default="minimal",
                    help="minimal = thinking off (recommended); default = model's own setting")
    ap.add_argument("--limit", type=int, default=0, help="only process this many prompts (0 = all)")
    a = ap.parse_args()

    rows = read_jsonl(a.prompts)
    assert rows, f"no prompts found in {a.prompts}. Run 02_make_prompts.py first."
    done = {r["id"] for r in read_jsonl(a.out)}
    todo = [r for r in rows if r["id"] not in done]          # file order: train first, then eval
    if a.limit:
        todo = todo[:a.limit]
    print(f"model {a.model} | prompts {len(rows)} | already done {len(done)} | this run {len(todo)} "
          f"| about {len(todo) / a.rpm:.0f} min at {a.rpm:g} requests/min", flush=True)

    client = make_client()
    config = build_config(a.temperature, a.thinking, a.max_output_tokens)
    interval = 60.0 / a.rpm
    last_start, made, skipped, stopped_by_quota = 0.0, 0, 0, False
    for n, row in enumerate(todo, 1):
        if made >= a.daily_limit:
            print(f"daily limit of {a.daily_limit} requests reached for this run.")
            break
        wait = interval - (time.time() - last_start)
        if wait > 0:
            SLEEP(wait)
        last_start = time.time()

        res = ask(client, a.model, gemma_request(row["instruction"], row["context"], a.max_words),
                  config)
        if res == "QUOTA":
            print("quota reached (repeated 429). Progress is saved: rerun later to continue.")
            stopped_by_quota = True
            break
        if res is None:
            skipped += 1
            continue
        if not res["answer"]:                      # empty text: do not save, so a rerun retries it
            print(f"  empty answer for id {row['id']} (finish {res['finish_reason']}, "
                  f"thinking tokens {res['thoughts_tokens']}); not saved", flush=True)
            skipped += 1
            continue
        append_jsonl(a.out, {"id": row["id"], "model": a.model, "temperature": a.temperature,
                             "thinking": a.thinking, "answer": res["answer"],
                             "finish_reason": res["finish_reason"], "thoughts_tokens": res["thoughts_tokens"],
                             "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        made += 1
        if made % 25 == 0 or made == len(todo):
            print(f"  {made} answers this run ({n}/{len(todo)} prompts tried)", flush=True)

    total_done = len(read_jsonl(a.out))
    print(f"\nthis run: {made} saved, {skipped} skipped | total saved {total_done}/{len(rows)}")
    if total_done < len(rows):
        print("Rerun the same command later to continue.")


if __name__ == "__main__":
    main()
