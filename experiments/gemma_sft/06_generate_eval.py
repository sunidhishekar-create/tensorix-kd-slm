"""Generate answers to the held-out evaluation prompts from several models, for blind comparison.

Every model gets the same prompts, the same decoding settings and the same random seed per prompt,
so differences in the answers come from the models, not from sampling luck.

  python experiments/gemma_sft/06_generate_eval.py --prompts /content/sft/prompts.jsonl \
      --out generations.jsonl --models pretrained=<baseline_full/best.pt> human=<sft_human_v2/sft.pt> gemma=<sft_gemma_v2/sft.pt>

Output: one line per prompt with the instruction, the human reference answer and each model's answer.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model import load_checkpoint                                        # noqa: E402
from common import EOS_ID, MAX_LEN, TOKENIZER, build_prompt, read_jsonl  # noqa: E402


@torch.no_grad()
def sample(model, prompt_ids, max_new, temperature, top_k, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([prompt_ids], device=device)
    new = []
    for _ in range(max_new):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits, _ = model(ids[:, -model.cfg.block_size:])
        logits = logits[0, -1].float() / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)
        tok = int(nxt)
        if tok == EOS_ID:                         # the model ended its answer
            break
        new.append(tok)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/sft/prompts.jsonl")
    ap.add_argument("--out", default="data/sft/generations.jsonl")
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs (checkpoints from train.py or 05_finetune.py)")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="only the first N eval prompts (0 = all)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    rows = [r for r in read_jsonl(a.prompts) if r["split"] == "eval"]
    if a.n:
        rows = rows[:a.n]
    prompt_ids = [tok(build_prompt(r["instruction"], r["context"]), add_special_tokens=False)["input_ids"]
                  for r in rows]
    results = {r["id"]: {} for r in rows}
    print(f"{len(rows)} eval prompts | models {[m.split('=')[0] for m in a.models]} | device {device} "
          f"| temperature {a.temperature}, top_k {a.top_k}, max_new_tokens {a.max_new_tokens}", flush=True)

    for spec in a.models:
        name, path = spec.split("=", 1)
        model, ck = load_checkpoint(path, device)
        model.eval()
        t0 = time.time()
        for n, (r, p) in enumerate(zip(rows, prompt_ids), 1):
            max_new = min(a.max_new_tokens, MAX_LEN - 1 - len(p))
            new = sample(model, p, max_new, a.temperature, a.top_k, device, a.seed + r["id"])
            results[r["id"]][name] = tok.decode(new).strip()
            if n % 50 == 0:
                print(f"  {name}: {n}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
        del model, ck
        if device == "cuda":
            torch.cuda.empty_cache()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "category": r["category"], "instruction": r["instruction"],
                                "context": r["context"], "human_reference": r["human_response"],
                                "answers": results[r["id"]]}, ensure_ascii=False) + "\n")
    json.dump({"models": a.models, "temperature": a.temperature, "top_k": a.top_k,
               "max_new_tokens": a.max_new_tokens, "seed": a.seed, "n_prompts": len(rows)},
              open(out.with_suffix(".meta.json"), "w"), indent=2)
    print(f"wrote {out}")
    for r in rows[:2]:
        print("\nINSTRUCTION:", r["instruction"])
        for name, text in results[r["id"]].items():
            print(f"  [{name}] {text[:200]}")


if __name__ == "__main__":
    main()
