"""Zero-shot multiple-choice benchmarks (HellaSwag, ARC-Easy) for a student checkpoint or a HF model.

Scoring follows the lm-evaluation-harness convention: for every answer choice, sum the log-probability
of the choice tokens given the question. `acc` picks the choice with the highest sum; `acc_norm`
divides each sum by the choice's length in characters first (removes the bias toward short answers).

  student: python src/bench_mc.py --ckpt <run>/best.pt --name baseline_short --out_dir results/bench
  teacher: python src/bench_mc.py --hf_model HuggingFaceTB/SmolLM2-360M --name teacher_base --out_dir results/bench
  quick  : add --limit 50

Every model gets the identical examples (the full validation/test split by default) and the identical
tokenization, so the numbers are directly comparable. Random-guess accuracy is about 25% on both tasks;
a 51M model trained on ~0.1-0.3B tokens is expected to sit only a few points above that, so read the
standard errors before calling a difference real.
"""
import argparse, json, math, re, time
from pathlib import Path

import torch
import torch.nn.functional as F

TOKENIZER = "HuggingFaceTB/SmolLM2-360M"


# ---------------- task loading (prompt formats as in lm-evaluation-harness) ----------------
def _hs_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(limit=0):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for r in ds:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        out.append({"query": _hs_clean(r["activity_label"] + ": " + ctx),
                    "choices": [" " + _hs_clean(e) for e in r["endings"]],
                    "label": int(r["label"])})
        if limit and len(out) >= limit:
            break
    return out


def load_arc_easy(limit=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    out = []
    for r in ds:
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        out.append({"query": f"Question: {r['question'].strip()}\nAnswer:",
                    "choices": [" " + t.strip() for t in r["choices"]["text"]],
                    "label": labels.index(r["answerKey"])})
        if limit and len(out) >= limit:
            break
    return out


TASKS = {"hellaswag": load_hellaswag, "arc_easy": load_arc_easy}


# ---------------- scoring ----------------
def build_requests(examples, tok, block_size):
    """One request per (example, choice): token ids, and where the choice tokens start."""
    reqs = []
    for ei, ex in enumerate(examples):
        q_ids = tok(ex["query"], add_special_tokens=False)["input_ids"]
        for ci, ch in enumerate(ex["choices"]):
            c_ids = tok(ch, add_special_tokens=False)["input_ids"][:block_size - 1]
            ctx = q_ids[-(block_size - len(c_ids)):]          # keep the end of the question if too long
            reqs.append({"ex": ei, "choice": ci, "ids": ctx + c_ids, "start": len(ctx),
                         "chars": len(ch)})
    return reqs


@torch.no_grad()
def score(reqs, logits_fn, device, batch_size):
    """Sum of log p(choice tokens | question) for every request (fp32 log-softmax)."""
    order = sorted(range(len(reqs)), key=lambda i: -len(reqs[i]["ids"]))   # similar lengths per batch
    ll = [0.0] * len(reqs)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        T = max(len(reqs[i]["ids"]) for i in idx)
        x = torch.zeros(len(idx), T, dtype=torch.long)                  # right padding: causal, so harmless
        for r, i in enumerate(idx):
            x[r, :len(reqs[i]["ids"])] = torch.tensor(reqs[i]["ids"])
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = logits_fn(x.to(device))
        for r, i in enumerate(idx):
            ids, s = reqs[i]["ids"], reqs[i]["start"]
            lp = F.log_softmax(logits[r, s - 1:len(ids) - 1].float(), dim=-1)
            tgt = torch.tensor(ids[s:], device=lp.device)
            ll[i] = lp.gather(1, tgt[:, None]).sum().item()
    return ll


def accuracy(examples, reqs, ll):
    per_ex = {}
    for q, v in zip(reqs, ll):
        per_ex.setdefault(q["ex"], []).append((q["choice"], v, v / max(1, q["chars"])))
    n = len(examples)
    hit = sum(max(per_ex[e], key=lambda t: t[1])[0] == examples[e]["label"] for e in range(n))
    hit_n = sum(max(per_ex[e], key=lambda t: t[2])[0] == examples[e]["label"] for e in range(n))
    acc, acc_n = hit / n, hit_n / n
    chance = sum(1 / len(ex["choices"]) for ex in examples) / n
    return {"n": n, "acc": round(acc, 4), "acc_stderr": round(math.sqrt(acc * (1 - acc) / n), 4),
            "acc_norm": round(acc_n, 4), "acc_norm_stderr": round(math.sqrt(acc_n * (1 - acc_n) / n), 4),
            "chance": round(chance, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="student checkpoint (best.pt / ckpt.pt)")
    g.add_argument("--hf_model", help="Hugging Face model id")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples per task (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=256, help="student context; HF models use the same")
    ap.add_argument("--out_dir", default="results/bench")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    info = {"name": a.name, "device": device, "block_size": a.block_size, "limit": a.limit,
            "scoring": "sum log p(choice | question); acc_norm divides by choice length in characters"}
    if a.ckpt:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model import load_checkpoint
        model, ck = load_checkpoint(a.ckpt, device)
        info.update(kind="student", source=a.ckpt, step=ck.get("step"))
        del ck
        model.eval()
        logits_fn = lambda x: model(x)[0]
    else:
        from transformers import AutoModelForCausalLM
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.hf_model, dtype=dtype).to(device).eval()
        info.update(kind="hf_model", source=a.hf_model)
        logits_fn = lambda x: model(input_ids=x).logits

    info["tasks"] = {}
    for t in a.tasks:
        t0 = time.time()
        examples = TASKS[t](a.limit)
        reqs = build_requests(examples, tok, a.block_size)
        ll = score(reqs, logits_fn, device, a.batch_size)
        info["tasks"][t] = accuracy(examples, reqs, ll)
        print(f"{a.name} | {t}: {info['tasks'][t]} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / f"{a.name}.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
