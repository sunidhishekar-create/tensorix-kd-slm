"""Build the two fine-tuning datasets (human answers vs Gemma answers) on the SAME prompts.

For every prompt that has a Gemma answer, both versions are built, and a prompt is kept only if BOTH
versions fit in 256 tokens, so the two datasets contain exactly the same prompts.

Each example is stored as two arrays of length 256, matching how train.py feeds the model:
  x : input ids   = (prompt + answer + EOS)[:-1], padded with 0
  y : target ids  = (prompt + answer + EOS)[1:],  with -100 on prompt positions and padding
-100 is ignored by the cross-entropy in model.py, so the loss is taken on the answer and EOS only.

  python experiments/gemma_sft/04_tokenize_sft.py
Output in data/sft/: human_train.npz, gemma_train.npz, human_eval.npz, gemma_eval.npz, sft_meta.json
"""
import argparse, json
from pathlib import Path

import numpy as np

from common import EOS_ID, MAX_LEN, TOKENIZER, build_prompt, read_jsonl


def pack(p_ids, a_ids):
    """One example as (x, y) arrays of length MAX_LEN."""
    s = p_ids + a_ids + [EOS_ID]
    x = np.zeros(MAX_LEN, dtype=np.uint16)
    y = np.full(MAX_LEN, -100, dtype=np.int32)
    x[:len(s) - 1] = s[:-1]
    y[len(p_ids) - 1:len(s) - 1] = s[len(p_ids):]       # answer tokens and EOS only
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/sft/prompts.jsonl")
    ap.add_argument("--answers", default="data/sft/gemma_answers.jsonl")
    ap.add_argument("--out_dir", default="data/sft")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    gemma = {r["id"]: r["answer"].strip() for r in read_jsonl(a.answers) if r.get("answer", "").strip()}
    out = Path(a.out_dir)
    meta = {"max_len": MAX_LEN, "splits": {}}
    for split in ("train", "eval"):
        rows = [r for r in read_jsonl(a.prompts) if r["split"] == split and r["id"] in gemma]
        if not rows:
            print(f"{split}: no prompts with Gemma answers yet, skipped")
            continue
        enc = lambda texts: tok(texts, add_special_tokens=False)["input_ids"]
        p_ids = enc([build_prompt(r["instruction"], r["context"]) for r in rows])
        h_ids = enc([r["human_response"].strip() for r in rows])
        g_ids = enc([gemma[r["id"]] for r in rows])

        keep = [i for i in range(len(rows))
                if len(p_ids[i]) + len(h_ids[i]) + 1 <= MAX_LEN
                and len(p_ids[i]) + len(g_ids[i]) + 1 <= MAX_LEN
                and len(h_ids[i]) > 0 and len(g_ids[i]) > 0]
        for name, ans in (("human", h_ids), ("gemma", g_ids)):
            xs, ys = zip(*(pack(p_ids[i], ans[i]) for i in keep))
            np.savez_compressed(out / f"{name}_{split}.npz", x=np.stack(xs), y=np.stack(ys),
                                ids=np.array([rows[i]["id"] for i in keep]))
        meta["splits"][split] = {
            "prompts_with_gemma_answer": len(rows), "kept": len(keep),
            "dropped_too_long": len(rows) - len(keep),
            "mean_answer_tokens_human": round(float(np.mean([len(h_ids[i]) for i in keep])), 1),
            "mean_answer_tokens_gemma": round(float(np.mean([len(g_ids[i]) for i in keep])), 1)}
    json.dump(meta, open(out / "sft_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()