"""Build the fixed prompt set from Databricks Dolly 15k.

Keeps examples whose prompt + human answer + EOS fit in the student's 256-token context, shuffles
them with a fixed seed, and writes the first n_eval as the held-out evaluation split and the next
n_train as the training split. The same prompts are used for the human-answer and Gemma-answer runs.

  python experiments/gemma_sft/02_make_prompts.py
Output: data/sft/prompts.jsonl and data/sft/prompts_meta.json (data/ is gitignored).
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

from common import MAX_LEN, TOKENIZER, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--max_prompt_tokens", type=int, default=160,
                    help="leave room for the answer inside the 256-token context")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"Dolly examples: {len(ds)}")

    prompts = [build_prompt(r["instruction"], r["context"]) for r in ds]
    p_len = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
    r_len = [len(x) for x in tok([r["response"] for r in ds], add_special_tokens=False)["input_ids"]]

    eligible = []
    for i, r in enumerate(ds):
        ok = (r["response"].strip() != "" and p_len[i] <= a.max_prompt_tokens
              and p_len[i] + r_len[i] + 1 <= MAX_LEN)          # +1 for EOS
        if ok:
            eligible.append(i)
    print(f"eligible (fit in {MAX_LEN} tokens): {len(eligible)}")

    need = a.n_train + a.n_eval
    assert len(eligible) >= need, f"only {len(eligible)} eligible, need {need}"
    random.Random(a.seed).shuffle(eligible)
    chosen = {"eval": eligible[:a.n_eval], "train": eligible[a.n_eval:need]}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = {}
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for split in ("train", "eval"):          # train first, so a partial run covers training prompts
            cats[split] = Counter()
            for i in chosen[split]:
                r = ds[i]
                cats[split][r["category"]] += 1
                f.write(json.dumps({"id": i, "split": split, "category": r["category"],
                                    "instruction": r["instruction"], "context": r["context"],
                                    "human_response": r["response"]}, ensure_ascii=False) + "\n")

    meta = {"source": "databricks/databricks-dolly-15k", "seed": a.seed, "max_len": MAX_LEN,
            "max_prompt_tokens": a.max_prompt_tokens, "eligible": len(eligible),
            "n_train": a.n_train, "n_eval": a.n_eval,
            "categories": {k: dict(v) for k, v in cats.items()}}
    json.dump(meta, open(out / "prompts_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
