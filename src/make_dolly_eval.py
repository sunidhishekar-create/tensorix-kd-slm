"""Build the held-out Dolly prompt set for the baseline-versus-distilled comparison.

Rebuilds the validation split of the Dolly SFT runs (sft_dolly_baseline_v3, sft_dolly_baseline_short_v1,
sft_dolly_kd_short_T2_v1) with exactly the notebook's rules: the v3 prompt template, examples whose
prompt + response + EOS exceed 256 tokens dropped, and the seed-1337 hash split (about 5% to validation).
None of these prompts was used for fine-tuning any of the three models.

  python src/make_dolly_eval.py --out results/final/dolly_eval_prompts.jsonl --n 300 \
      --ref_split /content/drive/MyDrive/tensorix/v2/checkpoints/sft_dolly_baseline_v3/split.json

With --ref_split the script stops unless its validation rows are identical to the ones saved by the
fine-tuning run. Output format is the one experiments/gemma_sft/06_generate_eval.py reads
(id, split="eval", category, instruction, context, human_response).
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

TOKENIZER = "HuggingFaceTB/SmolLM2-360M"
EOS_ID, MAX_LENGTH, SEED, VAL_FRACTION = 0, 256, 1337, 0.05


def build_prompt(instruction, context=""):
    """Identical to Section 7 of the Dolly SFT notebooks (template v3)."""
    p = "### Instruction:\n" + instruction.strip() + "\n\n"
    if context and context.strip():
        p += "### Context:\n" + context.strip() + "\n\n"
    return p + "### Response:\n"


def split_of(row):
    key = f"{SEED}|{row['instruction'].lower()}|{row['context'].lower()}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)
    return "val" if h < VAL_FRACTION else "train"


def fits(row, tok):
    if not row["instruction"] or not row["response"]:
        return False
    p = tok(build_prompt(row["instruction"], row["context"]), add_special_tokens=False)["input_ids"]
    r = tok(row["response"], add_special_tokens=False)["input_ids"]
    return bool(r) and EOS_ID not in p and EOS_ID not in r and len(p) + len(r) + 1 <= MAX_LENGTH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final/dolly_eval_prompts.jsonl")
    ap.add_argument("--n", type=int, default=300, help="random sample of the validation split (0 = all)")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--ref_split", default="", help="split.json written by a Dolly SFT run (recommended)")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    raw = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [{"row_id": i, "instruction": (r["instruction"] or "").strip(), "context": (r["context"] or "").strip(),
             "response": (r["response"] or "").strip(), "category": (r["category"] or "unknown").strip()}
            for i, r in enumerate(raw)]
    val = [r for r in rows if split_of(r) == "val" and fits(r, tok)]
    print(f"Dolly rows {len(rows)} | validation rows that fit in {MAX_LENGTH} tokens: {len(val)}")

    if a.ref_split:
        ref = set(json.load(open(a.ref_split))["val_row_ids"])
        ours = {r["row_id"] for r in val}
        if ours != ref:
            raise SystemExit(f"Validation split differs from {a.ref_split}: {len(ours ^ ref)} rows differ. Stop.")
        print(f"Validation split is identical to {a.ref_split} ({len(ref)} rows).")

    chosen = val if not a.n or a.n >= len(val) else random.Random(a.sample_seed).sample(val, a.n)
    chosen.sort(key=lambda r: r["row_id"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps({"id": r["row_id"], "split": "eval", "category": r["category"],
                                "instruction": r["instruction"], "context": r["context"],
                                "human_response": r["response"]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} prompts to {out}")
    print("per category:", dict(Counter(r["category"] for r in chosen).most_common()))


if __name__ == "__main__":
    main()
