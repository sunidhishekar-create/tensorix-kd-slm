"""Build a pruned-GPT2 tokenizer for TinyStories and encode the dataset to binary files."""
import argparse, itertools, json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def flat_ids(batch):
    return np.fromiter(itertools.chain.from_iterable(batch["ids"]), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--min_count", type=int, default=5, help="drop tokens rarer than this")
    ap.add_argument("--max_stories", type=int, default=0, help="0 = all; use e.g. 20000 to test")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    V, eos = tok.vocab_size, tok.eos_token_id
    ds = load_dataset("roneneldan/TinyStories")
    if args.max_stories:
        ds = {k: v.select(range(min(args.max_stories, len(v)))) for k, v in ds.items()}

    def enc(batch):
        return {"ids": [x + [eos] for x in tok(batch["text"])["input_ids"]]}
    ds = {k: v.map(enc, batched=True, remove_columns=v.column_names, num_proc=2)
          for k, v in ds.items()}

    counts = np.zeros(V, dtype=np.int64)
    for b in ds["train"].iter(batch_size=10000):
        counts += np.bincount(flat_ids(b), minlength=V)

    keep = np.where((counts >= args.min_count) | (np.arange(V) == eos))[0]
    unk_id = len(keep)
    lut = np.full(V, unk_id, dtype=np.int64)
    lut[keep] = np.arange(len(keep))
    vocab_size = len(keep) + 1

    stats = {}
    for split, name in [("train", "train"), ("validation", "val")]:
        n = 0
        with open(out / f"{name}.bin", "wb") as f:
            for b in ds[split].iter(batch_size=10000):
                ids = lut[flat_ids(b)].astype(np.uint16)
                f.write(ids.tobytes()); n += len(ids)
        stats[name] = n

    json.dump({"gpt2_ids_kept": keep.tolist(), "unk_id": unk_id, "eos_id": int(lut[eos]),
               "vocab_size": vocab_size, "tokens": stats}, open(out / "vocab.json", "w"))
    print(f"vocab size: {vocab_size} (from {V}) | tokens: {stats}")


if __name__ == "__main__":
    main()