"""Tokenize C4 (en) and TinyStories with the SmolLM2 tokenizer into uint16 .bin files.

Outputs in --out:
  train.bin   C4 tokens followed by TinyStories tokens (random windows give the mix)
  val_c4.bin, val_ts.bin   official validation splits, one per domain
  meta.json   token counts, mix, tokenizer info
Every document ends with the EOS id (0). Finished pieces are skipped on rerun.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOK_NAME = "HuggingFaceTB/SmolLM2-360M"
EOS = 0           # <|endoftext|>; the tokenizer adds no BOS/EOS by itself
BATCH = 1000      # documents per tokenizer call


def batches(stream, n):
    buf = []
    for ex in stream:
        buf.append(ex["text"])
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def encode_to_bin(stream, tok, n_tokens, path):
    """Tokenize documents, append EOS to each, write uint16 until n_tokens is reached."""
    if path.exists():
        print(f"skip {path.name} (already built)")
        return
    tmp = path.with_suffix(".tmp")
    total = docs = 0
    bar = tqdm(total=n_tokens, unit="tok", desc=path.name)
    with open(tmp, "wb") as f:
        for texts in batches(stream, BATCH):
            before = total
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids + [EOS], dtype=np.uint16)
                f.write(arr.tobytes())
                total += len(arr)
                docs += 1
                if total >= n_tokens:
                    break
            bar.update(total - before)
            if total >= n_tokens:
                break
    bar.close()
    tmp.replace(path)              # a half-written file never gets the final name
    print(f"{path.name}: {total:,} tokens, {docs:,} documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--c4_tokens", type=int, default=400_000_000)
    ap.add_argument("--ts_tokens", type=int, default=170_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000, help="per domain")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TOK_NAME)
    assert tok.eos_token_id == EOS and len(tok) < 65536

    def c4(split):
        return load_dataset("allenai/c4", "en", split=split, streaming=True)

    def ts(split):
        return load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    encode_to_bin(c4("train"), tok, args.c4_tokens, out / "c4_train.bin")
    encode_to_bin(ts("train"), tok, args.ts_tokens, out / "ts_train.bin")
    encode_to_bin(c4("validation"), tok, args.val_tokens, out / "val_c4.bin")
    encode_to_bin(ts("validation"), tok, args.val_tokens, out / "val_ts.bin")

    train = out / "train.bin"
    parts = [out / "c4_train.bin", out / "ts_train.bin"]
    want = sum(p.stat().st_size for p in parts)
    if not train.exists() or train.stat().st_size != want:   # stale or partial -> rebuild
        tmp = train.with_suffix(".tmp")
        with open(tmp, "wb") as dst:
            for p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, dst, 64 * 1024 * 1024)
        tmp.replace(train)

    names = ["c4_train", "ts_train", "train", "val_c4", "val_ts"]
    n = {k: (out / f"{k}.bin").stat().st_size // 2 for k in names}
    meta = {"tokenizer": TOK_NAME, "vocab_size": len(tok), "eos_id": EOS,
            "dtype": "uint16", "tokens": n,
            "mix": {"c4": n["c4_train"] / n["train"], "tinystories": n["ts_train"] / n["train"]}}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))

    for name in ("train", "val_c4", "val_ts"):     # eyeball check: decoded text should be sane
        d = np.memmap(out / f"{name}.bin", dtype=np.uint16, mode="r")
        print(f"\n--- {name}: max id {int(d.max())} | first 100 tokens decoded ---")
        print(tok.decode(d[:100].astype(np.int64)))


if __name__ == "__main__":
    main()