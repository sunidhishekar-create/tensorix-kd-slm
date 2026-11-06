"""Evaluate a student checkpoint or a Hugging Face model on the fixed validation windows.

Reports perplexity on C4 and TinyStories, parameter count, fp16 weight size, peak GPU
memory and latency, and writes everything to <out_dir>/<name>.json.

  student: python src/evaluate.py --data_dir /content/data --ckpt <run>/best.pt --name baseline_short
  teacher: python src/evaluate.py --data_dir /content/data --hf_model HuggingFaceTB/SmolLM2-360M --name teacher_base

The validation windows are built exactly as in train.py (seed 0, 1024 windows of 256 tokens per
domain), so numbers from here match train.py's log.csv and are comparable across every model
that shares the SmolLM2 tokenizer. Do not change --eval_windows or --block_size between runs.
"""
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def fixed_windows(data_dir, block_size, n_windows):
    """Identical to the validation-window code in train.py."""
    out = {}
    for k in ("val_c4", "val_ts"):
        arr = np.memmap(Path(data_dir) / f"{k}.bin", dtype=np.uint16, mode="r")
        starts = np.random.default_rng(0).integers(0, len(arr) - block_size - 1, n_windows)
        x = np.stack([arr[i:i + block_size] for i in starts]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + block_size] for i in starts]).astype(np.int64)
        out[k] = (torch.from_numpy(x), torch.from_numpy(y))
    return out


@torch.no_grad()
def mean_loss(logits_fn, X, Y, device, bs):
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            lg = logits_fn(x)
        loss = F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1))
        tot += loss.item() * len(x)
        n += len(x)
    return tot / n


@torch.no_grad()
def benchmark(logits_fn, device, block_size, reps=20):
    """Median latency of one forward pass on 1 x block_size tokens, and peak GPU memory."""
    x = torch.randint(0, 49152, (1, block_size), device=device)
    for _ in range(5):                                   # warm-up
        logits_fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        logits_fn(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    return statistics.median(times) * 1000, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--name", required=True, help="label for the output file, e.g. baseline_short")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="student checkpoint written by train.py (best.pt or ckpt.pt)")
    g.add_argument("--hf_model", help="Hugging Face model id, e.g. HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--eval_windows", type=int, default=1024)
    ap.add_argument("--eval_bs", type=int, default=16)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    windows = fixed_windows(a.data_dir, a.block_size, a.eval_windows)

    info = {"name": a.name, "device": device, "block_size": a.block_size, "eval_windows": a.eval_windows}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)

    if a.ckpt:
        from model import load_checkpoint
        model, ck = load_checkpoint(a.ckpt, device)
        model.eval()
        logits_fn = lambda x: model(x)[0]
        info.update(kind="student", source=a.ckpt, step=ck["step"])
    else:
        from transformers import AutoModelForCausalLM
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=dtype).to(device).eval()
        logits_fn = lambda x: model(input_ids=x).logits
        info.update(kind="hf_model", source=a.hf_model)

    n_params = sum(p.numel() for p in model.parameters())      # tied weights are counted once
    info["params_m"] = round(n_params / 1e6, 2)
    info["weights_fp16_mb"] = round(n_params * 2 / 1e6, 1)

    for k, (X, Y) in windows.items():
        loss = mean_loss(logits_fn, X, Y, device, a.eval_bs)
        info[f"{k}_loss"] = round(loss, 4)
        info[f"{k}_ppl"] = round(float(np.exp(loss)), 3)
    info["val_avg_loss"] = round((info["val_c4_loss"] + info["val_ts_loss"]) / 2, 4)

    if a.ckpt and device == "cuda":
        model.half()            # benchmark the student in fp16, like the teacher
    ms, peak = benchmark(logits_fn, device, a.block_size)
    info["latency_ms_1x%d" % a.block_size] = round(ms, 2)
    info["peak_mem_mb"] = round(peak, 1) if peak is not None else None

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / f"{a.name}.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()