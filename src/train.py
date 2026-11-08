"""Resumable training loop for the SLM. Baseline now; KD plugs into the loss later."""
import argparse, csv, json, math, os, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import SLM, ModelConfig


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with train.bin, val_c4.bin, val_ts.bin, meta.json")
    ap.add_argument("--out_dir", required=True, help="where checkpoints and log.csv go")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64, help="effective batch (sequences per step)")
    ap.add_argument("--grad_accum", type=int, default=2, help="micro-batches per step; raise if out of memory")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_windows", type=int, default=1024, help="fixed validation windows per domain")
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    a = get_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    assert a.batch_size % a.grad_accum == 0, "batch_size must be divisible by grad_accum"
    micro, T = a.batch_size // a.grad_accum, a.block_size
    torch.manual_seed(a.seed)

    # ---- data: memory-mapped, never fully loaded into RAM
    dd = Path(a.data_dir)
    meta = json.load(open(dd / "meta.json"))
    d = {k: np.memmap(dd / f"{k}.bin", dtype=np.uint16, mode="r") for k in ("train", "val_c4", "val_ts")}

    def make_batch(arr, starts):
        x = np.stack([arr[i:i + T] for i in starts]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + T] for i in starts]).astype(np.int64)   # shifted by 1
        return torch.from_numpy(x), torch.from_numpy(y)

    # fixed validation windows: identical text in every run, so runs are directly comparable
    evalset = {}
    for k in ("val_c4", "val_ts"):
        starts = np.random.default_rng(0).integers(0, len(d[k]) - T - 1, a.eval_windows)
        evalset[k] = make_batch(d[k], starts)

    rng = np.random.default_rng(a.seed)

    def train_batch():
        starts = rng.integers(0, len(d["train"]) - T - 1, micro)
        x, y = make_batch(d["train"], starts)
        return x.to(device), y.to(device)

    # ---- model and optimizer
    cfg = ModelConfig(vocab_size=meta["vocab_size"], block_size=T)
    model = SLM(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume if a checkpoint exists
    step, best_val = 0, float("inf")
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["config"] != asdict(cfg):
            raise SystemExit(f"{ckpt_path} was made with a different model config. Use a new --out_dir.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); step, best_val = ck["step"], ck["best_val"]
        rng = np.random.default_rng(a.seed + step)    # don't replay the same batches after resuming
        print(f"resumed from step {step}")

    def save(path, full=True):
        tmp = path.with_suffix(".tmp")
        state = {"model": model.state_dict(), "step": step, "best_val": best_val, "config": asdict(cfg)}
        if full:
            state.update({"opt": opt.state_dict(), "scaler": scaler.state_dict()})
        torch.save(state, tmp)
        os.replace(tmp, path)            # atomic: a half-written file never replaces a good one

    def get_lr(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = (s - a.warmup) / max(1, a.max_steps - a.warmup)
        return a.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (a.lr - a.min_lr)

    @torch.no_grad()
    def evaluate():
        model.eval(); res = {}
        for k, (X, Y) in evalset.items():
            tot, n = 0.0, 0
            for i in range(0, len(X), micro):
                x, y = X[i:i + micro].to(device), Y[i:i + micro].to(device)
                with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                    _, loss = model(x, y)
                tot += loss.item() * len(x); n += len(x)
            res[k] = tot / n
        model.train(); return res

    log_path = out / "log.csv"
    if not log_path.exists():
        open(log_path, "w", newline="").write("step,train_loss,val_c4,val_ts,val_avg,ppl_c4,ppl_ts,lr\n")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.1f}M vocab={cfg.vocab_size} "
          f"tokens/step={a.batch_size * T} micro_batch={micro} x {a.grad_accum}")
    model.train(); t0 = time.time(); run_sum, run_n = 0.0, 0
    while step < a.max_steps:
        lr = get_lr(step)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(a.grad_accum):
            x, y = train_batch()
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss / a.grad_accum).backward()
            step_loss += loss.item() / a.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1; run_sum += step_loss; run_n += 1

        if step % 50 == 0:
            dt = time.time() - t0; t0 = time.time()
            ms = dt / 50 * 1000
            print(f"step {step} | loss {step_loss:.3f} | lr {lr:.2e} | {ms:.0f} ms/step "
                  f"| projected full run {ms * a.max_steps / 3.6e6:.1f} h")
        if step % a.eval_interval == 0 or step == a.max_steps:
            r = evaluate(); avg = (r["val_c4"] + r["val_ts"]) / 2
            tr = run_sum / run_n; run_sum, run_n = 0.0, 0
            p_c4, p_ts = math.exp(r["val_c4"]), math.exp(r["val_ts"])
            print(f">>> step {step} | train {tr:.3f} | val C4 {r['val_c4']:.3f} (ppl {p_c4:.1f}) "
                  f"| val TS {r['val_ts']:.3f} (ppl {p_ts:.2f}) | avg {avg:.3f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, tr, r["val_c4"], r["val_ts"], avg, p_c4, p_ts, lr])
            if avg < best_val:
                best_val = avg; save(out / "best.pt", full=False)
        if step % a.save_interval == 0 or step == a.max_steps:
            save(ckpt_path)
    print("done")


if __name__ == "__main__":
    main()