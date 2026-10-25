"""Resumable training loop for the SLM. Works for the baseline now; KD plugs into the loss later."""
import argparse, csv, json, math, os, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import SLM, ModelConfig


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with train.bin, val.bin, vocab.json")
    ap.add_argument("--out_dir", required=True, help="where checkpoints and log.csv go")
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--teacher", default=None, help="teacher model, e.g. gpt2 (None = baseline training)")
    ap.add_argument("--kd_temp", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="distillation loss weight in [0, 1]")
    ap.add_argument("--teacher_chunk", type=int, default=16, help="sub-batch size for teacher forward pass to prevent OOM")
    return ap.parse_args()


def main():
    a = get_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_args.json", "w") as f:
        json.dump(vars(a), f, indent=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # ---- data: memory-mapped so the 900 MB file is never fully loaded into RAM
    vocab = json.load(open(Path(a.data_dir) / "vocab.json"))
    data = {s: np.memmap(Path(a.data_dir) / f"{s}.bin", dtype=np.uint16, mode="r")
            for s in ("train", "val")}

    def get_batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - a.block_size - 1, a.batch_size)
        x = np.stack([d[i:i + a.block_size] for i in ix]).astype(np.int64)
        y = np.stack([d[i + 1:i + 1 + a.block_size] for i in ix]).astype(np.int64)   # shifted by 1
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # ---- model and optimizer
    cfg = ModelConfig(vocab_size=vocab["vocab_size"], block_size=a.block_size)
    model = SLM(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]    # weight matrices get decay
    no_decay = [p for p in model.parameters() if p.dim() < 2]  # norm scales do not
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- optional knowledge distillation
    distiller = None
    if a.teacher:
        from distill import Distiller
        distiller = Distiller(
            vocab_data_or_path=Path(a.data_dir) / "vocab.json",
            teacher_model=a.teacher,
            temperature=a.kd_temp,
            alpha=a.kd_alpha,
            teacher_chunk=a.teacher_chunk,
            device=device,
        )

    # ---- resume if a checkpoint exists
    step, best_val = 0, float("inf")
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); step, best_val = ck["step"], ck["best_val"]
        np.random.seed(a.seed + step)    # don't replay the same batches after resuming
        print(f"resumed from step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_val": best_val,
                    "config": asdict(cfg)}, tmp)
        os.replace(tmp, path)            # atomic: a half-written file never replaces a good one

    def get_lr(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        t = (s - a.warmup) / max(1, a.max_steps - a.warmup)
        return a.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (a.lr - a.min_lr)

    @torch.no_grad()
    def evaluate():
        model.eval(); res = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(a.eval_iters):
                x, y = get_batch(split)
                with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                    _, loss = model(x, y)
                losses.append(loss.item())
            res[split] = float(np.mean(losses))
        model.train(); return res

    log_path = out / "log.csv"
    if not log_path.exists():
        open(log_path, "w", newline="").write("step,train_loss,val_loss,val_ppl,lr\n")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.1f}M vocab={cfg.vocab_size} "
          f"tokens/step={a.batch_size * a.block_size}"
          + (f" | teacher={a.teacher} T={a.kd_temp} alpha={a.kd_alpha}" if a.teacher else " | baseline"))
    model.train(); t0 = time.time()
    while step < a.max_steps:
        lr = get_lr(step)
        for g in opt.param_groups: g["lr"] = lr
        x, y = get_batch("train")
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            if distiller is not None:
                logits, _ = model(x, y)
                loss, last_ce, last_kl = distiller(x, y, logits)
            else:
                _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1

        if step % 50 == 0:
            dt = time.time() - t0; t0 = time.time()
            if distiller is not None:
                mem_str = f" | max_mem {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB" if device == "cuda" else ""
                print(f"step {step} | loss {loss.item():.3f} (ce {last_ce.item():.3f} kl {last_kl.item():.3f}) | lr {lr:.2e} | {dt/50*1000:.0f} ms/step{mem_str}")
            else:
                print(f"step {step} | loss {loss.item():.3f} | lr {lr:.2e} | {dt/50*1000:.0f} ms/step")
        if step % a.eval_interval == 0 or step == a.max_steps:
            r = evaluate(); ppl = math.exp(r["val"])
            print(f">>> step {step} | train {r['train']:.3f} | val {r['val']:.3f} | val ppl {ppl:.2f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, r["train"], r["val"], ppl, lr])
            if r["val"] < best_val:
                best_val = r["val"]; save(out / "best.pt")
        if step % a.save_interval == 0 or step == a.max_steps:
            save(ckpt_path)
    print("done")


if __name__ == "__main__":
    main()