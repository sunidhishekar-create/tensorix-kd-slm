"""Fine-tune the pretrained student on ONE instruction dataset (human answers or Gemma answers).

Run it twice with identical settings, once per dataset. Only the answer source differs:

  python experiments/gemma_sft/05_finetune.py --train <dir>/human_train.npz --eval <dir>/human_eval.npz <dir>/gemma_eval.npz \
      --init <ckpt>/baseline_full/best.pt --out_dir <ckpt>/sft_human
  python experiments/gemma_sft/05_finetune.py --train <dir>/gemma_train.npz --eval <dir>/human_eval.npz <dir>/gemma_eval.npz \
      --init <ckpt>/baseline_full/best.pt --out_dir <ckpt>/sft_gemma

The loss is taken on answer tokens only (prompt positions are -100 in the arrays from 04_tokenize_sft.py).
Every eval set is scored before training (epoch 0) and after each epoch, as answer-token loss and perplexity.
--init none starts from random weights and is only for a quick smoke test on a laptop.
"""
import argparse, csv, json, math, sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model import SLM, ModelConfig, load_checkpoint          # noqa: E402
from common import EOS_ID, TOKENIZER, build_prompt, read_jsonl  # noqa: E402


def load_npz(path):
    d = np.load(path)
    return torch.from_numpy(d["x"].astype(np.int64)), torch.from_numpy(d["y"].astype(np.int64))


@torch.no_grad()
def answer_loss(model, X, Y, device, bs):
    """Mean cross-entropy per answer token (prompt and padding are ignored)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits, _ = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-100, reduction="sum").item()
        n += int((y != -100).sum())
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="npz from 04_tokenize_sft.py")
    ap.add_argument("--eval", nargs="+", required=True, help="one or more eval npz files")
    ap.add_argument("--init", required=True, help="pretrained checkpoint (best.pt) or 'none' for a smoke test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_train", type=int, default=0, help="use only the first N training examples (smoke tests)")
    ap.add_argument("--max_eval", type=int, default=0, help="use only the first N examples of each eval set (smoke tests)")
    ap.add_argument("--prompts", default="data/sft/prompts.jsonl", help="for the sample generations")
    ap.add_argument("--n_samples", type=int, default=2, help="sample answers to print at the end (0 = none)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)           # same seed and same prompt order for both datasets
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, Y = load_npz(a.train)
    evals = {Path(p).stem: load_npz(p) for p in a.eval}
    if a.max_train:
        X, Y = X[:a.max_train], Y[:a.max_train]
    if a.max_eval:
        evals = {n: (ex[:a.max_eval], ey[:a.max_eval]) for n, (ex, ey) in evals.items()}
    print(f"device={device} | train examples {len(X)} from {a.train} | eval sets {list(evals)}")

    if a.init == "none":
        model = SLM(ModelConfig()).to(device)
        print("WARNING: random initialisation (smoke test only)")
    else:
        model, ck = load_checkpoint(a.init, device)
        print(f"loaded {a.init} (pretraining step {ck['step']})")
        del ck
    model.train()
    cfg = model.cfg

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = math.ceil(len(X) / a.batch_size)
    total = a.epochs * steps_per_epoch
    warm = min(a.warmup, max(1, total // 10))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        t = (s - warm) / max(1, total - warm)
        return a.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (a.lr - a.min_lr)

    log_path = out / "log.csv"
    cols = ["epoch", "train_loss"] + [f"{n}_loss" for n in evals] + [f"{n}_ppl" for n in evals]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    def evaluate(epoch, train_loss):
        res = {n: answer_loss(model, ex, ey, device, a.batch_size) for n, (ex, ey) in evals.items()}
        line = " | ".join(f"{n} {v:.3f} (ppl {math.exp(v):.1f})" for n, v in res.items())
        print(f">>> epoch {epoch} | train {train_loss:.3f} | answer-token loss: {line}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss] + [res[n] for n in evals]
                                   + [math.exp(res[n]) for n in evals])
        return res

    evaluate(0, float("nan"))                     # the pretrained model before any fine-tuning
    step = 0
    final = {}
    for epoch in range(1, a.epochs + 1):
        perm = rng.permutation(len(X))
        run, nb = 0.0, 0
        for i in range(0, len(X), a.batch_size):
            idx = torch.from_numpy(perm[i:i + a.batch_size])
            x, y = X[idx].to(device), Y[idx].to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            run += loss.item()
            nb += 1
        final = evaluate(epoch, run / nb)

    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step,
                "source_data": a.train, "init": a.init}, out / "sft.pt")
    json.dump({"train": a.train, "init": a.init, "epochs": a.epochs, "batch_size": a.batch_size,
               "lr": a.lr, "seed": a.seed, "steps": step, "final_eval_loss": final},
              open(out / "sft_result.json", "w"), indent=2)
    print(f"saved {out / 'sft.pt'} after {step} steps")

    if a.n_samples > 0 and Path(a.prompts).exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        rows = [r for r in read_jsonl(a.prompts) if r["split"] == "eval"][:a.n_samples]
        for r in rows:
            ids = tok(build_prompt(r["instruction"], r["context"]), add_special_tokens=False)["input_ids"]
            gen = model.generate(torch.tensor([ids], device=device), max_new_tokens=100)[0].tolist()[len(ids):]
            if EOS_ID in gen:
                gen = gen[:gen.index(EOS_ID)]
            print(f"\nINSTRUCTION: {r['instruction']}\nMODEL ANSWER: {tok.decode(gen)}")


if __name__ == "__main__":
    main()