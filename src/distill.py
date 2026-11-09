"""Knowledge Distillation (KD) module for training SLM with a frozen teacher.

Provides:
- load_teacher(): safely loads SmolLM2-360M frozen in fp16 (cuda) or fp32 (cpu).
- teacher_logits(): runs forward pass under torch.no_grad() and asserts vocab size.
- CLI diagnostics:
  --check_teacher: validates teacher cross-entropy against README benchmarks (3.047 C4, 1.907 TinyStories).
  --check_fp16: verifies numerical stability (NaN/inf check), top-1 agreement, and KL divergence between fp16 and fp32.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Allow imports from src/ whether invoked from repo root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import fixed_windows

STUDENT_VOCAB_SIZE = 49152


def load_teacher(name: str = "HuggingFaceTB/SmolLM2-360M", device: str = "cuda"):
    """
    Loads the teacher model in eval mode with all parameters frozen.

    Args:
        name: HuggingFace model identifier.
        device: Target device ("cuda" or "cpu").

    Returns:
        teacher model in eval mode with requires_grad=False.
        Never added to optimizer or student checkpoints.
    """
    dev_str = str(device)
    target_dtype = torch.float16 if "cuda" in dev_str else torch.float32

    # Compatibility: newer transformers accept dtype=, older accept torch_dtype=
    try:
        teacher = AutoModelForCausalLM.from_pretrained(name, dtype=target_dtype)
    except TypeError:
        teacher = AutoModelForCausalLM.from_pretrained(name, torch_dtype=target_dtype)

    teacher = teacher.to(device).eval()

    # Freeze all parameters so teacher never accumulates gradients
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


@torch.no_grad()
def teacher_logits(teacher, x: torch.Tensor) -> torch.Tensor:
    """
    Computes teacher forward pass under torch.no_grad().

    Args:
        teacher: frozen teacher model.
        x: input token tensor of shape (B, T) with dtype int64.

    Returns:
        logits: tensor of shape (B, T, 49152) in the teacher's native dtype.
    """
    # Teacher forward pass: input_ids shape (B, T)
    out = teacher(input_ids=x)
    logits = out.logits  # shape: (B, T, V)

    # Sanity check: student and teacher must share the identical 49,152 vocabulary
    assert logits.size(-1) == STUDENT_VOCAB_SIZE, (
        f"Teacher vocabulary size mismatch: expected {STUDENT_VOCAB_SIZE}, got {logits.size(-1)}"
    )
    return logits


def check_teacher(data_dir: str, teacher_name: str, device: str, block_size: int = 256, n_windows: int = 1024, bs: int = 16):
    """
    Evaluates the teacher model on fixed validation windows (1024 windows of 256 tokens)
    and compares mean cross-entropy loss against the README ground truth.
    README values: C4 = 3.047, TinyStories = 1.907.
    """
    print(f"\n[check_teacher] Loading teacher '{teacher_name}' on {device}...")
    teacher = load_teacher(teacher_name, device)

    print(f"[check_teacher] Loading {n_windows} fixed validation windows from '{data_dir}'...")
    windows = fixed_windows(data_dir, block_size, n_windows)

    readme_targets = {
        "val_c4": 3.047,
        "val_ts": 1.907,
    }

    print("\n" + "=" * 75)
    print(f"{'Domain':<15} | {'Computed Loss':<15} | {'README Target':<15} | {'Diff':<10} | {'Status'}")
    print("-" * 75)

    all_passed = True
    for domain, (X, Y) in windows.items():
        tot_loss, n_samples = 0.0, 0
        for i in range(0, len(X), bs):
            x_batch = X[i:i + bs].to(device)  # shape: (B, T)
            y_batch = Y[i:i + bs].to(device)  # shape: (B, T)

            # logits shape: (B, T, 49152)
            logits = teacher_logits(teacher, x_batch)

            # Compute cross-entropy in float32
            loss = F.cross_entropy(logits.float().view(-1, STUDENT_VOCAB_SIZE), y_batch.view(-1))
            tot_loss += loss.item() * len(x_batch)
            n_samples += len(x_batch)

        mean_loss = tot_loss / n_samples
        target = readme_targets.get(domain, None)
        if target is not None:
            diff = mean_loss - target
            passed = abs(diff) <= 0.01
            all_passed = all_passed and passed
            status_str = "PASS (<= 0.01)" if passed else f"WARN (diff={diff:+.4f})"
            print(f"{domain:<15} | {mean_loss:<15.4f} | {target:<15.4f} | {diff:<+10.4f} | {status_str}")
        else:
            print(f"{domain:<15} | {mean_loss:<15.4f} | {'N/A':<15} | {'N/A':<10} | OK")

    print("=" * 75)
    if all_passed:
        print("[check_teacher] SUCCESS: All teacher validation losses match README targets within 0.01 tolerance.\n")
    else:
        print("[check_teacher] WARNING: Validation losses differ slightly (> 0.01). Check data prep or environment.\n")


def check_fp16(data_dir: str, teacher_name: str, device: str, block_size: int = 256, bs: int = 4):
    """
    Compares fp16 teacher predictions with an fp32 reference:
    - NaN / Inf checks
    - Top-1 argmax agreement percentage
    - Forward KL divergence KL(fp32 || fp16) across temperatures T in [1, 2, 4]
    """
    n_windows = 16 if device == "cuda" else 4
    print(f"\n[check_fp16] Running precision sanity check on {n_windows} windows per domain ({device})...")

    print(f"[check_fp16] Loading primary teacher ({'fp16' if device == 'cuda' else 'fp32'})...")
    teacher_primary = load_teacher(teacher_name, device)

    if device == "cuda":
        print(f"[check_fp16] Loading reference fp32 teacher on cuda...")
        try:
            teacher_ref = AutoModelForCausalLM.from_pretrained(teacher_name, dtype=torch.float32)
        except TypeError:
            teacher_ref = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.float32)
        teacher_ref = teacher_ref.to("cuda").eval()
        for p in teacher_ref.parameters():
            p.requires_grad_(False)
    else:
        print(f"[check_fp16] CPU mode detected: comparing fp32 against fp32 reference...")
        teacher_ref = teacher_primary

    windows = fixed_windows(data_dir, block_size, n_windows)
    temps = [1.0, 2.0, 4.0]

    for domain, (X, Y) in windows.items():
        print(f"\n--- Checking Domain: {domain} ({len(X)} windows) ---")
        n_tokens_total = 0
        top1_matches = 0
        kl_sums = {T: 0.0 for T in temps}
        has_nan_or_inf = False

        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device)  # shape: (B, T)
            B, T_seq = x.shape
            n_tokens = B * T_seq

            logits_pri = teacher_logits(teacher_primary, x)  # (B, T, 49152)
            logits_ref = teacher_logits(teacher_ref, x)      # (B, T, 49152)

            # Check NaNs and Infs
            if torch.isnan(logits_pri).any() or torch.isinf(logits_pri).any():
                has_nan_or_inf = True
            if torch.isnan(logits_ref).any() or torch.isinf(logits_ref).any():
                has_nan_or_inf = True

            # Top-1 agreement: compare predicted token IDs
            pred_pri = logits_pri.argmax(dim=-1)  # (B, T)
            pred_ref = logits_ref.argmax(dim=-1)  # (B, T)
            top1_matches += (pred_pri == pred_ref).sum().item()
            n_tokens_total += n_tokens

            # Compute KL divergence KL(ref || primary) in float32 at each temperature
            l_pri_32 = logits_pri.float()  # (B, T, 49152)
            l_ref_32 = logits_ref.float()  # (B, T, 49152)

            for T_val in temps:
                # Scaled logits: (B, T, V)
                s_ref = l_ref_32 / T_val
                s_pri = l_pri_32 / T_val

                # Probabilities & log-probabilities in float32
                p_ref = F.softmax(s_ref, dim=-1)         # target distribution P: (B, T, V)
                log_p_ref = F.log_softmax(s_ref, dim=-1)  # log P: (B, T, V)
                log_q_pri = F.log_softmax(s_pri, dim=-1)  # log Q: (B, T, V)

                # KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
                # Sum over vocabulary (dim=-1), sum over tokens in batch
                kl_batch = (p_ref * (log_p_ref - log_q_pri)).sum(dim=-1).sum().item()
                kl_sums[T_val] += kl_batch

        top1_acc = 100.0 * top1_matches / n_tokens_total
        print(f"NaN / Inf detected: {has_nan_or_inf}")
        print(f"Top-1 Argmax Agreement: {top1_acc:.4f}% ({top1_matches}/{n_tokens_total} tokens)")
        for T_val in temps:
            mean_kl = kl_sums[T_val] / n_tokens_total
            print(f"Mean KL(ref || primary) at T={T_val:.1f}: {mean_kl:.6e}")

    # Clean up reference teacher from GPU VRAM if loaded separately
    if device == "cuda" and teacher_ref is not teacher_primary:
        del teacher_ref
        gc.collect()
        torch.cuda.empty_cache()
        print("\n[check_fp16] Freed reference fp32 teacher from GPU memory.")


def main():
    ap = argparse.ArgumentParser(description="Knowledge Distillation utilities & teacher verification.")
    ap.add_argument("--data_dir", required=True, help="Path containing val_c4.bin and val_ts.bin")
    ap.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-360M", help="Hugging Face teacher model id")
    ap.add_argument("--check_teacher", action="store_true", help="Score teacher cross-entropy on 1024 fixed windows")
    ap.add_argument("--check_fp16", action="store_true", help="Compare fp16 vs fp32 precision, NaN/inf, and KL")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--eval_windows", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.check_teacher:
        check_teacher(
            data_dir=args.data_dir,
            teacher_name=args.teacher,
            device=device,
            block_size=args.block_size,
            n_windows=args.eval_windows,
            bs=args.batch_size,
        )

    if args.check_fp16:
        check_fp16(
            data_dir=args.data_dir,
            teacher_name=args.teacher,
            device=device,
            block_size=args.block_size,
            bs=4,
        )

    if not args.check_teacher and not args.check_fp16:
        print("No diagnostic action specified. Use --check_teacher or --check_fp16.")


if __name__ == "__main__":
    main()

