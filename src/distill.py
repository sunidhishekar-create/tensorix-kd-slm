"""Knowledge Distillation (KD) module for training SLM with a frozen teacher.

Provides:
- load_teacher(): safely loads SmolLM2-360M frozen in fp16 (cuda) or fp32 (cpu).
- teacher_logits(): runs forward pass under torch.no_grad() and asserts vocab size.
- kd_loss(): memory-efficient chunked autograd function computing forward KL + CE with analytic backward.
- CLI diagnostics:
  --check_teacher: validates teacher cross-entropy against README benchmarks (3.047 C4, 1.907 TinyStories).
  --check_fp16: verifies numerical stability (NaN/inf check), top-1 agreement, and KL divergence between fp16 and fp32.
  --selftest: runs comprehensive unit tests for kd_loss (float64 reference, alpha=0, s==t, teacher no-grad, fp16 CUDA).
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

# Allow imports from src/ whether invoked from repo root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from evaluate import fixed_windows
except ImportError:
    fixed_windows = None

STUDENT_VOCAB_SIZE = 49152


# ==============================================================================
# 1. Teacher Model Loading & Inference
# ==============================================================================

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
    if AutoModelForCausalLM is None:
        raise ImportError("transformers package is required to load the teacher model.")

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


# ==============================================================================
# 2. Memory-Efficient Chunked KD Loss (torch.autograd.Function)
# ==============================================================================

class KDLossFunction(torch.autograd.Function):
    """
    Custom autograd function for Knowledge Distillation:
      - Loops over chunks of rows (e.g. 2048) in float32 to prevent OOM.
      - Computes analytic gradient directly during forward pass into a preallocated buffer.
      - Backward pass is simply grad_output * buffer (cast to logits dtype).
      - ce and kl are marked non-differentiable (returned detached).
    """

    @staticmethod
    def forward(
        ctx,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        temperature: float,
        chunk: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = student_logits.shape
        V = student_logits.size(-1)
        N = student_logits.numel() // V

        z = student_logits.view(N, V)
        z_T = teacher_logits.view(N, V)
        y = targets.view(N)

        # Precision policy: float64 for high-precision tests, float32 for training/fp16
        calc_dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32

        # Preallocate one single gradient buffer of shape (N, V)
        grad_buffer = torch.empty((N, V), dtype=calc_dtype, device=student_logits.device)

        tot_kl = 0.0
        tot_ce = 0.0

        for i in range(0, N, chunk):
            c_end = min(i + chunk, N)
            C = c_end - i

            # Extract chunk and cast to calculation precision
            z_c = z[i:c_end].to(calc_dtype)     # student logits chunk: (C, V)
            z_T_c = z_T[i:c_end].to(calc_dtype) # teacher logits chunk: (C, V)
            y_c = y[i:c_end]                    # target token IDs: (C,)

            # Teacher distribution at temperature T: p = softmax(z_T / T)
            s_T = z_T_c / temperature
            p = F.softmax(s_T, dim=-1)          # (C, V)
            log_p = F.log_softmax(s_T, dim=-1)  # (C, V)

            # Student distribution at temperature T: q_T = softmax(z / T)
            s_S = z_c / temperature
            q_T = F.softmax(s_S, dim=-1)        # (C, V)
            log_q_T = F.log_softmax(s_S, dim=-1)# (C, V)

            # KL divergence for chunk: sum_k p * (log p - log q_T)
            kl_chunk = (p * (log_p - log_q_T)).sum()
            tot_kl += kl_chunk.item()

            # Student cross-entropy at T=1 against targets
            q_1 = F.softmax(z_c, dim=-1)        # (C, V)
            ce_chunk = F.cross_entropy(z_c, y_c, reduction="sum")
            tot_ce += ce_chunk.item()

            # Analytic gradient w.r.t student logits z:
            # [ alpha * T * (q_T - p) + (1 - alpha) * (softmax(z) - onehot(y)) ] / N
            grad_c = alpha * temperature * (q_T - p) + (1.0 - alpha) * q_1
            # Subtract (1 - alpha) at target token indices
            grad_c[torch.arange(C, device=grad_c.device), y_c] -= (1.0 - alpha)
            grad_c.div_(N)

            # Write chunk gradient directly into preallocated buffer
            grad_buffer[i:c_end].copy_(grad_c)

        mean_kl = tot_kl / N
        mean_ce = tot_ce / N
        total_val = alpha * (temperature ** 2) * mean_kl + (1.0 - alpha) * mean_ce

        # Save gradient buffer and metadata for backward
        ctx.save_for_backward(grad_buffer)
        ctx.student_shape = orig_shape
        ctx.student_dtype = student_logits.dtype

        # Kept on purpose: this exact code produced kd_short_T2 and kd_short_T4 (verified on a T4); change only together with a GPU re-run of --selftest.
        total_loss = torch.tensor(total_val, dtype=calc_dtype, device=student_logits.device, requires_grad=True)
        ce_tensor = torch.tensor(mean_ce, dtype=calc_dtype, device=student_logits.device)
        kl_tensor = torch.tensor(mean_kl, dtype=calc_dtype, device=student_logits.device)

        # Mark CE and KL as non-differentiable outputs
        ctx.mark_non_differentiable(ce_tensor, kl_tensor)

        return total_loss, ce_tensor, kl_tensor

    @staticmethod
    def backward(ctx, grad_total, grad_ce=None, grad_kl=None):
        grad_buffer, = ctx.saved_tensors
        if grad_total is None:
            grad_student = None
        else:
            # Multiply incoming gradient (e.g. from GradScaler) by analytic buffer
            # Cast back to student's native dtype (e.g. fp16) and restore original tensor shape (B, T, V)
            grad_student = (grad_total * grad_buffer).to(ctx.student_dtype).view(ctx.student_shape)

        # Gradients for: student_logits, teacher_logits, targets, alpha, temperature, chunk
        return grad_student, None, None, None, None, None


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    temperature: float,
    chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes combined Knowledge Distillation loss:
      total = alpha * T^2 * kl + (1 - alpha) * ce

    Args:
        student_logits: tensor of shape (B, T, V) or (N, V)
        teacher_logits: tensor of shape (B, T, V) or (N, V)
        targets: target token tensor of shape (B, T) or (N,)
        alpha: KD weighting factor in [0, 1]
        temperature: softening temperature T
        chunk: number of rows processed per chunk to bound VRAM usage

    Returns:
        total: differentiable scalar total loss
        ce: detached mean student cross-entropy
        kl: detached mean forward KL divergence
    """
    total, ce, kl = KDLossFunction.apply(
        student_logits, teacher_logits, targets, float(alpha), float(temperature), int(chunk)
    )
    return total, ce.detach(), kl.detach()


# ==============================================================================
# 3. CLI Diagnostics & Validation
# ==============================================================================

def check_teacher(data_dir: str, teacher_name: str, device: str, block_size: int = 256, n_windows: int = 1024, bs: int = 16):
    """
    Evaluates the teacher model on fixed validation windows (1024 windows of 256 tokens)
    and compares mean cross-entropy loss against the README ground truth.
    README values: C4 = 3.047, TinyStories = 1.907.
    """
    if fixed_windows is None:
        raise ImportError("evaluate.fixed_windows could not be imported.")

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
    if fixed_windows is None:
        raise ImportError("evaluate.fixed_windows could not be imported.")

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
                s_ref = l_ref_32 / T_val
                s_pri = l_pri_32 / T_val

                p_ref = F.softmax(s_ref, dim=-1)
                log_p_ref = F.log_softmax(s_ref, dim=-1)
                log_q_pri = F.log_softmax(s_pri, dim=-1)

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


def run_selftest():
    """
    Comprehensive self-tests for kd_loss:
      a) float64 on CPU, N not divisible by chunk: value and gradient equal unchunked autograd reference (atol < 1e-10).
      b) alpha=0 gives exactly the plain cross-entropy (value and gradient).
      c) student logits == teacher logits and alpha=1: kl about 0 and gradient about 0.
      d) kl >= 0, and teacher logits get no gradient.
      e) if cuda is available: fp16 student logits, N=8192, V=49152, loss scaled by 65536; compare unscaled grad with fp32 reference.
    """
    print("\n" + "=" * 75)
    print("RUNNING KD_LOSS SELF-TESTS")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Test a) float64 CPU unchunked reference agreement
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    N_a, V_a, chunk_a = 350, 100, 128  # 350 is intentionally not divisible by 128
    T_a, alpha_a = 2.5, 0.65

    s_logits_a = torch.randn(N_a, V_a, dtype=torch.float64, requires_grad=True)
    t_logits_a = torch.randn(N_a, V_a, dtype=torch.float64)
    y_a = torch.randint(0, V_a, (N_a,))

    total_a, ce_a, kl_a = kd_loss(s_logits_a, t_logits_a, y_a, alpha=alpha_a, temperature=T_a, chunk=chunk_a)
    total_a.backward()

    # Unchunked autograd reference
    s_ref_a = s_logits_a.clone().detach().requires_grad_(True)
    p_ref_a = F.softmax(t_logits_a / T_a, dim=-1)
    log_q_ref_a = F.log_softmax(s_ref_a / T_a, dim=-1)
    ref_kl_a = F.kl_div(log_q_ref_a, p_ref_a, reduction="sum") / N_a
    ref_ce_a = F.cross_entropy(s_ref_a, y_a)
    ref_tot_a = alpha_a * (T_a ** 2) * ref_kl_a + (1.0 - alpha_a) * ref_ce_a
    ref_tot_a.backward()

    val_diff_a = (total_a - ref_tot_a).abs().item()
    grad_diff_a = (s_logits_a.grad - s_ref_a.grad).abs().max().item()
    assert val_diff_a < 1e-10, f"Test a failed on loss value diff: {val_diff_a:.2e}"
    assert grad_diff_a < 1e-10, f"Test a failed on grad diff: {grad_diff_a:.2e}"
    print(f"OK [test a] float64 CPU unchunked reference agreement (val diff: {val_diff_a:.2e}, grad diff: {grad_diff_a:.2e})")

    # --------------------------------------------------------------------------
    # Test b) alpha=0 gives exactly plain cross-entropy
    # --------------------------------------------------------------------------
    torch.manual_seed(101)
    N_b, V_b = 200, 60
    s_logits_b = torch.randn(N_b, V_b, dtype=torch.float64, requires_grad=True)
    t_logits_b = torch.randn(N_b, V_b, dtype=torch.float64)
    y_b = torch.randint(0, V_b, (N_b,))

    total_b, ce_b, kl_b = kd_loss(s_logits_b, t_logits_b, y_b, alpha=0.0, temperature=2.0, chunk=64)
    total_b.backward()

    s_ref_b = s_logits_b.clone().detach().requires_grad_(True)
    ref_ce_b = F.cross_entropy(s_ref_b, y_b)
    ref_ce_b.backward()

    val_diff_b = (total_b - ref_ce_b).abs().item()
    grad_diff_b = (s_logits_b.grad - s_ref_b.grad).abs().max().item()
    assert val_diff_b < 1e-10, f"Test b failed on value diff: {val_diff_b:.2e}"
    assert grad_diff_b < 1e-10, f"Test b failed on grad diff: {grad_diff_b:.2e}"
    print(f"OK [test b] alpha=0 matches plain cross-entropy (val diff: {val_diff_b:.2e}, grad diff: {grad_diff_b:.2e})")

    # --------------------------------------------------------------------------
    # Test c) student logits == teacher logits and alpha=1: kl ~ 0, grad ~ 0
    # --------------------------------------------------------------------------
    torch.manual_seed(202)
    N_c, V_c = 150, 80
    t_logits_c = torch.randn(N_c, V_c, dtype=torch.float64)
    s_logits_c = t_logits_c.clone().detach().requires_grad_(True)
    y_c = torch.randint(0, V_c, (N_c,))

    total_c, ce_c, kl_c = kd_loss(s_logits_c, t_logits_c, y_c, alpha=1.0, temperature=2.0, chunk=64)
    total_c.backward()

    max_grad_c = s_logits_c.grad.abs().max().item()
    kl_val_c = kl_c.item()
    assert kl_val_c < 1e-12, f"Test c failed: KL is {kl_val_c:.2e}"
    assert max_grad_c < 1e-12, f"Test c failed: max grad is {max_grad_c:.2e}"
    print(f"OK [test c] student == teacher at alpha=1 (kl: {kl_val_c:.2e}, max grad: {max_grad_c:.2e})")

    # --------------------------------------------------------------------------
    # Test d) kl >= 0, and teacher logits receive no gradient
    # --------------------------------------------------------------------------
    torch.manual_seed(303)
    N_d, V_d = 100, 50
    t_logits_d = torch.randn(N_d, V_d, dtype=torch.float64, requires_grad=True)
    s_logits_d = torch.randn(N_d, V_d, dtype=torch.float64, requires_grad=True)
    y_d = torch.randint(0, V_d, (N_d,))

    total_d, ce_d, kl_d = kd_loss(s_logits_d, t_logits_d, y_d, alpha=0.5, temperature=2.0, chunk=32)
    total_d.backward()

    assert kl_d.item() >= 0.0, f"Test d failed: KL is negative ({kl_d.item()})"
    assert t_logits_d.grad is None, "Test d failed: teacher logits unexpectedly received gradients!"
    print(f"OK [test d] kl >= 0 ({kl_d.item():.4f}) and teacher logits gradient is None")

    # --------------------------------------------------------------------------
    # Test e) if CUDA is available: fp16 N=8192 V=49152 with loss scale=65536
    # --------------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.manual_seed(404)
        N_e, V_e = 8192, 49152
        alpha_e, T_e = 0.5, 2.0
        loss_scale = 65536.0

        s_logits_e = torch.randn(N_e, V_e, dtype=torch.float16, device="cuda", requires_grad=True)
        t_logits_e = torch.randn(N_e, V_e, dtype=torch.float16, device="cuda")
        y_e = torch.randint(0, V_e, (N_e,), device="cuda")

        total_e, ce_e, kl_e = kd_loss(s_logits_e, t_logits_e, y_e, alpha=alpha_e, temperature=T_e, chunk=2048)
        (total_e * loss_scale).backward()

        # Unscale gradient
        grad_unscaled_e = s_logits_e.grad.float() / loss_scale

        # Compute exact analytic gradient in fp32 on the first 16 rows
        sub_rows = 16
        z_sub = s_logits_e[:sub_rows].float()
        z_T_sub = t_logits_e[:sub_rows].float()
        y_sub = y_e[:sub_rows]

        p_sub = F.softmax(z_T_sub / T_e, dim=-1)
        q_T_sub = F.softmax(z_sub / T_e, dim=-1)
        q_1_sub = F.softmax(z_sub, dim=-1)

        ref_grad_sub = alpha_e * T_e * (q_T_sub - p_sub) + (1.0 - alpha_e) * q_1_sub
        ref_grad_sub[torch.arange(sub_rows, device="cuda"), y_sub] -= (1.0 - alpha_e)
        ref_grad_sub.div_(N_e)

        # Compare relative error on entries above 1e-6
        mask = ref_grad_sub.abs() > 1e-6
        if mask.any():
            rel_errors = (grad_unscaled_e[:sub_rows][mask] - ref_grad_sub[mask]).abs() / ref_grad_sub[mask].abs()
            max_rel_err = rel_errors.max().item()
        else:
            max_rel_err = 0.0

        # With fp16 quantization, relative error on non-zero entries is typically < 1e-2
        assert max_rel_err < 5e-2, f"Test e failed: max relative error too high ({max_rel_err:.2e})"
        print(f"OK [test e] fp16 CUDA N=8192 V=49152 scale=65536 (max relative error on |grad| > 1e-6: {max_rel_err:.2e})")
    else:
        print("OK [test e] (CUDA not available in local environment; execute in Colab T4 GPU to test fp16 8192x49152)")

    print("-" * 75)
    print("ALL KD_LOSS SELF-TESTS PASSED SUCCESSFULLY.")
    print("=" * 75 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Knowledge Distillation utilities & teacher verification.")
    ap.add_argument("--data_dir", default=None, help="Path containing val_c4.bin and val_ts.bin")
    ap.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-360M", help="Hugging Face teacher model id")
    ap.add_argument("--check_teacher", action="store_true", help="Score teacher cross-entropy on 1024 fixed windows")
    ap.add_argument("--check_fp16", action="store_true", help="Compare fp16 vs fp32 precision, NaN/inf, and KL")
    ap.add_argument("--selftest", action="store_true", help="Run comprehensive numerical unit tests for kd_loss")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--eval_windows", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.selftest:
        run_selftest()
        return

    if args.check_teacher:
        if not args.data_dir:
            ap.error("--data_dir is required when running --check_teacher")
        check_teacher(
            data_dir=args.data_dir,
            teacher_name=args.teacher,
            device=device,
            block_size=args.block_size,
            n_windows=args.eval_windows,
            bs=args.batch_size,
        )

    if args.check_fp16:
        if not args.data_dir:
            ap.error("--data_dir is required when running --check_fp16")
        check_fp16(
            data_dir=args.data_dir,
            teacher_name=args.teacher,
            device=device,
            block_size=args.block_size,
            bs=4,
        )

    if not args.check_teacher and not args.check_fp16 and not args.selftest:
        print("No diagnostic action specified. Use --selftest, --check_teacher, or --check_fp16.")


if __name__ == "__main__":
    main()
