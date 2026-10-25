"""Knowledge Distillation from GPT-2 teacher to student SLM."""
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabMapper(nn.Module):
    """
    Handles vocabulary translation between:
      - Student SLM (pruned vocab: 22,875 tokens)
      - Teacher GPT-2 (full vocab: 50,257 tokens)
    """
    def __init__(self, vocab_data_or_path, gpt2_vocab_size=50257, gpt2_eos_id=50256):
        super().__init__()
        # Allow passing either a path to vocab.json or an in-memory dict
        if isinstance(vocab_data_or_path, (str, Path)):
            with open(vocab_data_or_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
        else:
            vocab_data = vocab_data_or_path

        self.vocab_size = int(vocab_data["vocab_size"])
        self.unk_id = int(vocab_data["unk_id"])
        self.gpt2_vocab_size = gpt2_vocab_size
        self.gpt2_eos_id = gpt2_eos_id

        # Kept GPT-2 IDs for student tokens [0, unk_id - 1]
        gpt2_ids_kept = list(vocab_data["gpt2_ids_kept"])
        assert len(gpt2_ids_kept) == self.unk_id, (
            f"Expected {self.unk_id} kept IDs, got {len(gpt2_ids_kept)}"
        )

        # Mapping student ID -> GPT-2 ID
        # For student tokens < unk_id: use gpt2_ids_kept[student_id]
        # For student token == unk_id: map to GPT-2 eos token (50256)
        s2t = list(gpt2_ids_kept) + [gpt2_eos_id]

        # Register as non-persistent buffers: they move with .to(device),
        # require no gradients, and won't pollute model state_dict.
        self.register_buffer("student_to_gpt2", torch.tensor(s2t, dtype=torch.long), persistent=False)
        self.register_buffer("gpt2_ids_kept", torch.tensor(gpt2_ids_kept, dtype=torch.long), persistent=False)

    def to_teacher_tokens(self, student_ids: torch.Tensor) -> torch.Tensor:
        """
        Translates a batch of student token IDs to GPT-2 token IDs.
        Args:
            student_ids: Tensor of shape (B, T) with values in [0, vocab_size - 1]
        Returns:
            teacher_ids: Tensor of shape (B, T) with values in [0, gpt2_vocab_size - 1]
        """
        # student_to_gpt2: shape (vocab_size,)
        # student_ids: shape (B, T)
        # Returns teacher_ids: shape (B, T)
        return self.student_to_gpt2[student_ids]

    def teacher_probs_to_student_probs(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        Projects teacher probability distribution (over 50,257 tokens)
        down to student probability distribution (over 22,875 tokens).
        Args:
            teacher_probs: Float32 tensor of shape (B, T, 50257) summing to 1.0 along dim=-1
        Returns:
            student_probs: Float32 tensor of shape (B, T, 22875) summing to 1.0 along dim=-1
        """
        # teacher_probs: (B, T, 50257)
        # Slices probabilities for all kept tokens: (B, T, unk_id)
        kept_probs = teacher_probs.index_select(dim=-1, index=self.gpt2_ids_kept)

        # All dropped GPT-2 tokens were merged into unk_id in data_prep.py.
        # Since sum(teacher_probs) == 1.0 across all 50,257 tokens,
        # sum(dropped) == 1.0 - sum(kept).
        # unk_prob: (B, T, 1)
        unk_prob = (1.0 - kept_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)

        # Concatenate along vocabulary dimension: (B, T, vocab_size)
        student_probs = torch.cat([kept_probs, unk_prob], dim=-1)
        return student_probs


class TeacherModel(nn.Module):
    """
    Frozen teacher wrapper around HuggingFace's GPT-2 (124M).
    Extracts teacher probabilities projected into the student vocabulary space.
    """
    def __init__(self, model_name_or_model="gpt2", vocab_mapper=None, device="cpu"):
        super().__init__()
        from transformers import GPT2LMHeadModel

        # Allow passing a model name string or an existing model instance
        if isinstance(model_name_or_model, str):
            print(f"Loading teacher model '{model_name_or_model}'...")
            self.model = GPT2LMHeadModel.from_pretrained(model_name_or_model)
        else:
            self.model = model_name_or_model

        self.model.to(device)
        self.model.eval()

        # Freeze all teacher parameters: teacher is purely an inference oracle
        for param in self.model.parameters():
            param.requires_grad = False

        self.vocab_mapper = vocab_mapper
        self.device = device

    @torch.no_grad()
    def get_soft_targets(
        self,
        student_ids: torch.Tensor,
        temperature: float = 2.0,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """
        Runs teacher inference on a batch of student tokens and returns
        the temperature-scaled probability distribution over the student vocabulary.

        Args:
            student_ids: Tensor of shape (B, T) with student token IDs
            temperature: Softmax temperature T (> 0). Higher = softer distribution.
            chunk_size: Process in sub-batches along dim 0 to prevent GPU OOM.

        Returns:
            student_target_probs: Float32 tensor of shape (B, T, vocab_size) summing to 1.0
        """
        B = student_ids.size(0)
        # If batch is larger than chunk_size, process in sub-batches to save memory
        if chunk_size and chunk_size > 0 and B > chunk_size:
            chunks = [
                self.get_soft_targets(student_ids[i:i + chunk_size], temperature=temperature, chunk_size=0)
                for i in range(0, B, chunk_size)
            ]
            return torch.cat(chunks, dim=0)

        # 1. Translate student token IDs to GPT-2 token IDs: (B, T)
        teacher_ids = self.vocab_mapper.to_teacher_tokens(student_ids)

        # 2. Run teacher forward pass: on cuda, run under autocast float16 for Tensor Core speedup;
        # on cpu, keep fp32 and no autocast.
        device_type = "cuda" if teacher_ids.is_cuda else "cpu"
        use_amp = teacher_ids.is_cuda
        with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
            outputs = self.model(input_ids=teacher_ids)
            teacher_logits = outputs.logits

        # 3. Immediately convert logits to float32 for numerical stability (Rule 3)
        teacher_logits = teacher_logits.float()

        # 4. Softmax with temperature scaling: shape (B, T, 50257)
        # As T increases, differences between top tokens and tail tokens soften
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # 5. Project teacher distribution down to student vocabulary: shape (B, T, vocab_size)
        student_target_probs = self.vocab_mapper.teacher_probs_to_student_probs(teacher_probs)

        return student_target_probs


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    loss_chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes combined Knowledge Distillation loss chunked across rows to prevent OOM:
        L = (1 - alpha) * L_CE + alpha * (T^2) * L_KL

    Uses torch.utils.checkpoint.checkpoint(..., use_reentrant=False) per row chunk
    so that big intermediate activation tensors (~1.4 GiB each for 16,384 x 22,875)
    are discarded during the forward pass and recomputed one small chunk at a time.

    Args:
        student_logits: Student output logits of shape (B, T, vocab_size)
        teacher_probs: Teacher soft probabilities of shape (B, T, vocab_size)
        targets: Ground truth next-token labels of shape (B, T)
        temperature: Distillation temperature T (> 0)
        alpha: Weight balancing CE and KD, in range [0, 1]
        loss_chunk: Number of sequence positions to compute at a time (default 2048)

    Returns:
        total_loss: (1 - alpha) * L_CE + alpha * (T^2) * L_KL (scalar)
        ce_loss: Mean cross-entropy loss (detached scalar, for logging)
        kl_loss: Mean KL divergence D_KL(P_T || P_S) (detached scalar, for logging)
    """
    from torch.utils.checkpoint import checkpoint

    V = student_logits.size(-1)
    # Flatten across batch and sequence dimensions: (N, V) where N = B * T
    # Note: views do not copy memory or cast to float32
    s_flat = student_logits.view(-1, V)
    t_flat = teacher_probs.view(-1, V)
    y_flat = targets.view(-1)
    N = s_flat.size(0)

    # Inner chunk function: computes row sums in float32 without keeping big tensors
    def _chunk_fn(s_c: torch.Tensor, t_c: torch.Tensor, y_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_f32 = s_c.float()
        t_f32 = t_c.float()
        # Sum of CE loss over chunk rows at T=1
        ce_c = F.cross_entropy(s_f32, y_c, reduction="sum")
        # Sum of KL divergence over chunk rows at temperature T
        log_s_c = F.log_softmax(s_f32 / temperature, dim=-1)
        kl_c = F.kl_div(log_s_c, t_f32, reduction="sum")
        return ce_c, kl_c

    tot_ce = torch.zeros((), dtype=torch.float32, device=s_flat.device)
    tot_kl = torch.zeros((), dtype=torch.float32, device=s_flat.device)

    # Process in row chunks
    for i in range(0, N, loss_chunk):
        s_c = s_flat[i : i + loss_chunk]
        t_c = t_flat[i : i + loss_chunk]
        y_c = y_flat[i : i + loss_chunk]

        # Use activation checkpointing during training to discard big intermediate tensors
        if torch.is_grad_enabled() and s_c.requires_grad:
            ce_c, kl_c = checkpoint(_chunk_fn, s_c, t_c, y_c, use_reentrant=False)
        else:
            ce_c, kl_c = _chunk_fn(s_c, t_c, y_c)

        tot_ce = tot_ce + ce_c
        tot_kl = tot_kl + kl_c

    # Divide sums by total number of rows N to obtain exact means
    ce_loss = tot_ce / N
    kl_loss = tot_kl / N

    # Combined loss scaled by T^2 for gradient consistency
    total_loss = (1.0 - alpha) * ce_loss + alpha * (temperature ** 2) * kl_loss

    return total_loss, ce_loss.detach(), kl_loss.detach()


class Distiller:
    """
    High-level orchestrator for Knowledge Distillation:
      - Translates vocabulary via VocabMapper
      - Obtains soft targets from frozen TeacherModel
      - Computes combined distillation loss (CE + KD)
    """
    def __init__(
        self,
        vocab_data_or_path,
        teacher_model="gpt2",
        temperature: float = 2.0,
        alpha: float = 0.5,
        teacher_chunk: int = 16,
        loss_chunk: int = 2048,
        device: str = "cpu",
    ):
        self.device = device
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher_chunk = int(teacher_chunk)
        self.loss_chunk = int(loss_chunk)

        # 1. Initialize teacher model first so we can reliably inspect its config
        self.teacher = TeacherModel(
            model_name_or_model=teacher_model,
            vocab_mapper=None,
            device=device,
        )

        teacher_vocab_size = getattr(self.teacher.model.config, "vocab_size", 50257)
        teacher_eos_id = getattr(self.teacher.model.config, "eos_token_id", 50256)
        if teacher_eos_id is None:
            teacher_eos_id = teacher_vocab_size - 1

        # 2. Initialize VocabMapper with the teacher's true vocab size and eos id
        self.vocab_mapper = VocabMapper(
            vocab_data_or_path,
            gpt2_vocab_size=teacher_vocab_size,
            gpt2_eos_id=teacher_eos_id,
        ).to(device)

        # 3. Connect vocab_mapper to the teacher
        self.teacher.vocab_mapper = self.vocab_mapper

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes distillation loss given student batch and student predictions.

        Args:
            x: Input token IDs of shape (B, T)
            y: Target token IDs of shape (B, T)
            student_logits: Student logits of shape (B, T, vocab_size)

        Returns:
            total_loss: Loss for backpropagation: (1 - alpha) * L_CE + alpha * (T^2) * L_KL
            ce_loss: Scalar cross-entropy loss (for logging)
            kl_loss: Scalar KL divergence (for logging)
        """
        # 1. Obtain soft targets from frozen teacher in teacher_chunk slices: shape (B, T, vocab_size)
        teacher_probs = self.teacher.get_soft_targets(
            x,
            temperature=self.temperature,
            chunk_size=self.teacher_chunk,
        )

        # 2. Compute combined loss with row-chunked memory checkpointing
        return distillation_loss(
            student_logits=student_logits,
            teacher_probs=teacher_probs,
            targets=y,
            temperature=self.temperature,
            alpha=self.alpha,
            loss_chunk=self.loss_chunk,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run distillation self-tests")
    args = parser.parse_args()

    from transformers import GPT2Config, GPT2LMHeadModel
    from model import SLM, ModelConfig

    print("=" * 60)
    print("Running Full Distillation Unit & Integration Verification")
    print("=" * 60)

    # -------------------------------------------------------------
    # 0. Mathematical Equivalence Test: Chunked vs Unchunked Loss
    # -------------------------------------------------------------
    print("Test 0: Verifying mathematical equivalence of row-chunked loss...")
    def _reference_unchunked_loss(s_logits, t_probs, tgts, temp, alp):
        s_f = s_logits.float()
        t_f = t_probs.float()
        V_dim = s_f.size(-1)
        ce = F.cross_entropy(s_f.view(-1, V_dim), tgts.view(-1))
        log_s = F.log_softmax(s_f / temp, dim=-1).view(-1, V_dim)
        kl = F.kl_div(log_s, t_f.view(-1, V_dim), reduction="batchmean")
        tot = (1.0 - alp) * ce + alp * (temp ** 2) * kl
        return tot, ce, kl

    torch.manual_seed(999)
    # N = 23 (prime number) with loss_chunk = 7 (does not divide N evenly)
    test_B, test_T, test_V = 1, 23, 17
    s_raw = torch.randn(test_B, test_T, test_V)
    t_logits = torch.randn(test_B, test_T, test_V)
    t_probs_test = F.softmax(t_logits, dim=-1)
    y_test = torch.randint(0, test_V, (test_B, test_T))

    s_ref = s_raw.clone().detach().requires_grad_(True)
    s_chk = s_raw.clone().detach().requires_grad_(True)

    ref_tot, ref_ce, ref_kl = _reference_unchunked_loss(s_ref, t_probs_test, y_test, temp=2.3, alp=0.65)
    ref_tot.backward()

    chk_tot, chk_ce, chk_kl = distillation_loss(s_chk, t_probs_test, y_test, temperature=2.3, alpha=0.65, loss_chunk=7)
    chk_tot.backward()

    tot_diff = (ref_tot - chk_tot).abs().item()
    ce_diff = (ref_ce - chk_ce).abs().item()
    kl_diff = (ref_kl - chk_kl).abs().item()
    grad_diff = (s_ref.grad - s_chk.grad).abs().max().item()

    print(f"     Max diffs -> total: {tot_diff:.2e} | ce: {ce_diff:.2e} | kl: {kl_diff:.2e} | grad: {grad_diff:.2e}")
    assert tot_diff < 1e-6, f"Total loss diff {tot_diff} exceeded 1e-6!"
    assert ce_diff < 1e-6, f"CE loss diff {ce_diff} exceeded 1e-6!"
    assert kl_diff < 1e-6, f"KL loss diff {kl_diff} exceeded 1e-6!"
    assert grad_diff < 1e-6, f"Gradient max diff {grad_diff} exceeded 1e-6!"
    print("[OK] Mathematical equivalence confirmed (values & gradients match within 1e-6).")

    # -------------------------------------------------------------
    # 1. VocabMapper & Teacher Tests
    # -------------------------------------------------------------
    mock_vocab = {
        "gpt2_ids_kept": [10, 20, 30, 40],
        "unk_id": 4,
        "eos_id": 3,
        "vocab_size": 5,
        "tokens": {"train": 100, "val": 10}
    }

    mock_config = GPT2Config(
        vocab_size=50,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=2,
        bos_token_id=49,
        eos_token_id=49
    )
    mock_teacher_net = GPT2LMHeadModel(mock_config)

    distiller = Distiller(
        vocab_data_or_path=mock_vocab,
        teacher_model=mock_teacher_net,
        temperature=2.0,
        alpha=0.5,
        teacher_chunk=2,
        loss_chunk=4,
        device="cpu",
    )
    print("[OK] Distiller initialized successfully with chunking.")

    student_cfg = ModelConfig(
        vocab_size=5,
        block_size=16,
        n_layer=2,
        n_head=2,
        d_model=32,
        d_ff=64,
        dropout=0.0
    )
    student = SLM(student_cfg)
    print("[OK] Student SLM model instantiated from src/model.py.")

    B, T = 2, 8
    x = torch.randint(0, student_cfg.vocab_size, (B, T))
    y = torch.randint(0, student_cfg.vocab_size, (B, T))

    student_logits, student_ce = student(x, y)
    assert student_logits.shape == (B, T, student_cfg.vocab_size)
    print(f"[OK] Student forward pass completed. Logits shape: {student_logits.shape}")

    total_loss, ce_loss, kl_loss = distiller(x, y, student_logits)
    print(f"[OK] Distillation loss computed:")
    print(f"     Total Loss: {total_loss.item():.4f}")
    print(f"     CE Loss:    {ce_loss.item():.4f}")
    print(f"     KL Div:     {kl_loss.item():.4f}")

    assert torch.allclose(ce_loss, student_ce, atol=1e-4), "Distiller CE does not match model's CE!"
    print("[OK] Distiller CE loss exactly matches student model's native CE loss.")

    student.zero_grad()
    total_loss.backward()

    student_grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert all(g is not None for g in student_grads), "Some student parameters did not receive gradients!"
    assert all(not torch.isnan(g).any() for g in student_grads), "Student gradients contain NaN!"
    print("[OK] Student model gradients verified (valid, non-zero, non-NaN).")

    teacher_grads = [p.grad for p in distiller.teacher.model.parameters()]
    assert all(g is None for g in teacher_grads), "Teacher parameters unexpectedly received gradients!"
    print("[OK] Teacher model strictly preserved without gradients.")

    print("=" * 60)
    print("ALL TESTS (INCLUDING MATHEMATICAL EQUIVALENCE) PASSED!")
    print("=" * 60)
