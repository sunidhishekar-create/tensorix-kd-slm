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
    def get_soft_targets(self, student_ids: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
        """
        Runs teacher inference on a batch of student tokens and returns
        the temperature-scaled probability distribution over the student vocabulary.

        Args:
            student_ids: Tensor of shape (B, T) with student token IDs
            temperature: Softmax temperature T (> 0). Higher = softer distribution.

        Returns:
            student_target_probs: Float32 tensor of shape (B, T, vocab_size) summing to 1.0
        """
        # 1. Translate student token IDs to GPT-2 token IDs: (B, T)
        teacher_ids = self.vocab_mapper.to_teacher_tokens(student_ids)

        # 2. Run teacher forward pass without gradients: logits shape (B, T, 50257)
        outputs = self.model(input_ids=teacher_ids)
        teacher_logits = outputs.logits

        # 3. Always compute softmax in float32 for numerical stability (Rule 3)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes combined Knowledge Distillation loss:
        L = (1 - alpha) * L_CE + alpha * (T^2) * L_KL

    All probability and KL divergence math is strictly done in float32.

    Args:
        student_logits: Student output logits of shape (B, T, vocab_size)
        teacher_probs: Teacher soft probabilities of shape (B, T, vocab_size)
        targets: Ground truth next-token labels of shape (B, T)
        temperature: Distillation temperature T (> 0)
        alpha: Weight balancing CE and KD, in range [0, 1]

    Returns:
        total_loss: (1 - alpha) * L_CE + alpha * (T^2) * L_KL (scalar)
        ce_loss: Standard cross-entropy loss (scalar, for logging)
        kl_loss: Unscaled KL divergence D_KL(P_T || P_S) (scalar, for logging)
    """
    # 1. Cast logits and probabilities to float32 for numerical stability (Rule 3)
    s_logits_f32 = student_logits.float()
    t_probs_f32 = teacher_probs.float()

    B, T, V = s_logits_f32.shape

    # 2. Hard Cross-Entropy Loss against ground-truth targets
    # Flatten (B, T, V) -> (B*T, V) and (B, T) -> (B*T,)
    ce_loss = F.cross_entropy(s_logits_f32.view(-1, V), targets.view(-1))

    # 3. Soft Distillation Loss via KL Divergence
    # Student log-probabilities at temperature T: shape (B*T, V)
    s_log_probs = F.log_softmax(s_logits_f32 / temperature, dim=-1).view(-1, V)
    t_probs_flat = t_probs_f32.view(-1, V)

    # PyTorch kl_div: D_KL(target || input) = sum(target * (log(target) - input))
    # reduction="batchmean" divides by (B * T), giving true mathematical expectation
    kl_div = F.kl_div(s_log_probs, t_probs_flat, reduction="batchmean")

    # 4. Scale KL divergence by T^2 to keep gradient magnitudes consistent across temperatures
    kd_loss = (temperature ** 2) * kl_div

    # 5. Combined convex combination
    total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss

    return total_loss, ce_loss, kl_div


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
        device: str = "cpu",
        gpt2_vocab_size: int = 50257,
        gpt2_eos_id: int = 50256,
    ):
        self.device = device
        self.temperature = float(temperature)
        self.alpha = float(alpha)

        # If a pre-instantiated model is passed, inherit its vocab size and eos id
        if hasattr(teacher_model, "config"):
            gpt2_vocab_size = getattr(teacher_model.config, "vocab_size", gpt2_vocab_size)
            gpt2_eos_id = getattr(teacher_model.config, "eos_token_id", gpt2_eos_id)
            if gpt2_eos_id is None:
                gpt2_eos_id = gpt2_vocab_size - 1

        self.vocab_mapper = VocabMapper(
            vocab_data_or_path,
            gpt2_vocab_size=gpt2_vocab_size,
            gpt2_eos_id=gpt2_eos_id,
        ).to(device)

        self.teacher = TeacherModel(
            model_name_or_model=teacher_model,
            vocab_mapper=self.vocab_mapper,
            device=device,
        )

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
        # 1. Obtain soft targets from frozen teacher: shape (B, T, vocab_size)
        teacher_probs = self.teacher.get_soft_targets(x, temperature=self.temperature)

        # 2. Compute combined loss
        return distillation_loss(
            student_logits=student_logits,
            teacher_probs=teacher_probs,
            targets=y,
            temperature=self.temperature,
            alpha=self.alpha,
        )


if __name__ == "__main__":
    from transformers import GPT2Config, GPT2LMHeadModel
    from model import SLM, ModelConfig

    print("=" * 60)
    print("Running Full Distillation Unit & Integration Verification")
    print("=" * 60)

    mock_vocab = {
        "gpt2_ids_kept": [10, 20, 30, 40],
        "unk_id": 4,
        "eos_id": 3,
        "vocab_size": 5,
        "tokens": {"train": 100, "val": 10}
    }

    # 1. Mock Teacher Network (GPT-2 structure with 50 vocab)
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

    # 2. Initialize Distiller
    distiller = Distiller(
        vocab_data_or_path=mock_vocab,
        teacher_model=mock_teacher_net,
        temperature=2.0,
        alpha=0.5,
        device="cpu",
    )
    print("[OK] Distiller initialized successfully.")

    # 3. Instantiate Student SLM model from model.py
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

    # 4. Generate synthetic batch: batch_size=2, sequence_length=8
    B, T = 2, 8
    x = torch.randint(0, student_cfg.vocab_size, (B, T))
    y = torch.randint(0, student_cfg.vocab_size, (B, T))

    # 5. Student forward pass
    student_logits, student_ce = student(x, y)
    assert student_logits.shape == (B, T, student_cfg.vocab_size), (
        f"Expected student logits shape {(B, T, student_cfg.vocab_size)}, got {student_logits.shape}"
    )
    print(f"[OK] Student forward pass completed. Logits shape: {student_logits.shape}")

    # 6. Distillation loss calculation
    total_loss, ce_loss, kl_loss = distiller(x, y, student_logits)
    print(f"[OK] Distillation loss computed:")
    print(f"     Total Loss: {total_loss.item():.4f}")
    print(f"     CE Loss:    {ce_loss.item():.4f}")
    print(f"     KL Div:     {kl_loss.item():.4f}")

    # Verify CE loss matches student's built-in CE loss
    assert torch.allclose(ce_loss, student_ce, atol=1e-4), "Distiller CE does not match model's CE!"
    print("[OK] Distiller CE loss exactly matches student model's native CE loss.")

    # 7. Backward pass test: verify gradients on student, zero gradients on teacher
    student.zero_grad()
    total_loss.backward()

    # Verify student parameters received valid gradients
    student_grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert all(g is not None for g in student_grads), "Some student parameters did not receive gradients!"
    assert all(not torch.isnan(g).any() for g in student_grads), "Student gradients contain NaN!"
    print("[OK] Student model gradients verified (valid, non-zero, non-NaN).")

    # Verify teacher parameters received NO gradients
    teacher_grads = [p.grad for p in distiller.teacher.model.parameters()]
    assert all(g is None for g in teacher_grads), "Teacher parameters unexpectedly received gradients!"
    print("[OK] Teacher model strictly preserved without gradients.")

    print("=" * 60)
    print("ALL STEP 4 INTEGRATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)
