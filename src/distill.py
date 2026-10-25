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
