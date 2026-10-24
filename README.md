# TENSORIX: Distilling Knowledge into a Small Language Model

AFML course project (default track). We train a small decoder-only Transformer (about 19.4M parameters) from scratch on TinyStories, then test whether knowledge distillation from a larger frozen teacher (GPT-2-small, 124M) gives a better model than training the same small model without a teacher, under identical compute.

## Team and roles

| Member | GitHub | Role |
|---|---|---|
| <Name 1> | <username> | Foundation: tokenizer, data pipeline, model, baseline training, README |
| <Name 2> | <username> | Distillation: teacher loading, KD loss, temperature sweep |
| <Name 3> | <username> | Instruction fine-tuning (Dolly 15k), optional feature-based KD |
| <Name 4> | <username> | Evaluation, results plots, then leads the PPT and demo video |

## Project idea

1. **Baseline:** train the small model on TinyStories with normal next-token loss.
2. **Distilled:** train the same model with the same budget, but also match the teacher's output probabilities (soft targets, temperature T).
3. **Compare:** validation perplexity, blind pairwise preference on held-out prompts, how much of the teacher's quality is kept vs. size/memory/latency, and the effect of different T.

## How the project works (read this first)

| Place | Job |
|---|---|
| **VS Code (laptop)** | Write and edit code |
| **GitHub (this repo)** | Shared source of truth. Code only, never data or checkpoints |
| **Colab (T4 GPU)** | Run training and evaluation |
| **Shared Drive folder `tensorix`** | Big files: data and checkpoints |

Loop: edit in VS Code, then `git add`, `git commit`, `git push` on your own branch, open a Pull Request and merge it, then in Colab `git clone` or `git pull` and run.

## Repo navigation

```
src/
  data_prep.py   builds the pruned tokenizer + train.bin / val.bin
  model.py       decoder-only Transformer (RMSNorm, RoPE, SwiGLU, weight tying)
  train.py       resumable training loop (baseline now, KD plugs into the loss)
notebooks/       Colab notebooks
configs/         settings files (empty so far)
results/         plots and tables
slides/          PPT template and final deck
```

Shared Drive folder `tensorix` (not in git):

```
tensorix/data/                    train.bin, val.bin, vocab.json
tensorix/checkpoints/baseline/    ckpt.pt (latest), best.pt (lowest val loss), log.csv
tensorix/checkpoints/<run>/       one folder per experiment, e.g. kd_T2, kd_T4
tensorix/results/                 evaluation outputs
```

## Key design decisions

- **Tokenizer:** GPT-2's BPE vocabulary pruned to tokens that occur at least 5 times in TinyStories. 50,257 tokens became **22,875**. Every dropped token maps to one `<unk>` id.
- **Why pruned GPT-2:** the student keeps GPT-2's token identities, so teacher logits can be aligned to the student. `data/vocab.json` stores `gpt2_ids_kept` (the GPT-2 id of each student id) and `unk_id`.
- **Model:** 6 layers, 6 heads, width 384, SwiGLU hidden size 1024, context 256, about 19.4M parameters. `forward(..., return_hidden=True)` also returns every block's output for feature-based distillation.
- **Data:** about 474M training tokens and 4.77M validation tokens, stored as uint16 arrays.

## Fixed training budget (do not change)

Every run, baseline or distilled, must use these so the comparison is fair:

`--max_steps 20000 --batch_size 64 --block_size 256`

The learning-rate schedule depends on `--max_steps`, so changing it changes the result.

## What has been done (<Name 1>)

- [x] Repo, branches, shared Drive folder
- [x] `data_prep.py` run on the full TinyStories dataset
- [x] `model.py` tested (19.4M params, initial loss about ln(vocab))
- [x] `train.py` with checkpointing and resume, tested on CPU and Colab T4
- [ ] Baseline run: 20,000 steps. Final val perplexity: **<fill in>**. Checkpoint in `tensorix/checkpoints/baseline/best.pt`

## How to run (Colab)

Set the runtime to **T4 GPU** first. Run each cell once, in order.

```python
# Cell 1: connect Drive (run once per session)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: code and data (data is copied to local disk because Drive is slow to read)
!git clone https://github.com/<owner>/tensorix-kd-slm.git
%cd tensorix-kd-slm
!mkdir -p /content/data
!cp /content/drive/MyDrive/tensorix/data/* /content/data/
```

```python
# Cell 3: train (safe to rerun after a disconnect; it resumes automatically)
!python src/train.py --data_dir /content/data --out_dir /content/drive/MyDrive/tensorix/checkpoints/<run_name> --max_steps 20000 --eval_interval 1000 --save_interval 1000
```

Before this, each teammate must add the shared folder to their Drive: Drive, Shared with me, right-click `tensorix`, Organize, Add shortcut, My Drive.

## Rules for working together

- Never push directly to `main`. Make a branch like `feature/distillation`, push it, open a Pull Request.
- Run `git pull` before every work session.
- Mostly edit your own files to avoid merge conflicts. If you need to change `model.py` or `train.py`, tell the owner first.
- Never commit data, checkpoints, tokens or API keys. The repo is public.

## To do

**Member 2: Distillation** (`src/distill.py`, plus a small hook in `train.py`)
- Load the frozen teacher (GPT-2-small) and run it on the same text as the student.
- Convert student ids to GPT-2 ids with `gpt2_ids_kept`. Slice teacher logits to the kept tokens and put the logsumexp of the dropped ones into the `<unk>` slot.
- Implement the KD loss: `alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * cross-entropy`. Try forward KL first, reverse KL if time allows.
- Train with T = 1, 2, 4 (at least two), same budget as the baseline.

**Member 3: Fine-tuning and feature KD**
- Format Dolly 15k as prompt/response pairs and write a fine-tuning script (test on GPT-2 first).
- Optional: feature-based distillation using `return_hidden=True`, with a projection from the student width (384) to the teacher width (768).
- Produce fine-tuned models for the baseline and the best distilled model.

**Member 4: Evaluation and deliverables**
- `src/evaluate.py`: validation perplexity, params, memory, latency (test on GPT-2 until real checkpoints exist).
- Blind pairwise comparison on held-out prompts, with a results table.
- Plots from each run's `log.csv` into `results/`.
- Lead the PPT and demo video (below).

## Final deliverables

**PPT** (template in `slides/AFML-ppt-template.pptx`): problem and idea, tokenizer and architecture, baseline, distillation method, results (perplexity curves, pairwise comparison, size/latency table), conclusion. Owners: architecture slides by Member 1, distillation by Member 2, fine-tuning and results by Member 3, structure and polish by Member 4.

**5-minute demo video:** one screen recording with narration, about 30 s each for the problem, the model, the baseline vs. distilled comparison, a live generation from both models side by side, and the conclusions. Record last, after the results exist.

## References

1. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network (2015). arxiv.org/abs/1503.02531
2. Sanh et al. DistilBERT (2019). arxiv.org/abs/1910.01108
3. Gu et al. MiniLLM (2023). arxiv.org/abs/2306.08543