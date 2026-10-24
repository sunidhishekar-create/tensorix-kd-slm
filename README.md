# TENSORIX: Distilling Knowledge into a Small Language Model

AFML course project (default track). We train a small decoder-only Transformer (about 19.4M parameters) from scratch on TinyStories. Then we test whether knowledge distillation from a larger frozen teacher (GPT-2-small, 124M) gives a better model than training the same small model without a teacher, under identical compute.

## Team and roles

| Member | GitHub | Role |
|---|---|---|
| Sunidhi Shekar | sunidhishekar-create | Foundation: tokenizer, data pipeline, model, baseline training, README |
| Bhuvi Sudheendra Katti | Bhuvi Sudheendra Katti | Distillation: teacher loading, KD loss, temperature sweep |
| Poojitha Lakshmi Bhaskara Raghavendra |Poojithaa| Instruction fine-tuning (Dolly 15k), optional feature-based KD |
| Diran Somanna | diran07 | Evaluation, results plots, then leads the PPT and demo video |

(Roles are suggestions. Swap them if needed, and update this table.)

## Project idea

1. **Baseline:** train the small model (the "student") on TinyStories with the normal next-token loss.
2. **Distilled:** train the same model with the same budget, but also match the teacher's output probabilities (soft targets at temperature T).
3. **Compare:**
   - validation perplexity
   - blind pairwise preference on held-out prompts
   - how much of the teacher's quality is kept versus size, memory and latency
   - the effect of different temperatures

## How the project works (read this first)

| Place | Job |
|---|---|
| **VS Code (laptop)** | Write and edit code |
| **GitHub (this repo)** | Shared source of truth. Code only, never data or checkpoints |
| **Colab (T4 GPU)** | Run training and evaluation |
| **Shared Drive folder `tensorix`** | Big files: data and checkpoints |

The loop: edit in VS Code, then `git add`, `git commit`, `git push` on your own branch, then open a Pull Request and merge it. In Colab, pull the latest code and run it. Colab's disk is wiped when a session ends, so nothing important is ever saved only there.

## Repo navigation

```
src/
  data_prep.py   builds the pruned tokenizer and train.bin / val.bin
  model.py       decoder-only Transformer (RMSNorm, RoPE, SwiGLU, weight tying)
  train.py       resumable training loop (baseline now; KD plugs into the loss)
notebooks/
  baseline_train.ipynb   Colab notebook used for the baseline run (reuse it for your runs)
configs/         settings files (empty so far)
results/         plots and tables
slides/          blank course PPT template
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
- **Why pruned GPT-2:** the student keeps GPT-2's token identities, so teacher outputs can be lined up with the student's. `vocab.json` stores:
  - `gpt2_ids_kept`: the GPT-2 id of each student id (student id `i` is GPT-2 id `gpt2_ids_kept[i]`)
  - `unk_id`: the student's catch-all id (equal to `len(gpt2_ids_kept)`)
  - `eos_id`, `vocab_size` and the token counts
- **Model:** 6 layers, 6 heads, width 384, SwiGLU hidden size 1024, context 256, about 19.4M parameters. `forward(..., return_hidden=True)` also returns every block's output, for feature-based distillation.
- **Data:** about 474M training tokens and 4.77M validation tokens, stored as uint16 arrays. 2,119,719 training stories and 21,990 validation stories.

## Fixed training budget (do not change)

Every run, baseline or distilled, must use these so the comparison is fair:

`--max_steps 20000 --batch_size 64 --block_size 256`

The learning-rate schedule depends on `--max_steps`, so changing it changes the result. Only `--out_dir` (and your own new flags) should differ between runs.

## What has been done (Sunidhi)

- [x] Repo, branches, shared Drive folder, Colab workflow
- [x] `data_prep.py` run on the full TinyStories dataset (vocab 22,875)
- [x] `model.py` tested: 19.4M parameters, untrained loss about ln(vocab)
- [x] `train.py` with checkpointing and resume, tested on CPU and on a Colab T4 (resume confirmed after a real disconnect)
- [x] Progress slides drafted (kept outside the public repo)
- [ ] Baseline run, 20,000 steps. Final validation perplexity: **`<fill in from log.csv>`**. Checkpoint: `tensorix/checkpoints/baseline/best.pt`

## Before you start (one-time, everyone)

1. Accept the GitHub collaborator invite and the Google Drive invite.
2. Drive, then **Shared with me**, right-click `tensorix`, **Organize**, **Add shortcut**, **My Drive**. Do not rename the folder, because the path `/content/drive/MyDrive/tensorix` must be identical for everyone.
3. Install Git and VS Code on your laptop, then clone: `git clone https://github.com/sunidhishekar-create/tensorix-kd-slm.git`.

## Using Colab (every new session)

Do you need to run the notebook before starting? **Not the training.** Do not rerun the baseline. What you do need, in every new Colab session, is the setup cells, because each session is a fresh empty machine. Set the runtime to **T4 GPU** first (Runtime, Change runtime type).

Open `notebooks/baseline_train.ipynb` in Colab (File, Open notebook, GitHub tab, paste the repo URL), or paste these cells into a new notebook.

```python
# Cell 1: connect Drive (run once per session; running it twice gives a "Mountpoint" error)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: latest code, and data on Colab's local disk (reading from Drive during training is slow)
import os
if not os.path.exists('/content/tensorix-kd-slm'):
    !git clone https://github.com/sunidhishekar-create/tensorix-kd-slm.git
%cd /content/tensorix-kd-slm
!git pull
!pip install -q -r requirements.txt
!mkdir -p /content/data
!cp -n /content/drive/MyDrive/tensorix/data/* /content/data/
!nvidia-smi | head -n 12
```

`nvidia-smi` should show a Tesla T4. If it says "command not found", the GPU is not set.

```python
# Cell 3: train. Safe to rerun after a disconnect: it resumes automatically.
!python src/train.py --data_dir /content/data --out_dir /content/drive/MyDrive/tensorix/checkpoints/<run_name> --max_steps 20000 --eval_interval 1000 --save_interval 1000
```

Use a new `<run_name>` for each experiment, for example `kd_T2`. Reusing a name resumes that run instead of starting a new one. If Colab disconnects, rerun Cells 1 to 3 unchanged.

Each run saves `ckpt.pt` (latest, for resuming), `best.pt` (lowest validation loss, use this for evaluation and as a distillation starting point) and `log.csv` (step, train loss, validation loss, validation perplexity, learning rate).

## Working on the code (VS Code)

```powershell
git checkout main
git pull
git checkout -b feature/<your-topic>
# edit, then:
git add <your files>
git commit -m "Describe what you changed"
git push -u origin feature/<your-topic>
```

Then open GitHub, click **Compare & pull request**, **Create pull request**, **Merge pull request**, **Confirm merge**.

Rules:

- Never push straight to `main`. Always use a branch and a Pull Request.
- Run `git pull` at the start of every work session.
- Mostly edit your own files to avoid merge conflicts. If you need to change `model.py` or `train.py`, tell Sunidhi first, because everyone's runs depend on them.
- Never commit data, checkpoints, tokens, API keys or SRNs. The repo is public.
- Test small first (a few steps, small batch) before spending Colab time.

## To do

**Distillation (`src/distill.py`, plus a small hook in `train.py`)**
- Load the frozen teacher (GPT-2-small) and run it on the same text as the student.
- Convert student ids to GPT-2 ids with `gpt2_ids_kept`. Rare tokens that became `<unk>` cannot be converted back, so replace them with a fallback id, or re-tokenize the raw text for the teacher.
- Slice the teacher's logits to the kept tokens. Put the logsumexp of all dropped tokens' logits into the `<unk>` slot, so probabilities stay consistent.
- Implement the KD loss: `alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * cross-entropy`. Forward KL first, reverse KL if time allows.
- Train with at least two temperatures (for example T = 1, 2, 4), same budget as the baseline.

**Instruction fine-tuning and feature KD**
- Format Dolly 15k as prompt and response pairs, and write a fine-tuning script (test on GPT-2 first).
- Optional: feature-based distillation using `return_hidden=True`, with a projection from the student width (384) to the teacher width (768).
- Fine-tune the baseline and the best distilled model.

**Evaluation and deliverables**
- `src/evaluate.py`: validation perplexity, parameter count, memory, latency (test on GPT-2 until real checkpoints exist).
- Blind pairwise comparison of baseline vs distilled on held-out prompts, with a results table.
- Plots from each run's `log.csv` into `results/`.
- Lead the final PPT and the demo video.

## Final deliverables

**PPT:** start from `slides/AFML-ppt-template.pptx`. A progress version covering the foundation work is kept on the shared Drive, so add the distillation and results slides to it. Suggested owners: architecture and tokenizer by Sunidhi, distillation by the distillation owner, fine-tuning and results by the fine-tuning owner, structure and polish by the evaluation owner.

**5-minute demo video:** one screen recording with narration, about 30 seconds each for the problem, the model, baseline versus distilled results, a live generation from both models side by side, and the conclusions. Record last, once results exist.

## References

1. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network (2015). https://arxiv.org/abs/1503.02531
2. Sanh, Debut, Chaumond, Wolf. DistilBERT, a distilled version of BERT (2019). https://arxiv.org/abs/1910.01108
3. Gu, Dong, Wei, Huang. MiniLLM: Knowledge Distillation of Large Language Models (2023, ICLR 2024). https://arxiv.org/abs/2306.08543
