# TENSORIX: Distilling Knowledge into a Small Language Model

AFML course project (default track). A small decoder-only Transformer (about 51M parameters) is trained from scratch on a mix of C4 and TinyStories. The project then tests whether knowledge distillation from a larger frozen teacher (SmolLM2-360M) gives a better small model than training the same model without a teacher, under the same training budget.

This is **version 2** of the project: more training data (C4 added to TinyStories), a different teacher (SmolLM2 instead of GPT-2), and a new shared tokenizer. Version 1 is preserved in git as the tag `v1-final`.

## Team and roles

| Member | GitHub | Role |
|---|---|---|
| Sunidhi Shekar | sunidhishekar-create | Foundation: tokenizer, data pipeline, model, baseline training, README |
| Bhuvi Sudheendra Katti | Bhuvi Sudheendra Katti | Distillation: teacher loading, KD loss, temperature sweep |
| Poojitha Lakshmi Bhaskara Raghavendra | Poojithaa | Instruction fine-tuning (Dolly 15k), optional feature-based KD |
| Diran Somanna | diran07 | Evaluation, results plots, then leads the PPT and demo video |


## Project idea

1. **Baseline:** train the small model (the "student") with the normal next-token loss.
2. **Distilled:** train the same model with the same budget, but also match the teacher's output probabilities (soft targets at temperature T).
3. **Compare:**
   - validation perplexity (C4 and TinyStories separately)
   - blind pairwise preference on held-out prompts
   - how much of the teacher's quality is kept, versus size, memory and latency
   - the effect of different temperatures

Success is measured against a matched baseline (same model size, same tokens seen), not by comparing the student directly with the teacher.

## Status

- [x] Repo reset for v2, v1 preserved as tag `v1-final`
- [x] Tokenizer decision: SmolLM2's tokenizer for everything (checked: Base and Instruct vocabularies are identical, 49,152 tokens)
- [x] Data pipeline run on the full data (570M training tokens)
- [x] Model and training loop updated, tested on CPU and on a Colab T4
- [x] Speed test on T4: 535 ms/step (the FULL run averaged about 589 ms/step)
- [x] SHORT baseline (5,000 steps) finished
- [x] FULL baseline (20,000 steps) finished
- [x] `src/evaluate.py` verified against `log.csv` (matches to four decimals for both baselines)
- [x] Clean notebook `notebooks/baseline_train.ipynb` tested in Colab on a T4 (Cells 1 to 6; Cell 5 reproduced the `log.csv` losses)
- [x] Teacher (SmolLM2-360M) scored with `evaluate.py`
- [ ] Distillation runs, instruction fine-tuning, final evaluation (teammates)
- [ ] Optional: custom-BPE tokenizer analysis (see Key design decisions)

## Results so far

Validation loss and perplexity on fixed windows (the same text for every model). Lower is better. The raw numbers are in `tensorix/v2/results/eval/*.json` (written by `src/evaluate.py`) and the training curves in `tensorix/v2/checkpoints/<run>/log.csv`.

| Run | Steps | Tokens seen | C4 loss | C4 ppl | TinyStories loss | TinyStories ppl |
|---|---|---|---|---|---|---|
| `baseline_short` | 5,000 | ~82M | 4.396 | **81.2** | 1.834 | **6.26** |
| `baseline_full` | 20,000 | ~328M | 4.085 | **59.4** | 1.619 | **5.05** |
| Teacher (SmolLM2-360M) | none | none | 3.047 | **21.0** | 1.907 | **6.73** |

Perplexities here use the SmolLM2 tokenizer and are not comparable with the v1 numbers (different vocabulary and data). Add every new run to this table, including distillation runs.

Two observations to keep in mind for the distillation work:

- **C4 has a large student-to-teacher gap** (loss 4.085 versus 3.047 for the FULL baseline), so this is where distillation has room to help.
- **On TinyStories the teacher is worse than the student** (loss 1.907 versus 1.619 for the FULL baseline). Check the reason; a likely explanation is that the student trained on 170M TinyStories tokens while the teacher was not trained for that story style. Consequences: the retention metric below only makes sense on C4, and a KD loss on TinyStories text may pull the student toward a weaker model. Test this (for example KD on C4 windows only, or a lower KD weight on TinyStories) and report both domains separately.

### Efficiency: student versus teacher

Measured with `evaluate.py` on a T4: one forward pass of 1 x 256 tokens, fp16 (the student is converted to fp16 for the benchmark so both models are compared in the same precision).

| Model | Parameters | fp16 weights | Peak GPU memory | Latency |
|---|---|---|---|---|
| Student (`baseline_short`, `baseline_full`) | 50.86M | 101.7 MB | 138.5 MB | about 8-9 ms (7.9 to 9.3 across runs) |
| Teacher SmolLM2-360M | 361.82M | 723.6 MB | 782.9 MB | 47.6 ms |

The student is about 7.1x smaller in parameters, about 5.7x smaller in peak memory and about 5-6x faster per forward pass. Latency varies slightly from run to run on the shared T4, while parameters, memory and losses are stable.

## How the project works (read this first)

| Place | Job |
|---|---|
| **VS Code (laptop)** | Write and edit code |
| **GitHub (this repo)** | Shared source of truth. Code only, never data or checkpoints |
| **Colab (T4 GPU)** | Run training and evaluation |
| **Shared Drive folder `tensorix`** | Big files: data and checkpoints |

The loop: edit in VS Code, then `git add`, `git commit`, `git push` on your own branch, then open a Pull Request and merge it. In Colab, pull the latest code and run it. **Colab's disk is wiped when a session ends**, so nothing important is ever saved only there.

## Repo navigation

```
src/
  data_prep.py         tokenizes C4 + TinyStories into train.bin, val_c4.bin, val_ts.bin, meta.json
  check_tokenizer.py   checks that SmolLM2 Base and Instruct share one vocabulary
  model.py             decoder-only Transformer + load_checkpoint()
  train.py             resumable training loop (baseline now; KD plugs into the loss)
  evaluate.py          scores any student checkpoint or Hugging Face model on the fixed validation windows
notebooks/
  baseline_train.ipynb Colab notebook used for every run (setup, train, results, samples)
configs/               settings files (empty so far)
results/               plots and tables
slides/                blank course PPT template
requirements.txt       Python packages (torch is not pinned; Colab ships its own)
```

Shared Drive folder `tensorix` (not in git). The old `tensorix/data` and `tensorix/checkpoints` folders are the v1 archive and are **incompatible with v2**. Use only `v2/`:

```
tensorix/v2/data/                  train.bin, val_c4.bin, val_ts.bin, meta.json
tensorix/v2/checkpoints/<run>/     ckpt.pt (latest, for resuming), best.pt (lowest val loss), log.csv
tensorix/v2/results/               plots (val_ppl.png) and eval/ (JSON files from evaluate.py)
```

## Key design decisions

- **Tokenizer: SmolLM2's, shared by student and teachers.** Vocabulary 49,152, byte-level BPE. Token-level distillation needs the student and teacher to use the same vocabulary, so there is no id mapping, no `<unk>` and no logit slicing (v1 needed all three). Checked facts: `<|endoftext|>` is both EOS and BOS at id 0, the tokenizer does not add a BOS or EOS by itself, there is **no pad token**, and all ids fit in `uint16`.
- **Custom tokenizer (optional analysis, not done yet):** the one-pager promised a custom tokenizer. Training a custom BPE would break distillation, so it is an analysis only: train a BPE with Hugging Face `tokenizers` on the project corpus and compare tokens per word and compression with SmolLM2's. It is not used for any model.
- **Model:** decoder-only, pre-norm, RMSNorm, RoPE, SwiGLU, tied input and output embeddings, no biases. 8 layers, 8 heads, width 512, SwiGLU hidden size 1,408, context 256 tokens, **50.9M parameters** (about half of them are the embedding matrix because of the 49k vocabulary).
- **Data:** `allenai/c4` (English) for broad web text and `roneneldan/TinyStories` for simple stories, about 70% / 30%. Every document ends with EOS (id 0), and `train.bin` is one long stream of `uint16` tokens. Training samples random 256-token windows, so a window can cross a document boundary.

  | File | Tokens |
  |---|---|
  | `train.bin` | 570,000,122 (C4 400,000,024 from 814,872 documents + TinyStories 170,000,098 from 758,589 documents; mix 0.702 / 0.298) |
  | `val_c4.bin` | 5,001,022 |
  | `val_ts.bin` | 4,735,556 (the whole TinyStories validation split) |

- **Teachers:** `HuggingFaceTB/SmolLM2-360M` (base) for distillation during pretraining, `HuggingFaceTB/SmolLM2-360M-Instruct` for the instruction fine-tuning stage.

## Fixed training budget (do not change)

There are two budget tiers, each with its own matched baseline. Every run in a tier, baseline or distilled, must use identical settings. Only `--out_dir` (and your own new flags) should differ.

| Tier | Use | `--max_steps` | Tokens seen | Baseline run |
|---|---|---|---|---|
| **SHORT** | temperature sweep (T = 1, 2, 4) | 5,000 | ~82M | `baseline_short` |
| **FULL** | baseline, plus KD at the best temperature only | 20,000 | ~328M | `baseline_full` |

Common settings: `--batch_size 64 --block_size 256` (16,384 tokens per step), peak learning rate 1e-3 decaying to 1e-4 (cosine, 200 warm-up steps), AdamW (betas 0.9 / 0.95, weight decay 0.1), gradient clipping 1.0, fp16 autocast with a gradient scaler (the T4 has no bf16), seed 1337.

The learning-rate schedule depends on `--max_steps`, so changing it changes the result. A 5,000-step run is **not** a prefix of a 20,000-step run, which is why SHORT has its own baseline.

**Validation is fixed:** 1,024 windows of 256 tokens per domain, chosen with seed 0, identical in every run and in `evaluate.py`. Do not change `--eval_windows` or `--block_size`, or numbers stop being comparable. `best.pt` is the checkpoint with the lowest average of the C4 and TinyStories validation losses.

**"Matched compute" means the same steps, tokens and student size.** A KD run also pays for the teacher's forward pass, so report that cost separately instead of hiding it.

### Time per run on a T4 (measured and estimated)

| Run | Time |
|---|---|
| Baseline: 535 ms/step in the speed test, about 589 ms/step over the FULL run (measured) | SHORT about 45-50 min, FULL about 3.3 h |
| KD: about 3.3x the baseline step (**estimate**, check against measured numbers) | SHORT about 2.5-3.5 h per temperature, FULL about 10-14 h (check against the first real KD run) |

A full sweep of three temperatures at SHORT is therefore about 8-10 GPU hours (estimate, check), spread over several Colab sessions. Training resumes after a disconnect, so this is workable.

## Before you start (one-time, everyone)

1. Accept the GitHub collaborator invite and the Google Drive invite.
2. Drive, then **Shared with me**, right-click `tensorix`, **Organize**, **Add shortcut**, **My Drive**. Do not rename the folder, because the path `/content/drive/MyDrive/tensorix` must be identical for everyone.
3. Install Git and VS Code, then clone: `git clone https://github.com/sunidhishekar-create/tensorix-kd-slm.git`.
4. Local environment (Windows PowerShell). Local work is for editing and small CPU tests; real training happens on Colab.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If activation fails with an execution-policy error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Using Colab (every new session)

Use **`notebooks/baseline_train.ipynb`**. Open it in Colab (File, Open notebook, GitHub tab, paste the repo URL, choose the branch) and set the runtime to **T4 GPU** first.

1. **Cell 1, Setup.** Run it first in every new session. It mounts Drive, pulls the code, installs requirements, copies the data to Colab's local disk and checks the GPU. A fresh session is an empty machine, so this is always needed.
2. **Cell 2, Train.** Edit `RUN_NAME`, `MAX_STEPS` and `EXTRA_FLAGS`, then run it. It is safe to rerun after a disconnect: rerun Cell 1, then Cell 2 unchanged, and training resumes from the last save.
3. **Cell 3, Results.** Prints the log table and plots validation curves. Add run names to the `RUNS` list to compare runs.
4. **Cell 4, Samples.** Generates text from a checkpoint.
5. **Cell 5, Evaluate.** Scores finished runs (and optionally the teacher) with `src/evaluate.py` and saves one JSON per model to `tensorix/v2/results/eval/`. Never run it while a training run is using the GPU.
6. **Cell 6, Summary table.** Prints all evaluated models side by side.

Rules of thumb:

- Use a **new `RUN_NAME` for every experiment** (see naming below). Reusing a name resumes that run instead of starting a new one.
- Do not use "Run all", and do not rerun `data_prep.py`: the data is already on Drive.
- Do not use "Save a copy in GitHub" from Colab (it would push cell outputs). Edit the notebook locally and push through git.
- Do not run two jobs in one GPU session (for example evaluation while training). They compete for memory and can crash the training run.
- Out of memory: set `EXTRA_FLAGS = "--grad_accum 4"`. The effective batch stays 64.
- Run names: `baseline_short`, `baseline_full`, `kd_short_T1`, `kd_short_T2`, `kd_short_T4`, `kd_full_T<best>`, `sft_<what>`.

### GPU quota

The free Colab tier gives a T4 GPU but limits its use. The limits are not published and change over time, and heavy recent use makes the message "Cannot connect to GPU backend" more likely. Colab's own advice is to wait a day for the quota to reset.

- Click **Close** on that message. Do not choose "Connect without GPU": training on a CPU is far too slow and the cell would keep writing to the run folder.
- Check the log line `device=cuda` at the start of every run. `device=cpu` means no GPU is attached: interrupt the cell.
- Disconnect the runtime (Runtime, Disconnect and delete runtime) as soon as a run or evaluation finishes. Idle GPU sessions waste quota.
- Each person runs their share of the work from their own Google account. Colab's FAQ lists using multiple accounts to work around usage limits as disallowed, so one person should not log in to several accounts to get more GPU time.
- Runs can be interrupted at any point: rerun Cell 1 and Cell 2 unchanged and training resumes from the last save (every 500 steps for SHORT, every 1,000 for FULL).

Each run folder has `ckpt.pt` (full state for resuming), `best.pt` (model and config only, use this for evaluation and as a starting point) and `log.csv` (step, train loss, validation losses and perplexities, learning rate).

## Notes for teammates

### Loading and using the student

```python
from model import load_checkpoint        # run from src/, or add src to sys.path
model, ck = load_checkpoint("<run>/best.pt", device="cuda")   # works for best.pt and ckpt.pt
logits, loss = model(x, y)               # logits (B, T, 49152); pass y to also get the loss
logits, loss, hiddens = model(x, y, return_hidden=True)   # hiddens: list of 8 block outputs, each (B, T, 512)
text_ids = model.generate(prompt_ids, max_new_tokens=80)
```

`load_checkpoint` rebuilds the model from the config stored inside the checkpoint, so never hardcode sizes. The model takes ids shaped `(B, T)` with `T <= 256`.

### Distillation (teacher and KD loss)

- **Load the teacher frozen:**

```python
import torch
from transformers import AutoModelForCausalLM
teacher = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M", torch_dtype=torch.float16).cuda().eval()   # the Colab transformers version warns that torch_dtype is deprecated: use dtype= if it is accepted (check)
for p in teacher.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    t_logits = teacher(input_ids=x).logits      # (B, T, 49152), same ids as the student
```

- **No id mapping.** The student and the teacher use the same vocabulary and the same ids, and the logits have the same shape. The Instruct model has the identical vocabulary.
- **Where to hook in:** in `train.py`, the micro-batch loop currently does `_, loss = model(x, y)`. Change it to `logits, ce = model(x, y)` and add your KD term behind a flag (for example `--kd_alpha`, `--kd_temp`, `--kd_kl forward|reverse`). Keep every other setting identical to the baseline.
- **Loss:** `alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * cross-entropy`. Compute the softmax and KL in float32. Forward KL first; reverse KL if time allows (check the MiniLLM paper for the trade-offs).
- **Memory:** the teacher and student logits for 32 x 256 x 49,152 values are large, and the extra float32 copies add more. Check the memory use of a KD micro-batch on the T4: `--grad_accum 4` or more may be needed for KD runs.
- **Feature-based KD (optional):** project the student's 512-wide hidden states to the teacher's hidden size with a learned linear layer. Check the teacher's hidden size and layer count in `AutoConfig` before designing the projection. Check: SmolLM2-360M is reportedly 960 wide with 32 layers, which would also need a student-to-teacher layer mapping.
- **Temperature sweep:** T = 1, 2, 4 at the SHORT tier, then run only the best T at FULL. Always compare against `baseline_short` or `baseline_full` of the same tier.
- Do not change batch size, block size, steps or validation settings. Score every run with `src/evaluate.py` (below), and add it to the Results table.

### Instruction fine-tuning

- **There is no pad token.** EOS and BOS are both id 0. Pick a pad id explicitly (for example 0) and mask padded positions in the loss. Pretraining uses packed windows and needs no padding.
- **Context is 256 tokens.** Check how many Dolly 15k examples exceed 256 tokens. Truncate or drop them, and state which in the write-up.
- **Chat format:** the Instruct teacher uses a chat template. Check which special tokens it uses (ChatML-style is likely) and check that they exist in the shared vocabulary: inspect `tok.apply_chat_template` and the token ids of the SmolLM2-360M-Instruct tokenizer before choosing a prompt format. Check also that the student has not seen these tokens in pretraining (expected: they are untrained until fine-tuning).
- Fine-tune both the baseline and the best distilled model with identical settings. Use `SmolLM2-360M-Instruct` as the teacher for the instruction stage.

### Evaluation

`src/evaluate.py` scores any model on the same fixed validation windows and writes one JSON per model (perplexity on C4 and TinyStories, parameter count, fp16 weight size, peak GPU memory, latency of a 1 x 256 forward pass).

```python
EVAL_DIR = f"{DRIVE}/results/eval"      # in the notebook, after Cell 1
!python src/evaluate.py --data_dir /content/data --ckpt {DRIVE}/checkpoints/<run>/best.pt --name <run> --out_dir {EVAL_DIR}
!python src/evaluate.py --data_dir /content/data --hf_model HuggingFaceTB/SmolLM2-360M --name teacher_base --out_dir {EVAL_DIR}
```

- **In the notebook:** Cell 5 runs `evaluate.py` for a list of runs (and the teacher with `EVAL_TEACHER = True`), and Cell 6 prints the summary table.
- **Consistency check (done for both baselines, repeat for any new kind of run):** the `val_c4_loss` and `val_ts_loss` from `evaluate.py` must match the last row of the run's `log.csv` to about three decimals (for `baseline_short`: 4.396 and 1.834). If they do not, stop and tell Sunidhi.
- **Retention metric**, per domain, from the JSON losses:

```
gap closed = (loss_baseline - loss_distilled) / (loss_baseline - loss_teacher)
```

0 means distillation gave no gain over the baseline and 1 means the student matched the teacher. **Use it on C4 only:** on TinyStories the teacher's loss (1.907) is higher than the baseline's (1.619), so the denominator is negative and the number is meaningless. Report TinyStories as plain loss and perplexity. Also report the size, memory and latency ratios between student and teacher.

- **Still to build (evaluation owner):** a blind pairwise comparison of baseline versus distilled responses on held-out prompts, and the final plots and tables in `results/`. The notebook's Cell 3 already plots validation curves from any list of runs.
- Run evaluation in its own session, not while a training run is using the GPU.

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
- Mostly edit your own files to avoid merge conflicts. If you need to change `model.py` or `train.py` (beyond the KD flag described above), tell Sunidhi first, because everyone's runs depend on them.
- Never commit data, checkpoints, tokens, API keys or SRNs. The repo is public. Keep any Hugging Face token in Colab's Secrets panel, never in code.
- Test small first (a few steps, small batch) before spending Colab time.

**Branch note:** the v2 foundation was merged into `main` (the working branch was `v2-rebuild`). Branches written against v1 (for example `feature/distillation`) assumed the old pruned-GPT-2 vocabulary, `vocab.json` and `gpt2_ids_kept`. Those no longer exist. Rebase or recreate such branches on top of `main`, and delete any id-mapping code. Old `feature/*` branches are v1 history and can be ignored.

## Rebuilding the data (only if Drive is lost)

The data already lives in `tensorix/v2/data/`. To rebuild it from scratch on Colab (about 30 minutes, streamed from Hugging Face; check: an optional `HF_TOKEN` in Colab Secrets may speed up the streaming):

```
python src/data_prep.py --out /content/data
```

It skips finished pieces on a rerun. Then copy `train.bin`, `val_c4.bin`, `val_ts.bin` and `meta.json` (not the intermediate `c4_train.bin` and `ts_train.bin`) to `tensorix/v2/data/`.

## Final deliverables

**PPT:** start from `slides/AFML-ppt-template.pptx`. A progress version covering the foundation work is kept on the shared Drive, so add the distillation and results slides to it. Suggested owners: architecture and tokenizer by Sunidhi, distillation by the distillation owner, fine-tuning and results by the fine-tuning owner, structure and polish by the evaluation owner.

**5-minute demo video:** one screen recording with narration, about 30 seconds each for the problem, the model, baseline versus distilled results, a live generation from both models side by side, and the conclusions. Record last, once results exist.

## References

1. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network (2015). https://arxiv.org/abs/1503.02531
2. Sanh, Debut, Chaumond, Wolf. DistilBERT, a distilled version of BERT (2019). https://arxiv.org/abs/1910.01108
3. Gu, Dong, Wei, Huang. MiniLLM: Knowledge Distillation of Large Language Models (2023, ICLR 2024). https://arxiv.org/abs/2306.08543
4. Hugging Face. SmolLM2 models (teacher and tokenizer): https://huggingface.co/HuggingFaceTB
5. Datasets: `allenai/c4` (English), `roneneldan/TinyStories`, `databricks/databricks-dolly-15k`.