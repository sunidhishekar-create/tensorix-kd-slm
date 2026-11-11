# Supplementary experiment: Gemma 4 answers versus human answers for instruction fine-tuning

This experiment adds a second kind of distillation to the main project. The main experiment distils from SmolLM2-360M during pretraining by matching the teacher's token probabilities (it needs the teacher's logits and a shared tokenizer). Here the teacher is **Gemma 4**, reached through an API that returns text only. Gemma 4 writes answers to instruction prompts, and the pretrained student is fine-tuned on those answers. This is *sequence-level* (synthetic-data) distillation.

Everything lives in `experiments/gemma_sft/`. Nothing in `src/` or the main README was changed for it. The scripts import the model from `src/model.py`.

## Question

Does fine-tuning the pretrained student (`baseline_full`, 50.9M parameters) on Gemma 4's answers produce a better instruction-following model than fine-tuning it on the original human-written answers, when everything else is identical?

## Design

- The pretrained student is fine-tuned twice from the same checkpoint on the **same prompts**, with identical settings, seed and prompt order. Only the answer source differs.
  - **Model A:** human answers (Databricks Dolly 15k).
  - **Model B:** answers written by `gemma-4-26b-a4b-it`.
- Both are compared with the pretrained model (no fine-tuning) on 300 held-out prompts by loss, answer statistics, and blind pairwise judging.

## Data

| Step | Count |
|---|---|
| Dolly 15k examples | 15,011 |
| Fit in 256 tokens (prompt + answer + end token, prompt at most 160 tokens) | 10,922 |
| Chosen (fixed seed 0) | 3,000 training prompts and 300 held-out prompts |
| Kept after requiring both the human and the Gemma version to fit | 2,990 training and 300 held-out |
| Training / validation split used by fine-tuning | 2,790 training and 200 validation (the last 200 training prompts, the same for A and B) |

Gemma's answers are longer: about 67.5 tokens against 58.0 on the training set, and 71.0 against 58.8 on the held-out set. Prompt format is plain text (`### Instruction:` / `### Context:` / `### Response:`) with the end token (id 0) after the answer. The loss is taken on answer tokens only.

The length filter removes most long-context tasks (summarization, extraction, closed QA), so the experiment covers short-context tasks only.

## Teacher and API notes

- Model used: `gemma-4-26b-a4b-it` through the Gemini API (free tier at the time of use; check current terms and limits). `gemma-4-31b-it` was listed for the key but returned HTTP 500 consistently during testing, so the 26B mixture-of-experts model was used for all answers. Mixing both teachers in one dataset was avoided on purpose.
- Gemma 4 thinks by default, and thinking tokens count toward the output limit, which produced blank answers. Thinking is switched off with `thinking_level` set to minimal (`thinking_budget` is rejected for Gemma 4). Default thinking mode also coincided with HTTP 500 errors.
- The API returns intermittent 500 errors. The generation script retries (5, 15, 45, 120 s), has a 90 s request timeout (a stalled request once hung a run for hours), saves every answer immediately and skips finished prompts on rerun.
- Gemma's answers contain factual errors, for example a wrong instrument classification and invented details. A student trained on them can copy those errors.
- Prompts are from a public dataset. Do not send private data to the free tier (check its data-use terms). Check the Gemma terms for using model outputs to train another model, and state the terms in the final report.
- The API key is read from the `GEMINI_API_KEY` environment variable only. It must never be written to a file in the repo.

## Pipeline

| Script | Job | Runs on |
|---|---|---|
| `common.py` | Shared prompt format, request text, JSONL helpers | |
| `01_test_call.py`, `01b_check_models.py`, `01c_thinking_test.py`, `01d_isolate.py` | One-call API tests and diagnostics (key, model list, thinking, 500 errors) | laptop |
| `02_make_prompts.py` | Builds the fixed prompt set from Dolly | laptop |
| `03_generate_answers.py` | Gemma answers, resumable, rate-limited | laptop (API only) |
| `show_answers.py` | Prints generated answers next to the human ones | laptop |
| `04_tokenize_sft.py` | Builds the two datasets on the same prompts (answer-only loss masks) | laptop |
| `05_finetune.py` | Fine-tunes on one dataset, validation split, keeps the best epoch | Colab GPU |
| `finetune_colab.ipynb` | Colab notebook for fine-tuning and generation | Colab GPU |
| `06_generate_eval.py` | Answers to the 300 held-out prompts from several models, same decoding and seed | Colab GPU |
| `07_judge.py` | Blind pairwise LLM judging (both orders, optional forced choice), human-check export | laptop (API only) |
| `08_report.py` | Builds `results/RESULTS.md` from all result files | laptop |

## Settings

- Fine-tuning: AdamW, learning rate 1e-4 decaying (cosine, 20 warm-up steps) to 1e-5, batch 16, up to 3 epochs, weight decay 0.01, fp16, seed 1337. The epoch with the lowest validation loss is kept (epoch 1 for both models). The held-out sets are not used for that choice.
- A first attempt (learning rate 2e-4, 3 epochs, no early stopping) overfit: held-out loss was lowest after epoch 1 and rose afterwards while training loss kept falling. The settings above were chosen after that attempt and are identical for A and B.
- Generation: temperature 0.7, top-k 40, at most 100 new tokens, stop at the end token, the same random seed per prompt for every model.
- Judge: `gemma-4-26b-a4b-it`, temperature 0, thinking off. Each pair is judged twice per prompt with the answers swapped. A model wins a prompt only if chosen in both orders. Statistics: exact two-sided sign test and a 95% Wilson interval.

## Results

Final numbers are generated into [`results/RESULTS.md`](results/RESULTS.md) by `08_report.py`. Main findings so far:

**Held-out loss per answer token** (lower is better; epoch kept for both models: 1)

| | Pretrained | A (human answers) | B (Gemma answers) |
|---|---|---|---|
| Loss on human answers | 3.929 (ppl 50.9) | **3.689** (ppl 40.0) | 3.886 (ppl 48.7) |
| Loss on Gemma answers | 4.001 (ppl 54.7) | 3.954 (ppl 52.1) | **3.594** (ppl 36.4) |

Each fine-tuned model is best at imitating its own training answers, and both improve slightly on the other source's answers. The two rows use different reference answers, so the losses cannot be compared across rows as a measure of quality.

**Blind judging, A against B (300 held-out prompts)**

| Protocol | A wins | B wins | Ties | Result |
|---|---|---|---|---|
| Tie allowed | 10 | 11 | 279 | Judge called 93% equal; no difference |
| Forced choice | 99 | 114 | 87 (order flips) | B wins 53.5% of decisive prompts (about 47% to 60%), sign test p = 0.34: not significant |

Neither protocol found a significant difference in answer quality between A and B.

**Validity check: the same judge against the pretrained model** (forced choice, first 100 held-out prompts)

| Pair | Fine-tuned wins | Pretrained wins | Ties (order flips) | Win rate, excl. ties (95% interval) | Sign-test p |
|---|---|---|---|---|---|
| A (human answers) vs pretrained | 68 | 10 | 22 | 87.2% (78% to 93%) | < 0.001 |
| B (Gemma answers) vs pretrained | 84 | 3 | 13 | 96.6% (90% to 99%) | < 0.001 |

The judge strongly prefers both fine-tuned models over the pretrained model, whose answers repeat the question in a loop. It can therefore detect large quality differences, which makes the near-tie between A and B informative: at this scale the two answer sources produce answers of similar quality. B beats the pretrained model somewhat more clearly than A does (96.6% against 87.2%; exact test on the decisive counts, p about 0.04), but this indirect comparison is weaker than the direct A-versus-B test (53.5%, p = 0.34), and a Gemma judge may favour Gemma-like text, so it is not evidence that B is better.

See `RESULTS.md` for the full tables, the answer statistics and the human check (if rater sheets were collected).

**Qualitative:** fine-tuning teaches the answer format (the pretrained model repeats the question in a loop), but answers from both fine-tuned models are weak: fluent but often off-topic or invented. A 51M model fine-tuned on about 2,800 examples is not expected to follow instructions reliably.

## What can and cannot be claimed

- Can be claimed: fine-tuning on either source reduces loss on its own kind of answer, both fine-tuned models are clearly preferred over the pretrained model by the blind judge (87% and 97% of decisive prompts), and the judge found no significant quality difference between the two sources at this scale.
- Cannot be claimed: that Gemma's answers are better or worse teaching material. The judge is itself Gemma (it may favour Gemma-like style, check with the human sheets) and shows position bias on about 29% of prompts in forced mode.

## Limitations

- Small student and small fine-tuning set; the answers of both models are low quality.
- Single run per model (one seed); run-to-run variation was not measured.
- One decoding setting; greedy decoding was not tried.
- The judge is Gemma, so self-preference is possible. A different-family judge was not run. The comparison against the pretrained model used only the first 100 prompts.
- Gemma answers are 16 to 21% longer, which affects both loss and judging.
- Gemma's answers contain factual errors, and the 31B model was unavailable.
- Short-context prompts only (length filter).

## Reproduce

Laptop (PowerShell, `.venv` active, key set from a file outside the repo):

```powershell
pip install -r experiments\gemma_sft\requirements.txt
$env:GEMINI_API_KEY = (Get-Content $env:USERPROFILE\apikey.txt).Trim()
python experiments\gemma_sft\02_make_prompts.py
python experiments\gemma_sft\03_generate_answers.py --rpm 10 --daily_limit 1400    # repeat until 3300/3300
python experiments\gemma_sft\04_tokenize_sft.py
```

Copy `data/sft/*.npz` and `prompts.jsonl` to `tensorix/v2/sft/` on Drive. In Colab, open `finetune_colab.ipynb`: run Cell 1 (setup), Cell 2 (model A), Cell 3 (model B) and Cell 5 (generation) one at a time. Then, on the laptop, with `generations.jsonl` downloaded to `data/sft/`:

```powershell
python experiments\gemma_sft\07_judge.py --pairs human:gemma
python experiments\gemma_sft\07_judge.py --pairs human:gemma --forced --judgments data\sft\judgments_forced.jsonl
python experiments\gemma_sft\07_judge.py --pairs human:pretrained gemma:pretrained --forced --n 100 --judgments data\sft\judgments_pretrained.jsonl
python experiments\gemma_sft\07_judge.py --pairs human:gemma --export_human 40
python experiments\gemma_sft\08_report.py
```

Data, answers, checkpoints and the key are never committed: `data/` is gitignored, and large files live on Drive under `tensorix/v2/sft/`.