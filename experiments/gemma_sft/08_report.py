"""Build the experiment's results report (RESULTS.md) from whatever result files exist.

Sections that need a missing file are marked "not available", so the script can be run at any time and
rerun when new files arrive (for example filled human-check sheets).

  python experiments/gemma_sft/08_report.py

Reads (all under data/sft/ unless noted):
  generations.jsonl, gemma_answers.jsonl                  answer statistics
  judgments.jsonl, judgments_forced.jsonl, judgments_pretrained.jsonl   LLM-judge results
  runs/human/{log.csv,sft_result.json}, runs/gemma/{...}  fine-tuning losses (download from Drive)
  human_check_<rater>.csv + human_check_key.csv           optional human check
Writes: experiments/gemma_sft/results/RESULTS.md (and copies of the small result files).
"""
import argparse, csv, glob, json, math, re, shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import read_jsonl

LABEL = {"human": "A (human answers)", "gemma": "B (Gemma answers)", "pretrained": "pretrained (no fine-tuning)"}


# ---------- helpers ----------
def words(t):
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def f1(a, b):
    overlap = sum((Counter(a) & Counter(b)).values())
    if not a or not b or overlap == 0:
        return 0.0
    p, r = overlap / len(a), overlap / len(b)
    return 2 * p * r / (p + r)


def has_repeat(ws, n=3):
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    return len(grams) != len(set(grams))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def sign_test_p(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def decide(v1, v2, x, y):
    w1, w2 = {"A": x, "B": y}.get(v1), {"A": y, "B": x}.get(v2)
    return w1 if (w1 is not None and w1 == w2) else None


def judged(path):
    """pair -> {prompt id -> winner or None}, only prompts judged in both orders."""
    by = defaultdict(lambda: defaultdict(dict))
    for j in read_jsonl(path):
        by[j["pair"]][j["id"]][j["order"]] = j["verdict"]
    out = {}
    for pair, d in by.items():
        x, y = pair.split(":")
        out[pair] = {pid: decide(o[1], o[2], x, y) for pid, o in d.items() if 1 in o and 2 in o}
    return out


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


# ---------- sections ----------
def section_losses(runs_dir):
    lines = ["## 1. Fine-tuning: held-out answer-token loss", ""]
    rows, ok = [], True
    for name in ("human", "gemma"):
        res, log = Path(runs_dir) / name / "sft_result.json", Path(runs_dir) / name / "log.csv"
        if not res.exists() or not log.exists():
            ok = False
            break
        r = json.load(open(res))
        epoch0 = next(csv.DictReader(open(log)))
        rows.append((name, r, epoch0))
    if not ok:
        return lines + [f"Not available: download `log.csv` and `sft_result.json` of `sft_human_v2` and "
                        f"`sft_gemma_v2` from Drive into `{runs_dir}/human/` and `{runs_dir}/gemma/`.", ""]
    lines += ["Loss per answer token on the 300 held-out prompts (lower is better). The epoch kept for each "
              "model is the one with the lowest loss on a 200-prompt validation split.", "",
              "| Model | Epoch kept | Loss on human answers | Loss on Gemma answers |", "|---|---|---|---|"]
    e0 = rows[0][2]
    lines.append(f"| pretrained (epoch 0) | 0 | {float(e0['human_eval_loss']):.3f} "
                 f"(ppl {math.exp(float(e0['human_eval_loss'])):.1f}) | {float(e0['gemma_eval_loss']):.3f} "
                 f"(ppl {math.exp(float(e0['gemma_eval_loss'])):.1f}) |")
    for name, r, _ in rows:
        s = r["selected_eval_loss"]
        lines.append(f"| {LABEL[name]} | {r['best_epoch']} | {s['human_eval']:.3f} (ppl {math.exp(s['human_eval']):.1f}) "
                     f"| {s['gemma_eval']:.3f} (ppl {math.exp(s['gemma_eval']):.1f}) |")
    lines += ["", "Losses against different reference answers are not directly comparable: Gemma's answers are "
              "more uniform and therefore easier to predict. Loss does not measure answer quality.", ""]
    return lines


def section_stats(gens, gemma_ref):
    lines = ["## 2. Answer statistics on the held-out prompts", ""]
    if not gens:
        return lines + ["Not available: `generations.jsonl` not found.", ""]
    names = list(gens[0]["answers"])
    lines += ["| Model | Mean words | Empty | Repeated 3-gram | Distinct bigrams | Word F1 vs human reference | "
              "Word F1 vs Gemma reference |", "|---|---|---|---|---|---|---|"]
    for n in names:
        ws = [words(g["answers"][n]) for g in gens]
        bigrams = [tuple(w[i:i + 2]) for w in ws for i in range(len(w) - 1)]
        f_h = sum(f1(w, words(g["human_reference"])) for w, g in zip(ws, gens)) / len(gens)
        gem = [(w, words(gemma_ref[g["id"]])) for w, g in zip(ws, gens) if g["id"] in gemma_ref]
        f_g = sum(f1(w, r) for w, r in gem) / len(gem) if gem else None
        lines.append(f"| {LABEL.get(n, n)} | {sum(map(len, ws)) / len(ws):.1f} | {pct(sum(not w for w in ws) / len(ws))} | "
                     f"{pct(sum(has_repeat(w) for w in ws) / len(ws))} | "
                     f"{pct(len(set(bigrams)) / len(bigrams) if bigrams else None)} | {f_h:.3f} | "
                     f"{'n/a' if f_g is None else f'{f_g:.3f}'} |")
    lines += ["", "Repeated 3-gram: share of answers in which some three-word sequence occurs twice (degenerate "
              "looping). Word F1 is the unigram overlap with a reference answer: higher means closer to that "
              "reference, not necessarily better.", ""]
    return lines


def section_judge(title, path, gens):
    lines = [f"### {title}", ""]
    if not Path(path).exists():
        return lines + [f"Not available: `{path}` not found.", ""]
    res = judged(path)
    lines += ["| Pair | Prompts | First model wins | Second model wins | Ties | Win rate of leader (excl. ties) "
              "| 95% interval | Sign-test p |", "|---|---|---|---|---|---|---|---|"]
    for pair, d in sorted(res.items()):
        x, y = pair.split(":")
        c = Counter(d.values())
        wx, wy, tie = c[x], c[y], c[None]
        dec = wx + wy
        k = max(wx, wy)
        ci = wilson(k, dec)
        leader = x if wx >= wy else y
        lines.append(f"| {LABEL.get(x, x)} vs {LABEL.get(y, y)} | {len(d)} | {wx} | {wy} | {tie} | "
                     f"{pct(k / dec) if dec else 'n/a'} ({leader if dec else '-'}) | "
                     f"{'n/a' if ci is None else f'{100 * ci[0]:.0f}% to {100 * ci[1]:.0f}%'} | "
                     f"{sign_test_p(k, dec):.3f} |")
    return lines + [""]


def section_human(data_dir, forced_path):
    lines = []
    key_path = Path(data_dir) / "human_check_key.csv"
    sheets = sorted(p for p in glob.glob(str(Path(data_dir) / "human_check_*.csv")) if not p.endswith("_key.csv"))
    if not key_path.exists() or not sheets:
        return lines + ["Not available: no filled sheets named `human_check_<rater>.csv` found in "
                        f"`{data_dir}`.", ""]
    key = {r["id"]: (r["answer_1_is"], r["answer_2_is"]) for r in csv.DictReader(open(key_path, encoding="utf-8-sig"))}
    judge = judged(forced_path).get("human:gemma", {}) if Path(forced_path).exists() else {}
    raters, lines_t = {}, []
    lines_t += ["| Rater | Rated | A (human) better | B (Gemma) better | Tie | Agreement with forced LLM judge |",
                "|---|---|---|---|---|---|"]
    for p in sheets:
        votes = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            v = (r.get("better (1, 2 or tie)") or "").strip().lower()
            if r["id"] in key and v in ("1", "2", "tie"):
                votes[r["id"]] = None if v == "tie" else key[r["id"]][int(v) - 1]
        raters[Path(p).stem.replace("human_check_", "")] = votes
        c = Counter(votes.values())
        both = [(w, judge.get(int(i))) for i, w in votes.items() if w and judge.get(int(i))]
        agree = f"{sum(a == b for a, b in both)}/{len(both)}" if both else "n/a"
        lines_t.append(f"| {Path(p).stem.replace('human_check_', '')} | {len(votes)} | {c['human']} | {c['gemma']} | "
                       f"{c[None]} | {agree} |")
    lines += lines_t
    names = list(raters)
    if len(names) >= 2:
        lines += ["", "Agreement between raters on prompts both rated (excluding ties):", ""]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                common = [k for k in raters[names[i]] if k in raters[names[j]]
                          and raters[names[i]][k] and raters[names[j]][k]]
                same = sum(raters[names[i]][k] == raters[names[j]][k] for k in common)
                lines.append(f"- {names[i]} and {names[j]}: {same}/{len(common)}")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/sft")
    ap.add_argument("--out_dir", default="experiments/gemma_sft/results")
    a = ap.parse_args()
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gens = read_jsonl(d / "generations.jsonl")
    gemma_ref = {r["id"]: r["answer"] for r in read_jsonl(d / "gemma_answers.jsonl")}
    meta_p = d / "generations.meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}

    lines = ["# Gemma SFT experiment: results", "",
             "Generated by `08_report.py`. Do not edit by hand: rerun the script when new files arrive.", ""]
    lines += section_losses(d / "runs")
    lines += section_stats(gens, gemma_ref)
    lines += ["## 3. Blind pairwise judging", "",
              "The judge sees two answers without model names, once in each order. A model wins a prompt only "
              "if it is chosen in both orders; otherwise the prompt is a tie. The interval is a 95% Wilson "
              "interval for the leader's win rate among decisive prompts, and the p-value is an exact two-sided "
              "sign test (small p means a real difference).", ""]
    lines += section_judge("3a. Tie allowed (judge may answer 'equal')", d / "judgments.jsonl", gens)
    lines += section_judge("3b. Forced choice (judge must pick one)", d / "judgments_forced.jsonl", gens)
    lines += section_judge("3c. Validity check against the pretrained model (forced choice, first prompts only)",
                           d / "judgments_pretrained.jsonl", gens)
    lines += ["## 4. Human check", ""]
    lines += section_human(d, d / "judgments_forced.jsonl")
    if meta:
        lines += ["## 5. Settings", "", f"Decoding: temperature {meta.get('temperature')}, top-k {meta.get('top_k')}, "
                  f"up to {meta.get('max_new_tokens')} new tokens, seed {meta.get('seed')} per prompt "
                  f"(identical for every model).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    for name in ("judgments.jsonl", "judgments_forced.jsonl", "judgments_pretrained.jsonl",
                 "judgments_summary.json", "judgments_forced_summary.json", "judgments_pretrained_summary.json",
                 "generations.meta.json", "prompts_meta.json", "sft_meta.json"):
        if (d / name).exists():
            shutil.copy(d / name, out / name)
    for n in ("human", "gemma"):
        for f in ("log.csv", "sft_result.json"):
            src = d / "runs" / n / f
            if src.exists():
                (out / n).mkdir(exist_ok=True)
                shutil.copy(src, out / n / f)
    print(f"wrote {out / 'RESULTS.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
