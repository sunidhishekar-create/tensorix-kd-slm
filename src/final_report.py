"""Final evaluation report: tables (RESULTS.md, final_metrics.json) and figures for the slides.

CPU only, no torch. Reads whatever result files exist and skips sections whose inputs are missing, so it
can be rerun whenever a new file arrives.

  python src/final_report.py --eval_dir results/final/eval --bench_dir results/final/bench \
      --log_dir results/final/logs --judge results/final/judge/*_summary.json \
      --gens results/final/generations.jsonl --sft_json results/final/sft_comparison.json \
      --out results/final

Inputs
  --eval_dir   one JSON per model from src/evaluate.py (perplexity, size, memory, latency)
  --log_dir    training curves: <dir>/<run>_log.csv or <dir>/<run>/log.csv (train.py format)
  --bench_dir  one JSON per model from src/bench_mc.py (HellaSwag, ARC-Easy)
  --judge      *_summary.json files from experiments/gemma_sft/07_judge.py
  --gens       generations.jsonl from experiments/gemma_sft/06_generate_eval.py (answer statistics)
  --sft_json   sft_comparison.json written by the Dolly SFT notebook (Section 13)
"""
import argparse, csv, json, math, os, re, tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# ---------------- names, tiers and colours (one fixed colour per model, everywhere) ----------------
MODELS = {   # key: (label, tier, colour)
    "baseline_short": ("Baseline SHORT", "SHORT", "#2a78d6"),
    "kd_short_T2":    ("KD SHORT T=2", "SHORT", "#eb6834"),
    "kd_short_T4":    ("KD SHORT T=4", "SHORT", "#1baf7a"),
    "kd_short_T1":    ("KD SHORT T=1", "SHORT", "#6250d6"),
    "baseline_full":  ("Baseline FULL", "FULL", "#eda100"),
    "kd_full_T2":     ("KD FULL T=2", "FULL", "#e87ba4"),
    "teacher_base":   ("Teacher SmolLM2-360M", None, "#52514e"),
}
SFT_MODELS = {
    "sft_base_short": ("SFT on Baseline SHORT", "#2a78d6"),
    "sft_kd_T2":      ("SFT on KD SHORT T=2", "#eb6834"),
    "sft_base_full":  ("SFT on Baseline FULL", "#eda100"),
    "pretrained":     ("No fine-tuning", "#52514e"),
}
TIER_BASELINE = {"SHORT": "baseline_short", "FULL": "baseline_full"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TIE_GREY = "#c3c2b7"


def label(k):
    return MODELS.get(k, SFT_MODELS.get(k, (k,)))[0]


def colour(k):
    if k in MODELS:
        return MODELS[k][2]
    return SFT_MODELS.get(k, (k, "#888888"))[1]


def style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


# ---------------- statistics ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------- loading ----------------
def read_log(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    dedup = {}
    for r in rows:                     # a resume can repeat a step: keep the last row for each step
        dedup[int(r["step"])] = r
    return [dedup[s] for s in sorted(dedup)]


def find_logs(log_dirs):
    logs = {}
    for d in log_dirs:
        d = Path(d)
        for p in sorted(d.glob("*_log.csv")) + sorted(d.glob("*/log.csv")):
            run = p.stem[:-4] if p.name.endswith("_log.csv") else p.parent.name
            if run in MODELS and run not in logs:
                logs[run] = read_log(p)
    return logs


def load_eval(eval_dirs, logs):
    """evaluate.py JSONs; falls back to the last log.csv row for losses (source recorded)."""
    ev = {}
    for d in eval_dirs:
        for p in sorted(Path(d).glob("*.json")):
            try:
                j = json.load(open(p))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if "val_c4_loss" in j and j.get("name") in MODELS and j["name"] not in ev:
                ev[j["name"]] = dict(j, _source=str(p))
    for run, rows in logs.items():
        if run not in ev and rows:
            last = rows[-1]
            ev[run] = {"name": run, "step": int(last["step"]), "val_c4_loss": round(last["val_c4"], 4),
                       "val_c4_ppl": round(math.exp(last["val_c4"]), 3), "val_ts_loss": round(last["val_ts"], 4),
                       "val_ts_ppl": round(math.exp(last["val_ts"]), 3), "_source": "log.csv (last row)"}
    return ev


# ---------------- sections ----------------
def ordered(keys):
    order = list(MODELS)
    return sorted(keys, key=lambda k: order.index(k) if k in order else 99)


def section_quality(ev, md, metrics):
    if not ev:
        md.append("## 1. Language-modelling quality\n\nNot available: no evaluate.py JSON or log.csv found.\n")
        return
    teacher = ev.get("teacher_base")
    md += ["## 1. Language-modelling quality (fixed validation windows, lower is better)", "",
           "1,024 windows of 256 tokens per domain, seed 0, identical for every model. Gap closed on C4 = "
           "(loss_baseline - loss_model) / (loss_baseline - loss_teacher), against the baseline of the same tier. "
           "It is not computed on TinyStories, where the teacher is worse than the students.", "",
           "| Model | Tier | Step | C4 loss | C4 ppl | C4 ppl vs tier baseline | TinyStories loss | TinyStories ppl "
           "| TS ppl vs tier baseline | Gap closed on C4 | Source |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    metrics["quality"] = {}
    for k in ordered(ev):
        e = ev[k]
        tier = MODELS[k][1]
        base = ev.get(TIER_BASELINE.get(tier, ""), None)
        c4_chg = ts_chg = gap = None
        if base and k != TIER_BASELINE.get(tier):
            c4_chg = (e["val_c4_ppl"] / base["val_c4_ppl"] - 1) * 100
            ts_chg = (e["val_ts_ppl"] / base["val_ts_ppl"] - 1) * 100
            if teacher and abs(base["val_c4_loss"] - teacher["val_c4_loss"]) > 1e-6:
                gap = (base["val_c4_loss"] - e["val_c4_loss"]) / (base["val_c4_loss"] - teacher["val_c4_loss"]) * 100
        is_base = k == TIER_BASELINE.get(tier)
        md.append(f"| {label(k)} | {tier or '-'} | {e.get('step', '-') if e.get('step') is not None else '-'} "
                  f"| {e['val_c4_loss']:.4f} | {e['val_c4_ppl']:.1f} "
                  f"| {'reference' if is_base else (f'{c4_chg:+.1f}%' if c4_chg is not None else '-')} "
                  f"| {e['val_ts_loss']:.4f} | {e['val_ts_ppl']:.2f} "
                  f"| {'reference' if is_base else (f'{ts_chg:+.1f}%' if ts_chg is not None else '-')} "
                  f"| {'0% (reference)' if is_base else (f'{gap:.1f}%' if gap is not None else '-')} "
                  f"| {'evaluate.py' if e['_source'].endswith('.json') else e['_source']} |")
        metrics["quality"][k] = {"c4_loss": e["val_c4_loss"], "c4_ppl": e["val_c4_ppl"], "ts_loss": e["val_ts_loss"],
                                 "ts_ppl": e["val_ts_ppl"], "c4_ppl_change_pct": c4_chg, "ts_ppl_change_pct": ts_chg,
                                 "c4_gap_closed_pct": gap, "source": e["_source"]}
    md.append("")


def fig_quality(ev, out):
    keys = ordered(ev)
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dom, title in ((axes[0], "c4", "C4 validation perplexity"),
                           (axes[1], "ts", "TinyStories validation perplexity")):
        vals = [ev[k][f"val_{dom}_ppl"] for k in keys]
        y = range(len(keys))
        ax.barh(y, vals, color=[colour(k) for k in keys], height=0.62, edgecolor="white", linewidth=2)
        ax.set_yticks(list(y), [label(k) for k in keys])
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}" if dom == "c4" else f" {v:.2f}", va="center", fontsize=9, color=INK)
        style(ax, title, "perplexity (lower is better)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlim(0, max(vals) * 1.15)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    p = out / "fig_quality.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_gap(metrics, out):
    q = metrics.get("quality", {})
    keys = [k for k in ordered(q) if q[k]["c4_gap_closed_pct"] is not None]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(keys) + 1.6))
    vals = [q[k]["c4_gap_closed_pct"] for k in keys]
    ax.barh(range(len(keys)), vals, color=[colour(k) for k in keys], height=0.55, edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(keys)), [label(k) for k in keys])
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=10, color=INK)
    style(ax, "Share of the student-teacher C4 loss gap closed by distillation",
          "% of gap closed vs the matched baseline (0% = no gain, 100% = teacher)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(min(0, min(vals) * 1.2), max(100 * 0.3, max(vals) * 1.3))
    ax.axvline(0, color=MUTED, linewidth=1)
    fig.tight_layout()
    p = out / "fig_gap_closed.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_curves(logs, out):
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=False)
    for ax, col, title in ((axes[0], "val_c4", "C4 validation perplexity"),
                           (axes[1], "val_ts", "TinyStories validation perplexity")):
        for k in ordered(logs):
            rows = logs[k]
            xs = [r["step"] for r in rows]
            ys = [math.exp(r[col]) for r in rows]
            ax.plot(xs, ys, color=colour(k), linewidth=2, label=f"{label(k)} (final {ys[-1]:.2f})",
                    linestyle="--" if k.startswith("baseline") else "-")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "2468" else ""))
        style(ax, title, "training step", "perplexity (log scale)")
        ax.grid(axis="both", color=GRID, linewidth=0.6, which="both")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = out / "fig_val_ppl_curves.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def section_efficiency(ev, md, metrics, out):
    have = [k for k in ordered(ev) if ev[k].get("params_m") and ev[k].get("peak_mem_mb")]
    teacher = ev.get("teacher_base")
    students = [k for k in have if k != "teacher_base"]
    if not teacher or not students or "latency_ms_1x256" not in teacher:
        md += ["## 2. Efficiency: student versus teacher", "",
               "Not available: needs evaluate.py JSONs (with size, memory and latency) for the teacher and at least one student.", ""]
        return None
    s = ev[students[0]]
    rat = {"params": teacher["params_m"] / s["params_m"], "peak_mem": teacher["peak_mem_mb"] / s["peak_mem_mb"],
           "latency": teacher["latency_ms_1x256"] / s["latency_ms_1x256"]}
    gpu = teacher.get("gpu", teacher.get("device", "?"))
    md += ["## 2. Efficiency: student versus teacher", "",
           f"One forward pass of 1 x 256 tokens in fp16 on {gpu} (median of 20). Every student has the same "
           "architecture, so size and memory are identical across students; latency differs only by measurement noise.", "",
           "| Model | Parameters | fp16 weights | Peak GPU memory | Latency (1 x 256) |", "|---|---|---|---|---|"]
    for k in have:
        e = ev[k]
        md.append(f"| {label(k)} | {e['params_m']:.2f}M | {e['weights_fp16_mb']:.1f} MB | {e['peak_mem_mb']:.1f} MB "
                  f"| {e['latency_ms_1x256']:.1f} ms |")
    lat = [ev[k]["latency_ms_1x256"] for k in students]
    md += ["", f"Teacher / student: **{rat['params']:.1f}x** parameters, **{rat['peak_mem']:.1f}x** peak memory, "
           f"**{rat['latency']:.1f}x** latency (student latency range {min(lat):.1f} to {max(lat):.1f} ms across "
           f"{len(lat)} students).", ""]
    metrics["efficiency"] = {"ratios_teacher_over_student": {k: round(v, 2) for k, v in rat.items()},
                             "student_latency_ms_range": [min(lat), max(lat)]}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    pairs = [("params_m", "Parameters (M)", "{:.0f}M"), ("peak_mem_mb", "Peak GPU memory (MB)", "{:.0f} MB"),
             ("latency_ms_1x256", "Latency, 1 x 256 tokens (ms)", "{:.1f} ms")]
    keys = [students[0], "teacher_base"]
    names = ["Student (51M)", "Teacher (360M)"]
    for ax, (f, t, fmt) in zip(axes, pairs):
        vals = [ev[students[0]][f] if f != "latency_ms_1x256" else sum(lat) / len(lat), teacher[f]]
        ax.bar([0, 1], vals, color=[colour("kd_short_T2"), colour("teacher_base")], width=0.55,
               edgecolor="white", linewidth=2)
        ax.set_xticks([0, 1], names)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        style(ax, t)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle(f"Teacher is {rat['params']:.1f}x larger, needs {rat['peak_mem']:.1f}x the memory and is "
                 f"{rat['latency']:.1f}x slower", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    p = out / "fig_efficiency.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def section_bench(bench_dirs, md, metrics, out):
    b = {}
    for d in bench_dirs:
        for p in sorted(Path(d).glob("*.json")):
            j = json.load(open(p))
            if "tasks" in j and j.get("name") not in b:
                b[j["name"]] = j
    if not b:
        md += ["## 3. Zero-shot benchmarks", "", "Not available: no bench_mc.py JSON found.", ""]
        return None
    tasks = sorted({t for j in b.values() for t in j["tasks"]})
    keys = ordered(b)
    md += ["## 3. Zero-shot benchmarks (higher is better)", "",
           "Log-likelihood multiple choice as in lm-evaluation-harness; acc_norm divides by answer length. "
           "+/- is one standard error. Random guessing gives about 25%.", "",
           "| Model | " + " | ".join(f"{t} acc | {t} acc_norm" for t in tasks) + " |",
           "|---|" + "---|---|" * len(tasks)]
    for k in keys:
        cells = []
        for t in tasks:
            r = b[k]["tasks"].get(t)
            cells += ([f"{r['acc']*100:.1f} +/- {r['acc_stderr']*100:.1f}",
                       f"{r['acc_norm']*100:.1f} +/- {r['acc_norm_stderr']*100:.1f}"] if r else ["-", "-"])
        md.append(f"| {label(k)} | " + " | ".join(cells) + " |")
    n_info = {t: sorted({j["tasks"][t]["n"] for j in b.values() if t in j["tasks"]}) for t in tasks}
    md += ["", "Examples per task: " + ", ".join(f"{t} {'/'.join(map(str, n))}" for t, n in n_info.items())
           + (" (WARNING: models were scored on different numbers of examples)" if any(len(n) > 1 for n in n_info.values())
              else "") + ".", ""]
    metrics["benchmarks"] = {k: b[k]["tasks"] for k in keys}

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 0.5 * len(keys) + 1.8))
    axes = [axes] if len(tasks) == 1 else axes
    for ax, t in zip(axes, tasks):
        ks = [k for k in keys if t in b[k]["tasks"]]
        vals = [b[k]["tasks"][t]["acc_norm"] * 100 for k in ks]
        errs = [b[k]["tasks"][t]["acc_norm_stderr"] * 100 for k in ks]
        ax.barh(range(len(ks)), vals, xerr=errs, color=[colour(k) for k in ks], height=0.6,
                edgecolor="white", linewidth=2, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        ax.set_yticks(range(len(ks)), [label(k) for k in ks] if ax is axes[0] else [""] * len(ks))
        ax.invert_yaxis()
        chance = b[ks[0]]["tasks"][t]["chance"] * 100
        ax.axvline(chance, color=MUTED, linestyle=":", linewidth=1.2)
        ax.text(chance, -0.7, f" chance {chance:.0f}%", fontsize=8, color=MUTED, va="bottom")
        for i, v in enumerate(vals):
            ax.text(v + errs[i], i, f"  {v:.1f}%", va="center", fontsize=9, color=INK)
        style(ax, f"{t} (acc_norm)", "accuracy (%)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    p = out / "fig_benchmarks.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def section_sft(sft_json, md, metrics):
    if not sft_json or not Path(sft_json).exists():
        md += ["## 4. Instruction fine-tuning (Dolly 15k)", "", "Not available: sft_comparison.json not given.", ""]
        return
    j = json.load(open(sft_json))
    md += ["## 4. Instruction fine-tuning (Dolly 15k validation, 592 examples)", "",
           "Response-only cross-entropy of each model's selected checkpoint (lowest validation loss). "
           "Same data split, prompt template and hyperparameters for every run. Not comparable with C4/TinyStories losses.", "",
           "| Run | Updates | Best update | Best val loss | Final val loss |", "|---|---|---|---|---|"]
    for r in j.get("sft_runs", []):
        md.append(f"| `{r['run']}` | {r['updates_done']} | {r['best_update']} | {r['best_val']} | {r['final_val']} |")
    md.append("")
    metrics["sft"] = j.get("sft_runs", [])


def words(t):
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def f1(a, b):
    ov = sum((Counter(a) & Counter(b)).values())
    if not a or not b or ov == 0:
        return 0.0
    p, r = ov / len(a), ov / len(b)
    return 2 * p * r / (p + r)


def section_answers(gens_path, md, metrics):
    if not gens_path or not Path(gens_path).exists():
        return
    gens = [json.loads(l) for l in open(gens_path, encoding="utf-8") if l.strip()]
    models = list(gens[0]["answers"])
    md += [f"## 5. Answer statistics on {len(gens)} held-out Dolly prompts", "",
           "Same prompts, decoding settings and random seed per prompt for every model. Word F1 is unigram overlap "
           "with the human reference answer (closer to the reference, not necessarily better).", "",
           "| Model | Mean words | Empty | Repeated 3-gram | Distinct bigrams | Word F1 vs reference |", "|---|---|---|---|---|---|"]
    metrics["answers"] = {}
    for m in models:
        ws = [words(g["answers"].get(m, "")) for g in gens]
        rep = sum(len({tuple(w[i:i + 3]) for i in range(len(w) - 2)}) != max(0, len(w) - 2) for w in ws) / len(ws)
        bi = [tuple(w[i:i + 2]) for w in ws for i in range(len(w) - 1)]
        dist = len(set(bi)) / len(bi) if bi else 0.0
        empty = sum(not w for w in ws) / len(ws)
        fs = sum(f1(w, words(g["human_reference"])) for w, g in zip(ws, gens)) / len(gens)
        mean_w = sum(map(len, ws)) / len(ws)
        md.append(f"| {label(m)} | {mean_w:.1f} | {empty*100:.1f}% | {rep*100:.1f}% | {dist*100:.1f}% | {fs:.3f} |")
        metrics["answers"][m] = {"mean_words": mean_w, "empty": empty, "repeated_3gram": rep,
                                 "distinct_bigrams": dist, "word_f1_vs_reference": fs}
    md.append("")


def section_judge(judge_files, md, metrics, out):
    rows = []
    for f in judge_files or []:
        if not Path(f).exists():
            continue
        j = json.load(open(f))
        for pair, s in j["pairs"].items():
            x, y = pair.split(":")
            dec = s[x] + s[y]
            lead = x if s[x] >= s[y] else y
            ci = wilson(s[lead], dec)
            rows.append({"file": Path(f).name, "judge": j.get("judge"), "forced": j.get("forced_choice"), "x": x, "y": y,
                         "n": s["prompts_judged"], "wx": s[x], "wy": s[y], "tie": s["tie"], "lead": lead,
                         "rate": s[lead] / dec if dec else None, "ci": ci, "p": s["sign_test_p"],
                         "flips": s["tie_breakdown"]["order_flips"]})
    if not rows:
        md += ["## 6. Blind pairwise preference", "", "Not available: no judge summary found.", ""]
        return None
    md += ["## 6. Blind pairwise preference (LLM judge, both answer orders)", "",
           "A model wins a prompt only if the judge picks it in both orders; otherwise the prompt is a tie. "
           "Interval: 95% Wilson interval for the leader's share of decisive prompts. p: exact two-sided sign test.", "",
           "| Pair | Protocol | Judge | Prompts | First wins | Second wins | Ties (of which order flips) "
           "| Leader's win rate (excl. ties) | 95% interval | Sign-test p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f"{r['ci'][0]*100:.0f}% to {r['ci'][1]*100:.0f}%" if r["ci"] else "-"
        rate = f"{r['rate']*100:.1f}% ({label(r['lead'])})" if r["rate"] is not None else "-"
        md.append(f"| {label(r['x'])} vs {label(r['y'])} | {'forced choice' if r['forced'] else 'tie allowed'} "
                  f"| {r['judge']} | {r['n']} | {r['wx']} | {r['wy']} | {r['tie']} ({r['flips']}) | {rate} | {ci} "
                  f"| {fmt_p(r['p'])} |")
    md.append("")
    metrics["pairwise"] = [{k: v for k, v in r.items()} for r in rows]

    fig, ax = plt.subplots(figsize=(10, 0.75 * len(rows) + 1.6))
    for i, r in enumerate(rows):
        n = r["n"] or 1
        segs = [(r["wx"] / n, colour(r["x"]), f"{label(r['x'])} {r['wx']}"),
                (r["tie"] / n, TIE_GREY, f"tie {r['tie']}"),
                (r["wy"] / n, colour(r["y"]), f"{label(r['y'])} {r['wy']}")]
        left = 0
        for w, c, t in segs:
            ax.barh(i, w * 100, left=left * 100, color=c, height=0.55, edgecolor="white", linewidth=2)
            if w > 0.08:
                ax.text((left + w / 2) * 100, i, t, ha="center", va="center", fontsize=8,
                        color="white" if c != TIE_GREY else INK)
            left += w
        ax.text(101, i, f"p = {fmt_p(r['p'])}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(rows)), [f"{label(r['x'])} vs {label(r['y'])}\n"
                                     f"({'forced' if r['forced'] else 'tie allowed'}, n={r['n']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    style(ax, "Blind pairwise preference on held-out Dolly prompts", "% of prompts")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = out / "fig_pairwise.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", nargs="*", default=[])
    ap.add_argument("--log_dir", nargs="*", default=[])
    ap.add_argument("--bench_dir", nargs="*", default=[])
    ap.add_argument("--judge", nargs="*", default=[])
    ap.add_argument("--gens", default="")
    ap.add_argument("--sft_json", default="")
    ap.add_argument("--out", default="results/final")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = find_logs(a.log_dir)
    ev = load_eval(a.eval_dir, logs)
    md = ["# TENSORIX final evaluation", "",
          "Generated by `src/final_report.py`. Do not edit by hand: rerun the script when new result files arrive.", ""]
    metrics, figs = {}, []
    section_quality(ev, md, metrics)
    figs += [fig_quality(ev, out), fig_gap(metrics, out), fig_curves(logs, out)]
    figs.append(section_efficiency(ev, md, metrics, out))
    figs.append(section_bench(a.bench_dir, md, metrics, out))
    section_sft(a.sft_json, md, metrics)
    section_answers(a.gens, md, metrics)
    figs.append(section_judge(a.judge, md, metrics, out))
    figs = [f for f in figs if f]
    md += ["## Figures", ""] + [f"![{f.stem}]({f.name})" for f in figs] + [""]
    md += ["## Limitations", "",
           "* One training seed per configuration: differences of a few hundredths in loss are within run-to-run noise.",
           "* SHORT-tier runs (5,000 steps) are compared only with `baseline_short`, FULL-tier runs only with `baseline_full`.",
           "* Benchmarks at this model size sit close to chance; treat small differences with the standard errors in mind.",
           "* The LLM judge is a single model; its position bias shows up as order flips (counted as ties).", ""]
    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    json.dump(metrics, open(out / "final_metrics.json", "w"), indent=2, default=str)
    print("\n".join(md))
    print(f"\nwrote {out / 'RESULTS.md'}, {out / 'final_metrics.json'} and {len(figs)} figures")


if __name__ == "__main__":
    main()
