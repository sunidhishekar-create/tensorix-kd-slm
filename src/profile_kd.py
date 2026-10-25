"""Accurate CUDA profiling script for Knowledge Distillation training steps.

Measures fine-grained GPU timing for:
  (a) Student forward
  (b) Teacher get_soft_targets (split into GPT-2 forward, float cast / T,
      softmax, kept gather, dropped mass, and concatenation)
  (c) Distillation loss forward (row-chunked checkpointing)
  (d) Backward pass (including checkpoint recomputations)
  (e) Optimizer step
Also benchmarks a baseline training step (no teacher, standard cross-entropy).
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from model import SLM, ModelConfig
from distill import Distiller, VocabMapper, TeacherModel, distillation_loss


class Timer:
    """CUDA event timer with CPU fallback for accurate asynchronous timing."""
    def __init__(self, is_cuda=True):
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
        self.cpu_start = 0.0
        self.cpu_end = 0.0

    def start(self):
        if self.is_cuda:
            self.start_evt.record()
        else:
            self.cpu_start = time.perf_counter()

    def stop(self):
        if self.is_cuda:
            self.end_evt.record()
        else:
            self.cpu_end = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.is_cuda:
            self.end_evt.synchronize()
            return self.start_evt.elapsed_time(self.end_evt)
        return (self.cpu_end - self.cpu_start) * 1000.0


def instrumented_get_soft_targets(
    teacher: TeacherModel,
    student_ids: torch.Tensor,
    temperature: float,
    chunk_size: int,
    timers: dict,
    is_cuda: bool,
):
    """
    Executes teacher get_soft_targets while recording precise GPU sub-stage timings:
      1. to_teacher_tokens
      2. GPT-2 forward pass (autocast fp16)
      3. Logits float cast & temperature division
      4. Softmax over 50,257 classes
      5. Kept-column gather (index_select)
      6. Dropped-column pooling (sum & clamp)
      7. Assembly concatenation
    """
    B = student_ids.size(0)
    device_type = "cuda" if student_ids.is_cuda else "cpu"
    use_amp = student_ids.is_cuda

    # If chunk_size is provided, process in sub-batches along dim 0
    sub_batch_slices = [
        (i, min(i + chunk_size, B)) for i in range(0, B, chunk_size)
    ] if (chunk_size and chunk_size > 0 and B > chunk_size) else [(0, B)]

    out_chunks = []
    for start_idx, end_idx in sub_batch_slices:
        chunk_ids = student_ids[start_idx:end_idx]

        # Stage 1: Vocabulary token translation
        t_trans = Timer(is_cuda); t_trans.start()
        teacher_ids = teacher.vocab_mapper.to_teacher_tokens(chunk_ids)
        t_trans.stop()
        timers["b1_token_translation"].append(t_trans)

        # Stage 2: GPT-2 forward pass
        t_fwd = Timer(is_cuda); t_fwd.start()
        with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
            outputs = teacher.model(input_ids=teacher_ids)
            t_logits = outputs.logits
        t_fwd.stop()
        timers["b2_gpt2_forward"].append(t_fwd)

        # Stage 3: Float32 conversion and temperature division
        t_scale = Timer(is_cuda); t_scale.start()
        t_logits_f32 = t_logits.float() / temperature
        t_scale.stop()
        timers["b3_logits_float_and_temp"].append(t_scale)

        # Stage 4: Softmax over 50,257 classes
        t_sm = Timer(is_cuda); t_sm.start()
        t_probs = F.softmax(t_logits_f32, dim=-1)
        t_sm.stop()
        timers["b4_softmax_50k"].append(t_sm)

        # Stage 5: Kept-column gather (index_select)
        t_gath = Timer(is_cuda); t_gath.start()
        kept_probs = t_probs.index_select(dim=-1, index=teacher.vocab_mapper.gpt2_ids_kept)
        t_gath.stop()
        timers["b5_kept_gather"].append(t_gath)

        # Stage 6: Dropped-column pooling (logsumexp equivalent: 1 - sum(kept))
        t_drop = Timer(is_cuda); t_drop.start()
        unk_prob = (1.0 - kept_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        t_drop.stop()
        timers["b6_dropped_pooling"].append(t_drop)

        # Stage 7: Concat & student vocab assembly
        t_cat = Timer(is_cuda); t_cat.start()
        student_probs = torch.cat([kept_probs, unk_prob], dim=-1)
        t_cat.stop()
        timers["b7_concat_assembly"].append(t_cat)

        out_chunks.append(student_probs)

    if len(out_chunks) == 1:
        return out_chunks[0]
    return torch.cat(out_chunks, dim=0)


def get_args():
    parser = argparse.ArgumentParser(description="Profile Knowledge Distillation step latency breakdown")
    parser.add_argument("--data_dir", required=True, help="path to train.bin, vocab.json")
    parser.add_argument("--teacher", default="gpt2", help="teacher model name")
    parser.add_argument("--teacher_chunk", type=int, default=4, help="teacher batch chunk size")
    parser.add_argument("--loss_chunk", type=int, default=2048, help="loss row chunk size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--timed_steps", type=int, default=10)
    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device == "cuda"
    use_amp = is_cuda
    print(f"Profiling on device: {device} (CUDA available: {torch.cuda.is_available()})")
    if is_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = json.load(open(vocab_path))
    data = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")

    def get_batch():
        ix = np.random.randint(0, len(data) - args.block_size - 1, args.batch_size)
        x = np.stack([data[i:i + args.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # 2. Instantiate Student Model & Optimizer
    cfg = ModelConfig(vocab_size=vocab["vocab_size"], block_size=args.block_size)
    student = SLM(cfg).to(device)
    decay = [p for p in student.parameters() if p.dim() >= 2]
    no_decay = [p for p in student.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 3. Instantiate Teacher / Distiller
    print(f"Loading teacher '{args.teacher}'...")
    distiller = Distiller(
        vocab_data_or_path=vocab_path,
        teacher_model=args.teacher,
        temperature=args.kd_temp,
        alpha=args.kd_alpha,
        teacher_chunk=args.teacher_chunk,
        loss_chunk=args.loss_chunk,
        device=device,
    )

    # =========================================================================
    # Phase A: Benchmark Baseline Training Step (No Teacher)
    # =========================================================================
    print(f"\n--- Benchmarking Baseline Step ({args.warmup_steps} warmup, {args.timed_steps} timed) ---")
    baseline_times = []
    student.train()

    total_baseline_steps = args.warmup_steps + args.timed_steps
    for step_i in range(total_baseline_steps):
        x, y = get_batch()
        if is_cuda:
            torch.cuda.synchronize()

        t_base = Timer(is_cuda); t_base.start()
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            _, loss = student(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if is_cuda:
            torch.cuda.synchronize()
        t_base.stop()

        if step_i >= args.warmup_steps:
            baseline_times.append(t_base.elapsed_ms())

    mean_baseline = np.mean(baseline_times)
    std_baseline = np.std(baseline_times)
    print(f"Baseline Step: {mean_baseline:.2f} ms +/- {std_baseline:.2f} ms")

    # =========================================================================
    # Phase B: Benchmark Full Knowledge Distillation Step with Granular Timing
    # =========================================================================
    print(f"\n--- Benchmarking KD Step (teacher_chunk={args.teacher_chunk}, loss_chunk={args.loss_chunk}) ---")
    kd_stage_times = {
        "student_forward": [],
        "b1_token_translation": [],
        "b2_gpt2_forward": [],
        "b3_logits_float_and_temp": [],
        "b4_softmax_50k": [],
        "b5_kept_gather": [],
        "b6_dropped_pooling": [],
        "b7_concat_assembly": [],
        "teacher_total": [],
        "loss_forward": [],
        "backward_pass": [],
        "optimizer_step": [],
        "kd_total_step": [],
    }

    total_kd_steps = args.warmup_steps + args.timed_steps
    for step_i in range(total_kd_steps):
        is_timed = (step_i >= args.warmup_steps)
        x, y = get_batch()
        if is_cuda:
            torch.cuda.synchronize()

        step_timer = Timer(is_cuda); step_timer.start()

        # (a) Student forward
        t_stud = Timer(is_cuda); t_stud.start()
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            student_logits, _ = student(x, y)
        t_stud.stop()

        # (b) Teacher get_soft_targets with internal split
        t_teach = Timer(is_cuda); t_teach.start()
        # Per-step dictionary for sub-stage accumulation across chunks
        sub_timers = {k: [] for k in ["b1_token_translation", "b2_gpt2_forward",
                                     "b3_logits_float_and_temp", "b4_softmax_50k",
                                     "b5_kept_gather", "b6_dropped_pooling", "b7_concat_assembly"]}
        teacher_probs = instrumented_get_soft_targets(
            distiller.teacher,
            x,
            temperature=args.kd_temp,
            chunk_size=args.teacher_chunk,
            timers=sub_timers,
            is_cuda=is_cuda,
        )
        t_teach.stop()

        # (c) Distillation loss forward
        t_loss = Timer(is_cuda); t_loss.start()
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            loss, ce_val, kl_val = distillation_loss(
                student_logits=student_logits,
                teacher_probs=teacher_probs,
                targets=y,
                temperature=args.kd_temp,
                alpha=args.kd_alpha,
                loss_chunk=args.loss_chunk,
            )
        t_loss.stop()

        # (d) Backward pass
        t_bwd = Timer(is_cuda); t_bwd.start()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        t_bwd.stop()

        # (e) Optimizer step
        t_opt = Timer(is_cuda); t_opt.start()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        t_opt.stop()

        if is_cuda:
            torch.cuda.synchronize()
        step_timer.stop()

        if is_timed:
            kd_stage_times["student_forward"].append(t_stud.elapsed_ms())
            for k, timer_list in sub_timers.items():
                kd_stage_times[k].append(sum(t.elapsed_ms() for t in timer_list))
            kd_stage_times["teacher_total"].append(t_teach.elapsed_ms())
            kd_stage_times["loss_forward"].append(t_loss.elapsed_ms())
            kd_stage_times["backward_pass"].append(t_bwd.elapsed_ms())
            kd_stage_times["optimizer_step"].append(t_opt.elapsed_ms())
            kd_stage_times["kd_total_step"].append(step_timer.elapsed_ms())

    # =========================================================================
    # Phase C: Formatting and Printing Measured Breakdown Table
    # =========================================================================
    mean_kd_total = np.mean(kd_stage_times["kd_total_step"])
    std_kd_total = np.std(kd_stage_times["kd_total_step"])

    rows = [
        ("(a) Student Forward Pass", np.mean(kd_stage_times["student_forward"]), np.std(kd_stage_times["student_forward"])),
        ("(b) Teacher get_soft_targets (Total)", np.mean(kd_stage_times["teacher_total"]), np.std(kd_stage_times["teacher_total"])),
        ("    |-- GPT-2 Forward Pass (fp16)", np.mean(kd_stage_times["b2_gpt2_forward"]), np.std(kd_stage_times["b2_gpt2_forward"])),
        ("    |-- Logits float32 cast & / T", np.mean(kd_stage_times["b3_logits_float_and_temp"]), np.std(kd_stage_times["b3_logits_float_and_temp"])),
        ("    |-- Softmax over 50,257 classes", np.mean(kd_stage_times["b4_softmax_50k"]), np.std(kd_stage_times["b4_softmax_50k"])),
        ("    |-- Kept-column gather (index_select)", np.mean(kd_stage_times["b5_kept_gather"]), np.std(kd_stage_times["b5_kept_gather"])),
        ("    |-- Dropped-column pooling (logsumexp equiv)", np.mean(kd_stage_times["b6_dropped_pooling"]), np.std(kd_stage_times["b6_dropped_pooling"])),
        ("    |-- Concat & student vocab assembly", np.mean(kd_stage_times["b7_concat_assembly"]), np.std(kd_stage_times["b7_concat_assembly"])),
        ("    \\-- Token Translation (s2t)", np.mean(kd_stage_times["b1_token_translation"]), np.std(kd_stage_times["b1_token_translation"])),
        ("(c) Distillation Loss Forward", np.mean(kd_stage_times["loss_forward"]), np.std(kd_stage_times["loss_forward"])),
        ("(d) Backward Pass (inc. checkpointing)", np.mean(kd_stage_times["backward_pass"]), np.std(kd_stage_times["backward_pass"])),
        ("(e) Optimizer Step & Grad Clip", np.mean(kd_stage_times["optimizer_step"]), np.std(kd_stage_times["optimizer_step"])),
    ]

    print("\n" + "=" * 82)
    print("           KNOWLEDGE DISTILLATION LATENCY BREAKDOWN (MEASURED ON GPU)")
    print("=" * 82)
    print(f"{'Component / Operation':<44} | {'Mean (ms)':>10} | {'Std (ms)':>9} | {'% of Total':>10}")
    print("-" * 82)
    for name, mean_v, std_v in rows:
        pct = (mean_v / mean_kd_total) * 100.0
        print(f"{name:<44} | {mean_v:>10.2f} | {std_v:>9.2f} | {pct:>9.1f}%")
    print("-" * 82)
    print(f"{'TOTAL KD STEP TIME':<44} | {mean_kd_total:>10.2f} | {std_kd_total:>9.2f} | {'100.0%':>10}")
    print(f"{'BASELINE STEP TIME (No Teacher)':<44} | {mean_baseline:>10.2f} | {std_baseline:>9.2f} | {mean_baseline/mean_kd_total*100.0:>9.1f}%")
    print("=" * 82)
    print(f"Summary: KD step is {mean_kd_total / mean_baseline:.2f}x slower than baseline ({mean_kd_total:.1f} ms vs {mean_baseline:.1f} ms).")
    if is_cuda:
        print(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
