"""Print the first few generated answers next to the human answers, to inspect quality and style.

  python experiments/gemma_sft/show_answers.py          # first 5
  python experiments/gemma_sft/show_answers.py 10       # first 10
"""
import sys
from common import read_jsonl

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
prompts = {r["id"]: r for r in read_jsonl("data/sft/prompts.jsonl")}
answers = read_jsonl("data/sft/gemma_answers.jsonl")
words = [len(a["answer"].split()) for a in answers]
print(f"{len(answers)} answers saved. Finish reasons: "
      f"{sorted({str(a['finish_reason']) for a in answers})}")
if words:
    print(f"Gemma answer length in words: min {min(words)}, max {max(words)}, "
          f"mean {sum(words) / len(words):.0f}\n")
for a in answers[:n]:
    p = prompts[a["id"]]
    print("INSTRUCTION:", p["instruction"])
    if p["context"].strip():
        print("CONTEXT    :", p["context"][:200].replace("\n", " "))
    print("HUMAN      :", p["human_response"][:300].replace("\n", " "))
    print("GEMMA      :", a["answer"].replace("\n", " "))
    print("-" * 70)
