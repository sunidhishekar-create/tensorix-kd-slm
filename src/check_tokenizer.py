"""Verify SmolLM2 Base and Instruct share one vocab, and that ids fit in uint16."""
from transformers import AutoTokenizer, AutoConfig

BASE = "HuggingFaceTB/SmolLM2-360M"
INST = "HuggingFaceTB/SmolLM2-360M-Instruct"

tb, ti = AutoTokenizer.from_pretrained(BASE), AutoTokenizer.from_pretrained(INST)
cb, ci = AutoConfig.from_pretrained(BASE), AutoConfig.from_pretrained(INST)

print("len(tokenizer)   :", len(tb), len(ti))
print("config vocab_size:", cb.vocab_size, ci.vocab_size)

vb, vi = tb.get_vocab(), ti.get_vocab()
print("vocab identical  :", vb == vi)
if vb != vi:
    diff = {t: (vb.get(t), vi.get(t)) for t in set(vb) | set(vi) if vb.get(t) != vi.get(t)}
    print("  differing tokens:", dict(list(diff.items())[:20]))

print("eos :", tb.eos_token, tb.eos_token_id, "| bos:", tb.bos_token, tb.bos_token_id)
print("pad :", tb.pad_token, tb.pad_token_id)

s = "Once upon a time, a little robot learned to count to ten."
ids = tb(s)["input_ids"]
print("encoded:", ids)
print("roundtrip ok:", tb.decode(ids) == s)
print("max id < 65536 (uint16 safe):", max(vb.values()) < 65536)