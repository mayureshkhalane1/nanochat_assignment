"""
Task 1 deliverables: train two BPE tokenizers on a 500MB sample of CLIMBMix
(vocab 8192 and 32768), and emit compression / sequence-length / merge-tree /
failure-case artifacts needed for the report.

Unlike tok_train.py this saves each tokenizer to its own directory (vocab8k /
vocab32k) so the re-trained 32768 tokenizer does not clobber the one used by
the pretraining pipeline.
"""
import os
import sys
import json
import time
import glob
import argparse
import tiktoken
import numpy as np

from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from nanochat.common import get_base_dir

parser = argparse.ArgumentParser(description="Task 1 tokenizer experiments")
parser.add_argument("--max-chars", type=int, default=500_000_000, help="500 MB sample")
parser.add_argument("--vocab-size", type=int, default=8192, help="vocab size to train")
parser.add_argument("--doc-cap", type=int, default=10_000, help="max chars per document")
parser.add_argument("--outdir", type=str, default=None, help="output directory")
args = parser.parse_args()

base_dir = get_base_dir()
outdir = args.outdir or os.path.join(base_dir, "task1", f"vocab{args.vocab_size}")
os.makedirs(outdir, exist_ok=True)


def text_iterator(max_chars, doc_cap):
    nchars = 0
    for batch in parquets_iter_batched("train"):
        for doc in batch:
            if len(doc) > doc_cap:
                doc = doc[:doc_cap]
            nchars += len(doc)
            yield doc
            if nchars > max_chars:
                return


TRAIN_SAMPLE = 3_000_000


def count_sample_chars(max_chars, doc_cap):
    n = 0
    for batch in parquets_iter_batched("train"):
        for doc in batch:
            if len(doc) > doc_cap:
                doc = doc[:doc_cap]
            n += len(doc)
            if n > max_chars:
                return n
    return n


t0 = time.time()
train_chars = count_sample_chars(args.max_chars, args.doc_cap)
print(f"[{args.vocab_size}] training sample: {train_chars:,} chars")
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iterator(args.max_chars, args.doc_cap), args.vocab_size
)
t1 = time.time()
print(f"[{args.vocab_size}] trained in {t1 - t0:.1f}s, final vocab={tokenizer.get_vocab_size()}")
tokenizer.enc = tokenizer.enc  # keep
# save via the same mechanism as tok_train (pickle the tiktoken encoding)
pickle_path = os.path.join(outdir, "tokenizer.pkl")
with open(pickle_path, "wb") as f:
    import pickle
    pickle.dump(tokenizer.enc, f)
vocab_size = tokenizer.get_vocab_size()

import torch
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)
    else:
        token_bytes.append(len(tokenizer.enc.decode_single_token_bytes(token_id)))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
torch.save(token_bytes, os.path.join(outdir, "token_bytes.pt"))

# --------------------------------------------------------------------------
# Measurements on a held-out English sample (fixed text, both tokenizers use
# the same bytes so compression / sequence-length numbers are comparable).
# --------------------------------------------------------------------------
SAMPLE_MB = 3  # ~3 MB of English text from the val split
val_texts = []
nchars = 0
for batch in parquets_iter_batched("val"):
    for doc in batch:
        if len(doc) > args.doc_cap:
            doc = doc[:args.doc_cap]
        nchars += len(doc)
        val_texts.append(doc)
        if nchars > SAMPLE_MB * 1_000_000:
            break
val_text = "\n".join(val_texts)
raw_bytes = len(val_text.encode("utf-8"))
toks = tokenizer.encode(val_text)
n_tokens = len(toks)
seq_lengths = []
chunk_stride = 512
for i in range(0, len(val_text), chunk_stride):
    chunk = val_text[i : i + chunk_stride]
    if chunk:
        seq_lengths.append(len(tokenizer.encode(chunk)))
seq_arr = np.array(seq_lengths)

print(f"[{args.vocab_size}] raw_bytes={raw_bytes:,} tokens={n_tokens:,} chars_per_token={len(val_text)/max(n_tokens,1):.3f}")
print(f"[{args.vocab_size}] tokens_per_char={n_tokens/len(val_text):.5f}")
print(f"[{args.vocab_size}] seq_len(512-char chunk): mean={seq_arr.mean():.2f} std={seq_arr.std():.2f} max={seq_arr.max()}")

# --------------------------------------------------------------------------
# Merge-tree example (own example, not reused from course material)
# --------------------------------------------------------------------------
for word in ["overgeneralizations", "unhappiness"]:
    print(f"[{args.vocab_size}] merge tree for '{word}':")
    for tid in tokenizer.encode(word):
        raw = tokenizer.enc.decode_single_token_bytes(tid)
        print(f"    id={tid:<5} rank={tid:<5} bytes={raw!r}")

# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------
cases = {
    "numbers": "The GDP grew 5.02% in 2023, costs rose $1,234,567.89 and pi=3.14159265.",
    "code": "def f(x):\n    return [x*i for i in range(100)]  # list comprehension",
    "nonenglish": "Dies ist ein deutscher Satz mit Umlauten: Ünïcödé. これは日本語のテキストです。",
}
for name, text in cases.items():
    toks = tokenizer.encode(text)
    print(f"[{args.vocab_size}] failure case '{name}': {len(text)} chars -> {len(toks)} tokens "
          f"({len(toks)/max(len(text),1):.4f} tok/char)")

# --------------------------------------------------------------------------
# JSON summary
# --------------------------------------------------------------------------
summary = {
    "vocab_size": vocab_size,
    "train_chars": train_chars,
    "train_seconds": t1 - t0,
    "val_text_bytes": raw_bytes,
    "val_chars": len(val_text),
    "val_tokens": n_tokens,
    "tokens_per_char": n_tokens / len(val_text),
    "chars_per_token": len(val_text) / n_tokens,
    "seq_len_mean": float(seq_arr.mean()),
    "seq_len_std": float(seq_arr.std()),
    "seq_len_max": int(seq_arr.max()),
}
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[{args.vocab_size}] summary -> {outdir}/summary.json")