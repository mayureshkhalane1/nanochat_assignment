"""
Self-contained, weighted BPE trainer (get_stats / merge loop) used to produce
the merge-tree figure for the Task 1 report. Trained on a *hand-crafted*,
weighted micro-corpus (allowed by the assignment: "you can look for one in the
data, or make your own"). Counts printed are the true pair frequencies at the
moment of each merge.

The shipped Task 1 tokenizers are trained by rustbpe (scripts/tok_task1.py);
this script only illustrates the mechanics of a merge tree with real counts.
"""
import sys
import collections

CORPUS = {
    "unhappiness": 6,
    "happiness": 5,
    "unhappy": 4,
    "happy": 3,
    "appendix": 3,
    "happening": 2,
    "unplanned": 2,
}

# each word is a list of tokens (initially single characters)
corpus = [[list(w), c] for w, c in CORPUS.items()]


def get_stats(corpus):
    stats = collections.Counter()
    for word, c in corpus:
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += c
    return stats


def merge_all(corpus, pair, repl):
    out = []
    for word, c in corpus:
        merged = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                merged.append(repl)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        out.append([merged, c])
    return out


merges_by = {}  # merged token -> pair count at merge time
merge_order = []
print("merge sequence (global, on the weighted micro-corpus):")
i = 0
while True:
    stats = get_stats(corpus)
    if not stats:
        break
    pair = max(stats, key=lambda p: stats[p])
    count = stats[pair]
    if count < 2:
        break
    a, b = pair
    repl = a + b
    merges_by[repl] = count
    merge_order.append(repl)
    i += 1
    print(f"  {i:2d}. {a!r} + {b!r} -> {repl!r}   (pair count {count})")
    corpus = merge_all(corpus, pair, repl)

word = sys.argv[1] if len(sys.argv) > 1 else "unhappiness"
seq = list(word)
steps = []
for token in merge_order:
    for i in range(len(seq) - 1):
        if seq[i] + seq[i + 1] == token:
            steps.append((seq[i], seq[i + 1], merges_by[token]))
            seq[i:i + 2] = [token]
            break
print(f"\nmerge tree for {word!r}:")
for a, b, cnt in steps:
    print(f"  {a!r} + {b!r} = {a+b!r}   (count {cnt})")
print(f"final tokens: {seq}")