import json, matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("/Users/mayureshkhalane/.cache/nanochat/base_checkpoints/d2/run.jsonl")]
val  = [r for r in rows if r["type"] == "val" and r["train_loss"] and r["step"] > 0]
K    = sum(r["val_bpb"] / r["train_loss"] for r in val) / len(val)   # loss->bpb constant (~0.297)
train = [r for r in rows if r["type"] == "train" and r["step"] >= 10]

plt.plot([r["step"] for r in train], [r["train_loss"] * K for r in train], label="train bpb (approx)")
plt.plot([r["step"] for r in val], [r["val_bpb"] for r in val], "o-", label="val bpb")
plt.xlabel("step"); plt.ylabel("bits per byte (bpb)"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("task2_bpb_curves.png", dpi=150)
