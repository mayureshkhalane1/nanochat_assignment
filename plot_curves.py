import argparse
import json

import matplotlib.pyplot as plt

BASE = "/Users/mayureshkhalane/.cache/nanochat/base_checkpoints"


def load(depth):
    rows = [json.loads(l) for l in open(f"{BASE}/d{depth}/run.jsonl")]
    val = [r for r in rows if r["type"] == "val" and r.get("train_loss") and r["step"] > 0]
    K = sum(r["val_bpb"] / r["train_loss"] for r in val) / len(val)
    train = [r for r in rows if r["type"] == "train" and r["step"] >= 10]
    return train, val, K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=int, nargs="+", default=[2], help="depths to plot")
    p.add_argument("--out", default=None, help="output png path")
    p.add_argument("--compare", action="store_true", help="overlay all depths on one axes")
    args = p.parse_args()

    data = {d: load(d) for d in args.depths}

    if args.compare:
        fig, ax = plt.subplots(figsize=(6, 4))
        for d, (train, val, K) in data.items():
            ax.plot([r["step"] for r in train], [r["train_loss"] * K for r in train],
                    lw=1, alpha=0.6, label=f"d{d} train bpb")
            ax.plot([r["step"] for r in val], [r["val_bpb"] for r in val], "o-",
                    lw=1.5, ms=3, label=f"d{d} val bpb (final {val[-1]['val_bpb']:.3f})")
        ax.set_xlabel("step")
        ax.set_ylabel("bits per byte (bpb)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.out or "task2_bpb_curves_compare.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
        return

    for d in args.depths:
        train, val, K = data[d]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["step"] for r in train], [r["train_loss"] * K for r in train], label="train bpb (approx)")
        ax.plot([r["step"] for r in val], [r["val_bpb"] for r in val], "o-", label="val bpb")
        ax.set_xlabel("step")
        ax.set_ylabel("bits per byte (bpb)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.out or f"task2_bpb_curves_d{d}.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()