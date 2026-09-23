# Assignment: commands, run logs, and results

Commands run for each task. Effective hyper-parameters match the `user_config`
written into each checkpoint's `meta_*.json` (see [`checkpoints/`](checkpoints/)).
Run logs are in [`runs/`](runs/).

## Task 1 - Tokenization (`vocab` 8,192 and 32,768)

```bash
.venv/bin/python -m scripts.tok_task1 --vocab-size 8192   --max-chars 500000000
.venv/bin/python -m scripts.tok_task1 --vocab-size 32768 --max-chars 500000000
.venv/bin/python scripts/tok_mergetree.py unhappiness   # merge-tree illustration with counts
```

`scripts/tok_task1.py` trains a rustbpe tokenizer on a 500 MB sample of the
CLIMBMix train split, saves it under `~/.cache/nanochat/task1/vocab{size}/` (so the
shared pretraining tokenizer is never overwritten), and emits compression /
sequence-length / failure-case measurements in `summary.json`.

### Teammate's Task 1 additions

Additional Task 1 scripts, committed in `b09fc2e`/`b45aa8b`/`e5b06c4`/`33503a4`.
`runs/tokenizer.sh` drives the whole flow: train both vocab sizes with the
upstream `scripts.tok_train`, snapshot each tokenizer to
`~/.cache/nanochat/tokenizer_{8192,32768}`, then run the evals below:

```bash
python -m scripts.tok_train --vocab-size 8192        # then 32768
cp -r ~/.cache/nanochat/tokenizer ~/.cache/nanochat/tokenizer_8192
python -m scripts.tok_eval                           # compression comparison
python -m scripts.tok_artifacts --text "1,586,291 people live in country 224393."
python -m scripts.tok_artifacts --text $'def hello_world():\n    print(\'Hello, world!\')\nhello_world()'
python -m scripts.tok_artifacts --text "Hola, ¿cómo estás?"
python -m scripts.tok_artifacts --text "정직한 사실 위에, 공정한 시선을 더하다"
```

- `scripts/tok_eval.py`: compares our tokenizer's compression (bytes/tokens
  ratio, plus the added `tok_per_char` metric) against GPT-2 and GPT-4 (cl100k)
  tokenizers on fixed news / Korean / code / math / science / CLIMBMix-train/val
  snippets.
- `scripts/tok_artifacts.py`: tokenizes a probe text and prints each token's
  id + decoded fragment, used to inspect failure cases for numbers, source code,
  and non-English text.

Note: his flow trains into the shared `~/.cache/nanochat/tokenizer` and only then
snapshots a copy; ours trains into per-vocab dirs directly and never touches the
shared tokenizer. Both cover vocab 8,192 and 32,768.

## Task 2 - Pretraining (`depth 2`, comparison at depth 4 and 8)

```bash
.venv/bin/python -m scripts.base_train --run dummy --depth 2 \
    --max-seq-len 2048 --window-pattern L --device-batch-size 16 \
    --eval-every 50 --eval-tokens 524288 --save-every 250
# comparison runs, same flags at --depth 4 and --depth 8
.venv/bin/python -m scripts.base_train --run dummy --depth 4 \
    --max-seq-len 2048 --window-pattern L --device-batch-size 16 \
    --eval-every 50 --eval-tokens 524288 --save-every 250
.venv/bin/python -m scripts.base_train --run dummy --depth 8 \
    --max-seq-len 2048 --window-pattern L --device-batch-size 8 \
    --eval-every 50 --eval-tokens 524288 --save-every 250
```

Logs: `runs/task2_d2.log`, `runs/task2_d4{,,_eval}.log`, `runs/task2_d8{,,_eval}.log`.
Final checkpoint: `checkpoints/d2/model_000420.pt` (`meta_000420.json`).

Validation bits-per-byte curves across the depth-2 run:

![task2 bpb curves](task2_bpb_curves.png)

Depth-2 vs depth-4/8 comparison:

![task2 bpb curves comparison](task2_bpb_curves_compare.png)

## Task 3 - Chat fine-tuning (Stage 1 mid-training, Stage 2 SFT)

```bash
# Stage 1: mid-training (MMLU + GSM8K) on top of the d2 base checkpoint
.venv/bin/python -m scripts.chat_sft --run task3_d2_mid --model-tag d2 \
    --model-step 420 --stage mid --source base --save-tag d2_mid
# Stage 2: SFT (SmolTalk) on top of the stage 1 checkpoint
.venv/bin/python -m scripts.chat_sft --run task3_d2_sft --model-tag d2_mid \
    --model-step 928 --stage sft --source sft --save-tag d2_sft
# held-out ARC / GSM-8K evaluation at each stage
.venv/bin/python -m scripts.chat_eval -i base  -g d2     -s 420   # base
.venv/bin/python -m scripts.chat_eval -i sft   -g d2_mid -s 928   # mid
.venv/bin/python -m scripts.chat_eval -i sft   -g d2_sft -s 2868  # sft
```

`scripts/chat_sft.py` persists the resolved hyper-parameters (max seq len, device
and total batch sizes, learning rates) into each checkpoint's meta and
`user_config`, so a later stage inherits exactly what the earlier stage used.

Logs: `runs/task3_*.log`. Checkpoints: `checkpoints/d2_mid/`, `checkpoints/d2_sft/`.

## Task 4 - Temperature study and inference

```bash
# interactive chat against the final d2_sft checkpoint at different temperatures
.venv/bin/python scripts/chat_cli.py -i sft -g d2_sft -s 2868 -t 0.1
.venv/bin/python scripts/chat_cli.py -i sft -g d2_sft -s 2868 -t 0.7
.venv/bin/python scripts/chat_cli.py -i sft -g d2_sft -s 2868 -t 1.5
```

Logs: `runs/task4temp/*.log`.
