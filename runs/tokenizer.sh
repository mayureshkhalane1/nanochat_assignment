#!/bin/bash

# Download 5 data shards (~100 MB each as stated in speedrun.sh)
python -m nanochat.dataset -n 5


# Experiment 1: 8,192 vocab size
python -m scripts.tok_train --vocab-size 8192

cp -r ~/.cache/nanochat/tokenizer ~/.cache/nanochat/tokenizer_8192

python -m scripts.tok_eval

# Experiment 2: 32768 vocab size
python -m scripts.tok_train --vocab-size 32768

cp -r ~/.cache/nanochat/tokenizer ~/.cache/nanochat/tokenizer_32768

python -m scripts.tok_eval
