#!/bin/bash

# Download 5 data shards (~100 MB each as stated in speedrun.sh)
python -m nanochat.dataset -n 5

python -m scripts.tok_train --vocab-size 32768

python -m scripts.tok_eval