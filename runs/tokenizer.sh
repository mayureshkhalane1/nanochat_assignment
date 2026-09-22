#!/bin/bash

# Download 5 data shards (~100 MB each as stated in speedrun.sh)
python -m nanochat.dataset -n 5


# Experiment 1: 8,192 vocab size
python -m scripts.tok_train --vocab-size 8192

cp -r ~/.cache/nanochat/tokenizer ~/.cache/nanochat/tokenizer_8192

python -m scripts.tok_eval

# check for numbers
python -m scripts.tok_artifacts --text "1,586,291 people live in country 224393."

# check for source code
python -m scripts.tok_artifacts --text "
def hello_world():
    print('Hello, world!')
hello_world()"

# check for non-english text
python -m scripts.tok_artifacts --text "Hola, ¿cómo estás?"
python -m scripts.tok_artifacts --text "정직한 사실 위에, 공정한 시선을 더하다"


# Experiment 2: 32768 vocab size
python -m scripts.tok_train --vocab-size 32768

cp -r ~/.cache/nanochat/tokenizer ~/.cache/nanochat/tokenizer_32768

python -m scripts.tok_eval


# check for numbers
python -m scripts.tok_artifacts --text "1,586,291 people live in country 224393."

# check for source code
python -m scripts.tok_artifacts --text "
def hello_world():
    print('Hello, world!')
hello_world()"

# check for non-english text
python -m scripts.tok_artifacts --text "Hola, ¿cómo estás?"
python -m scripts.tok_artifacts --text "정직한 사실 위에, 공정한 시선을 더하다"
