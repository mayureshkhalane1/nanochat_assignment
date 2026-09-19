from nanochat.tokenizer import get_tokenizer
import argparse

parser = argparse.ArgumentParser(description='Check artifacts tokenizer')
parser.add_argument('--text', type=str, help='Text to tokenize')
args = parser.parse_args()

if not args.text:
    raise ValueError("Text to tokenize is required")

tokenizer = get_tokenizer()

encoded = tokenizer.encode(args.text)

print(f"Text: {args.text}")
print("#tokens: ", len(encoded))

for id in encoded:
    print(f"{id}: {tokenizer.decode([id])}")




