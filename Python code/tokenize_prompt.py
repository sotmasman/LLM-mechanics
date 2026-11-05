"""
Tokenize a prompt with a model's tokenizer only (no model), and print its token IDs.
Used by Matlab code\model_tokenize.m.

    py tokenize_prompt.py <file with the prompt, UTF-8> [model folder]

Prints one line of space-separated IDs, starting with <|endoftext|> (50256 for
GPT-2, 0 for Pythia), the same way every saved prompt starts. The prompt is read
from a file so that any characters (Greek, accents) survive the Windows command line.

Model folder: default GPT-2 small (all GPT-2 sizes share the tokenizer). A folder
with tokenizer.json (Pythia) uses that tokenizer and the bos_token_id in its
config.json. Uses the `tokenizers` library, which gives exactly the same IDs as
the Hugging Face / TransformerLens tokenizer but starts in well under a second.
"""

import sys, json
from pathlib import Path

MODEL_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "GPT-2 small model"
if (MODEL_DIR / "tokenizer.json").exists():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    bos = json.loads((MODEL_DIR / "config.json").read_text())["bos_token_id"]
    encode = lambda t: tok.encode(t, add_special_tokens=False).ids
else:
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(MODEL_DIR / "vocab.json"), str(MODEL_DIR / "merges.txt"))
    bos = 50256
    encode = lambda t: tok.encode(t).ids

with open(sys.argv[1], encoding="utf-8") as f:
    text = f.read()

if sys.argv[1].endswith(".json"):        # a JSON list of prompts: one line of IDs per prompt
    for t in json.loads(text):
        print(" ".join(map(str, [bos] + encode(t))))
else:
    ids = [bos] + encode(text)           # <|endoftext|> first
    print(" ".join(map(str, ids)))
