"""prepare_data.py - tokenize text into a binary token file for train_gpt2.py (nanoGPT format).

Each document is tokenized with the GPT-2 tokenizer and followed by <|endoftext|> (50256); all tokens
are written as uint16 to <out_dir>/train.bin and <out_dir>/val.bin (the last val_frac of the documents).

Usage:  python prepare_data.py <source> <tokenizer_dir> <out_dir> [val_frac]
  source         a .parquet file with a 'text' column (e.g. WikiText-2), or a folder of .parquet / .txt files
  tokenizer_dir  folder with vocab.json, merges.txt and tokenizer_config.json (the GPT-2 small model folder; nothing is downloaded)
Needs: numpy, pandas, pyarrow, transformers.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast

src, tok_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
val_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
out.mkdir(parents=True, exist_ok=True)
tok = GPT2TokenizerFast.from_pretrained(str(tok_dir))       # local folder (vocab.json, merges.txt, tokenizer_config.json)
assert len(tok) == 50257 and tok.eos_token_id == 50256, "tokenizer did not load the GPT-2 vocabulary"

# 1. Documents: rows of parquet 'text' columns, or whole .txt files
files = sorted(src.glob("*.parquet")) + sorted(src.glob("*.txt")) if src.is_dir() else [src]
docs = []
for f in files:
    if f.suffix == ".parquet":
        docs += [t for t in pd.read_parquet(f, columns=["text"])["text"] if t.strip()]
    else:
        docs.append(f.read_text(encoding="utf-8"))
print(f"{len(docs):,} documents from {len(files)} file(s)")

# 2. Tokenize, append <|endoftext|>, split by document, write uint16
n_val = max(1, int(len(docs) * val_frac))
for name, part in (("train", docs[:-n_val]), ("val", docs[-n_val:])):
    ids = tok(part)["input_ids"]
    arr = np.fromiter((t for d in ids for t in d + [tok.eos_token_id]), dtype=np.uint16)
    arr.tofile(out / f"{name}.bin")
    print(f"{name}: {len(arr):,} tokens -> {out / (name + '.bin')}")
