"""prepare_openwebtext.py - tokenize OpenWebText (the Zenodo release, openwebtext.tar.xz) for train_gpt2.py.

The archive holds ~20,600 files openwebtext/urlsf_subset*_data.xz, each a compressed tar of ~390 documents
(.txt). Step 1 unpacks the outer archive once (~13 GB); step 2 tokenizes the inner files in parallel with
the GPT-2 tokenizer, each document followed by <|endoftext|> (50256), and writes uint16 tokens in a fixed
order to <out_dir>/train.bin and val.bin (val = the last n_val inner files, ~0.05% of the data, ~4M tokens).

Usage:  python prepare_openwebtext.py <openwebtext.tar.xz> <tokenizer_dir> <out_dir> [workers] [n_val]
Needs: numpy, transformers (tokenizer from a local folder: vocab.json, merges.txt, tokenizer_config.json).
"""
import sys, tarfile, lzma, io, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

arc, tok_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 20
n_val = int(sys.argv[5]) if len(sys.argv) > 5 else 10
tok = None

def init():
    global tok
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(tok_dir))
    assert len(tok) == 50257 and tok.eos_token_id == 50256

def tokenize_file(f):                                  # one inner .xz: a tar of .txt documents
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(Path(f).read_bytes())), mode="r:") as t:
        docs = [t.extractfile(m).read().decode("utf-8", errors="replace") for m in t.getmembers() if m.isfile()]
    ids = tok(docs)["input_ids"]
    return np.fromiter((x for d in ids for x in d + [50256]), dtype=np.uint16), len(docs)

if __name__ == "__main__":
    out.mkdir(parents=True, exist_ok=True)
    inner_dir = arc.parent / "openwebtext"
    if not inner_dir.exists():                         # 1. unpack the outer archive (once)
        t0 = time.time();  print("unpacking", arc, flush=True)
        with tarfile.open(arc, "r:xz") as t: t.extractall(arc.parent)
        print(f"unpacked in {time.time() - t0:.0f} s", flush=True)
    files = sorted(str(p) for p in inner_dir.glob("*.xz"))
    print(f"{len(files):,} inner files", flush=True)
    t0 = time.time()                                   # 2. tokenize in parallel, write in order
    with Pool(workers, initializer=init) as pool:
        for name, part in (("train", files[:-n_val]), ("val", files[-n_val:])):
            n_tok = n_doc = 0
            with open(out / f"{name}.bin", "wb") as fo:
                for i, (arr, nd) in enumerate(pool.imap(tokenize_file, part, chunksize=4)):
                    arr.tofile(fo);  n_tok += len(arr);  n_doc += nd
                    if (i + 1) % 1000 == 0:
                        print(f"{name}: {i + 1:,}/{len(part):,} files, {n_doc:,} docs, {n_tok / 1e9:.2f}B tokens, {time.time() - t0:.0f} s", flush=True)
            print(f"{name}: {n_doc:,} documents, {n_tok:,} tokens -> {out / (name + '.bin')}", flush=True)
