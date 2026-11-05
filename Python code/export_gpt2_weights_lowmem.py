"""
Export GPT-2 weights (any size: small, medium, large, XL) to one .mat file for
the MATLAB forward pass, using little memory. Written for GPT-2 XL, which is
too large for export_gpt2_weights.py on a 16 GB machine (TransformerLens briefly
holds two full copies of the model).

Usage:   py export_gpt2_weights_lowmem.py ["model folder"] ["output .mat"]
Default: <repository>\\GPT-2 XL model -> <model folder>\\gpt2_full_weights.mat

How: reads model.safetensors one tensor at a time and writes each MATLAB array
to the .mat file as soon as it is built, so at most one array (~2 GB for XL's
W_in) is held at once. It does the same weight processing as
TransformerLens HookedTransformer.from_pretrained(..., fold_ln=True,
center_unembed=True) with its defaults center_writing_weights=True and
fold_value_biases=True, in this order:
  1. fold_ln                 LayerNorm scale/bias folded into the next weights;
                             the reading weights (W_Q, W_K, W_V, W_in, W_U) are
                             centered over d_model
  2. center_writing_weights  W_E, W_pos, W_O, b_O, W_out, b_out centered over d_model
  3. center_unembed          W_U and b_U centered over the vocabulary
  4. fold_value_biases       b_V folded into b_O (b_V becomes 0)
Same layout as export_gpt2_weights.py: every matrix [out × in], head and layer
as the last dimensions. Checked against export_gpt2_weights.py on GPT-2 small.
"""

import sys
import numpy as np
import scipy.io
from scipy.io.matlab._mio5 import MatFile5Writer
from safetensors import safe_open
from pathlib import Path
import json

MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "GPT-2 XL model"
OUT_FILE  = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_DIR / "gpt2_full_weights.mat"

hf = json.loads((MODEL_DIR / "config.json").read_text())
nL, nH, d = hf["n_layer"], hf["n_head"], hf["n_embd"]
dh, dm, V = d // nH, 4 * d, hf["vocab_size"]
print(f"{MODEL_DIR.name}: {nL} layers, {nH} heads, d_model {d}, d_mlp {dm}")

if (MODEL_DIR / "model.safetensors").exists():
    st = safe_open(str(MODEL_DIR / "model.safetensors"), framework="np")
    get, keys = st.get_tensor, st.keys()
else:                                           # pytorch_model.bin (e.g. Stanford CRFM checkpoints)
    import torch
    sd = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    get, keys = (lambda k: sd[k].float().numpy()), sd.keys()
prefix = "transformer." if "transformer.wte.weight" in keys else ""
def T(name):                                    # one tensor, float64 for exact folding
    return get(prefix + name).astype(np.float64)

out = open(OUT_FILE, "wb")
writer = MatFile5Writer(out)
def write(name, arr):                           # write one array now and free it
    writer.put_variables({name: np.asarray(arr, dtype=np.float32)})
    print(f"  wrote {name:6s} {arr.shape}")

def attn(L):
    """Attention weights of layer L in TransformerLens shapes, fully processed."""
    w, b = T(f"h.{L}.attn.c_attn.weight"), T(f"h.{L}.attn.c_attn.bias")      # [d, 3d], [3d]
    g, beta = T(f"h.{L}.ln_1.weight"), T(f"h.{L}.ln_1.bias")
    r = {}
    for i, k in enumerate("QKV"):
        W = w[:, i * d:(i + 1) * d].reshape(d, nH, dh).transpose(1, 0, 2)   # [head, d_model, d_head]
        bb = b[i * d:(i + 1) * d].reshape(nH, dh)                           # [head, d_head]
        bb = bb + np.einsum("m,imh->ih", beta, W)                           # fold LN bias
        W = W * g[None, :, None]                                            # fold LN scale
        W = W - W.mean(axis=1, keepdims=True)                               # center over d_model
        r["W_" + k], r["b_" + k] = W, bb
    WO = T(f"h.{L}.attn.c_proj.weight").reshape(nH, dh, d)                  # [head, d_head, d_model]
    WO = WO - WO.mean(axis=2, keepdims=True)                                # center writing weights
    bO = T(f"h.{L}.attn.c_proj.bias")
    bO = bO - bO.mean()
    r["b_O"] = bO + np.einsum("ih,ihm->m", r["b_V"], WO)                     # fold value biases
    r["b_V"] = np.zeros_like(r["b_V"])
    r["W_O"] = WO
    return r

# ── Embeddings ───────────────────────────────────────────────────────────────
wte = T("wte.weight")                                                       # [V, d]
write("W_E", (wte - wte.mean(axis=1, keepdims=True)).T)                     # d × V
wpe = T("wpe.weight")
write("W_pos", (wpe - wpe.mean(axis=1, keepdims=True)).T)                   # d × 1024
del wpe

# ── Unembedding (tied to wte) ────────────────────────────────────────────────
g, beta = T("ln_f.weight"), T("ln_f.bias")
WU = wte.T                                                                  # [d, V]
bU = beta @ WU                                                              # fold LN bias
WU = WU * g[:, None]                                                        # fold LN scale
WU = WU - WU.mean(axis=0, keepdims=True)                                    # center over d_model
WU = WU - WU.mean(axis=1, keepdims=True)                                    # center_unembed
bU = bU - bU.mean()
write("W_U", WU.T)                                                          # V × d
write("b_U", bU.reshape(1, -1))                                             # 1 × V
del wte, WU

# ── Attention: Q, K, V, O (one array at a time, built layer by layer) ────────
for name, shape in [("W_Q", (dh, d, nH, nL)), ("W_K", (dh, d, nH, nL)),
                    ("W_V", (dh, d, nH, nL)), ("W_O", (d, dh, nH, nL))]:
    A = np.empty(shape, dtype=np.float32)
    for L in range(nL):
        A[..., L] = attn(L)[name].transpose(2, 1, 0)                        # [head, a, b] -> b × a × head
    write(name, A); del A
for name in ["b_Q", "b_K", "b_V"]:
    write(name, np.stack([attn(L)[name].T for L in range(nL)], axis=2))     # d_head × head × layer
write("b_O", np.stack([attn(L)["b_O"] for L in range(nL)], axis=1))         # d_model × layer

# ── MLP ──────────────────────────────────────────────────────────────────────
A = np.empty((dm, d, nL), dtype=np.float32); b_in = np.empty((dm, nL))
for L in range(nL):
    W = T(f"h.{L}.mlp.c_fc.weight")                                         # [d, d_mlp]
    g, beta = T(f"h.{L}.ln_2.weight"), T(f"h.{L}.ln_2.bias")
    b_in[:, L] = T(f"h.{L}.mlp.c_fc.bias") + beta @ W
    W = W * g[:, None]
    A[:, :, L] = (W - W.mean(axis=0, keepdims=True)).T                     # d_mlp × d_model
write("W_in", A); del A
write("b_in", b_in)
A = np.empty((d, dm, nL), dtype=np.float32); b_out = np.empty((d, nL))
for L in range(nL):
    W = T(f"h.{L}.mlp.c_proj.weight")                                       # [d_mlp, d]
    A[:, :, L] = (W - W.mean(axis=1, keepdims=True)).T                     # d_model × d_mlp
    bb = T(f"h.{L}.mlp.c_proj.bias")
    b_out[:, L] = bb - bb.mean()
write("W_out", A); del A
write("b_out", b_out)

cfg = {"n_layers": nL, "n_heads": nH, "d_model": d, "d_head": dh, "d_mlp": dm,
       "n_ctx": hf["n_positions"], "n_vocab": V, "ln_eps": hf["layer_norm_epsilon"],
       "attn_scale": float(np.sqrt(dh)), "act_fn": hf["activation_function"],
       "scale_attn_by_inverse_layer_idx": bool(hf.get("scale_attn_by_inverse_layer_idx", False)),
       "export_settings": "export_gpt2_weights_lowmem.py: same processing as TransformerLens "
                          "from_pretrained(fold_ln=True, center_unembed=True) with defaults "
                          "center_writing_weights=True, fold_value_biases=True; "
                          "layer norms are LayerNormPre (center and scale only)"}
writer.put_variables({"cfg": cfg})
out.close()
print(f"Saved {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.0f} MB)")
