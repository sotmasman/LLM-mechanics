"""
Export Pythia (GPT-NeoX) weights to one .mat file for the MATLAB forward pass
(model_forward.m), in the same layout and with the same processing as
export_gpt2_weights_lowmem.py, so the GPT-2 scripts run unchanged.

Usage:   py export_pythia_weights.py ["model folder"] ["output .mat"]
Default: <repository>\\Pythia 160M model -> <model folder>\\gpt2_full_weights.mat
Reads model.safetensors (or pytorch_model.bin, e.g. for training checkpoints)
and config.json. Also writes vocab.json (from tokenizer.json) for model_token.

Pythia vs GPT-2 (handled here and in model_forward via cfg):
  - no positional embedding: rotary embeddings on the first rotary_dim (25%) of
    each head's query and key dimensions. W_pos is written as zeros so that
    code adding W_E + W_pos still works.
  - parallel residual: attention and MLP both read the residual stream entering
    the layer (each through its own layer norm) and both write to it:
    x_next = x + attn(LN1(x)) + mlp(LN2(x)).
  - separate (untied) unembedding, embed_out, without bias.
  - fused query_key_value weight, laid out per head as [q (d_head), k, v].
  - exact (erf) GELU; vocabulary padded to 50304 (tokens 50277+ unused).
Processing, as TransformerLens: fold_ln (each layer norm folded into the weights
that read it; reading weights centered over d_model), center_writing_weights
(W_E, W_O, b_O, W_out, b_out centered over d_model), center_unembed,
fold_value_biases.
"""

import sys, json
import numpy as np
from scipy.io.matlab._mio5 import MatFile5Writer
from pathlib import Path

MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "Pythia 160M model"
OUT_FILE  = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_DIR / "gpt2_full_weights.mat"

hf = json.loads((MODEL_DIR / "config.json").read_text())
nL, nH, d = hf["num_hidden_layers"], hf["num_attention_heads"], hf["hidden_size"]
dh, dm, V = d // nH, hf["intermediate_size"], hf["vocab_size"]
rot = int(dh * hf["rotary_pct"])
print(f"{MODEL_DIR.name}: {nL} layers, {nH} heads, d_model {d}, d_mlp {dm}, rotary dims {rot}")
assert hf.get("use_parallel_residual", True), "only parallel-residual GPT-NeoX is supported"

if (MODEL_DIR / "model.safetensors").exists():
    from safetensors import safe_open
    st = safe_open(str(MODEL_DIR / "model.safetensors"), framework="np")
    get = st.get_tensor
else:
    import torch
    sd = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    get = lambda k: sd[k].float().numpy()
def T(name):                                    # one tensor, float64 for exact folding
    return get(name).astype(np.float64)
P = "gpt_neox."

out = open(OUT_FILE, "wb")
writer = MatFile5Writer(out)
def write(name, arr):                           # write one array now and free it
    writer.put_variables({name: np.asarray(arr, dtype=np.float32)})
    print(f"  wrote {name:6s} {arr.shape}")

def attn(L):
    """Attention weights of layer L as [head, d_model, d_head] (in -> out), fully processed."""
    pre = f"{P}layers.{L}."
    w = T(pre + "attention.query_key_value.weight").reshape(nH, 3, dh, d)   # [head, q/k/v, d_head, d_model]
    b = T(pre + "attention.query_key_value.bias").reshape(nH, 3, dh)
    g, beta = T(pre + "input_layernorm.weight"), T(pre + "input_layernorm.bias")
    r = {}
    for i, k in enumerate("QKV"):
        W = w[:, i].transpose(0, 2, 1)                                      # [head, d_model, d_head]
        bb = b[:, i] + np.einsum("m,imh->ih", beta, W)                      # fold LN bias
        W = W * g[None, :, None]                                            # fold LN scale
        W = W - W.mean(axis=1, keepdims=True)                               # center over d_model
        r["W_" + k], r["b_" + k] = W, bb
    WO = T(pre + "attention.dense.weight").T.reshape(nH, dh, d)             # [head, d_head, d_model]
    WO = WO - WO.mean(axis=2, keepdims=True)                                # center writing weights
    bO = T(pre + "attention.dense.bias")
    bO = bO - bO.mean()
    r["b_O"] = bO + np.einsum("ih,ihm->m", r["b_V"], WO)                     # fold value biases
    r["b_V"] = np.zeros_like(r["b_V"])
    r["W_O"] = WO
    return r

# ── Embeddings (no positional embedding: zeros, for compatibility) ───────────
we = T(P + "embed_in.weight")                                              # [V, d]
write("W_E", (we - we.mean(axis=1, keepdims=True)).T)                      # d × V
del we
write("W_pos", np.zeros((d, hf["max_position_embeddings"])))              # d × n_ctx, all zero

# ── Unembedding (untied, no bias) ────────────────────────────────────────────
g, beta = T(P + "final_layer_norm.weight"), T(P + "final_layer_norm.bias")
WU = T("embed_out.weight").T                                                # [d, V]
bU = beta @ WU                                                              # fold LN bias
WU = WU * g[:, None]                                                        # fold LN scale
WU = WU - WU.mean(axis=0, keepdims=True)                                    # center over d_model
WU = WU - WU.mean(axis=1, keepdims=True)                                    # center_unembed
bU = bU - bU.mean()
write("W_U", WU.T)                                                          # V × d
write("b_U", bU.reshape(1, -1))                                             # 1 × V
del WU

# ── Attention: Q, K, V, O ────────────────────────────────────────────────────
for name, shape in [("W_Q", (dh, d, nH, nL)), ("W_K", (dh, d, nH, nL)),
                    ("W_V", (dh, d, nH, nL)), ("W_O", (d, dh, nH, nL))]:
    A = np.empty(shape, dtype=np.float32)
    for L in range(nL):
        A[..., L] = attn(L)[name].transpose(2, 1, 0)                        # [head, a, b] -> b × a × head
    write(name, A); del A
for name in ["b_Q", "b_K", "b_V"]:
    write(name, np.stack([attn(L)[name].T for L in range(nL)], axis=2))     # d_head × head × layer
write("b_O", np.stack([attn(L)["b_O"] for L in range(nL)], axis=1))         # d_model × layer

# ── MLP (reads LN2 of the residual stream entering the layer) ───────────────
A = np.empty((dm, d, nL), dtype=np.float32); b_in = np.empty((dm, nL))
for L in range(nL):
    pre = f"{P}layers.{L}."
    W = T(pre + "mlp.dense_h_to_4h.weight").T                               # [d, d_mlp]
    g, beta = T(pre + "post_attention_layernorm.weight"), T(pre + "post_attention_layernorm.bias")
    b_in[:, L] = T(pre + "mlp.dense_h_to_4h.bias") + beta @ W
    W = W * g[:, None]
    A[:, :, L] = (W - W.mean(axis=0, keepdims=True)).T                     # d_mlp × d_model
write("W_in", A); del A
write("b_in", b_in)
A = np.empty((d, dm, nL), dtype=np.float32); b_out = np.empty((d, nL))
for L in range(nL):
    pre = f"{P}layers.{L}."
    W = T(pre + "mlp.dense_4h_to_h.weight").T                               # [d_mlp, d]
    A[:, :, L] = (W - W.mean(axis=1, keepdims=True)).T                     # d_model × d_mlp
    bb = T(pre + "mlp.dense_4h_to_h.bias")
    b_out[:, L] = bb - bb.mean()
write("W_out", A); del A
write("b_out", b_out)

cfg = {"arch": "neox", "n_layers": nL, "n_heads": nH, "d_model": d, "d_head": dh, "d_mlp": dm,
       "n_ctx": hf["max_position_embeddings"], "n_vocab": V, "ln_eps": hf["layer_norm_eps"],
       "attn_scale": float(np.sqrt(dh)), "act_fn": hf["hidden_act"],
       "rotary_dim": rot, "rotary_base": float(hf["rotary_emb_base"]),
       "parallel_residual": True, "bos_id": int(hf["bos_token_id"]),
       "export_settings": "export_pythia_weights.py: TransformerLens-style processing "
                          "(fold_ln, center_writing_weights, center_unembed, fold_value_biases); "
                          "layer norms are LayerNormPre (center and scale only)"}
writer.put_variables({"cfg": cfg})
out.close()
print(f"Saved {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.0f} MB)")

# vocab.json for model_token / model_decode (same byte-level encoding as GPT-2)
tj = MODEL_DIR / "tokenizer.json"
if tj.exists() and not (MODEL_DIR / "vocab.json").exists():
    voc = json.loads(tj.read_text(encoding="utf-8"))["model"]["vocab"]
    (MODEL_DIR / "vocab.json").write_text(json.dumps(voc, ensure_ascii=False), encoding="utf-8")
    print("Saved vocab.json")
