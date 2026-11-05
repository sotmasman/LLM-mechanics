"""train_sae.py - train a sparse autoencoder (SAE) on GPT-2 small's residual stream (Lesson 9 tab 16).

Recipe: Anthropic's "Update on how we train SAEs" (Conerly et al., April 2024), used by Templeton et al. 2024:
  f = ReLU(W_e x + b_e),  x̂ = W_d f + b_d,
  loss = mean over the batch of ||x − x̂||² + λ Σ_i f_i ||W_d[:, i]||
  b_e = b_d = 0 at the start; W_d's columns point in random directions with length init_norm (0.1); W_e = W_dᵀ;
  x scaled by one constant c so that the mean ||x|| = sqrt(768) (c is fixed from the first buffer of activations);
  Adam (0.9, 0.999), no weight decay, lr 5e-5 decayed linearly to 0 over the last 20% of steps,
  gradient norm clipped to 1, λ raised linearly from 0 over the first 5% of steps; no resampling of dead features.
  After training the decoder norms are folded in: W_e' = ||W_d||·W_e, b_e' = ||W_d||·b_e, W_d' = W_d/||W_d|| (per feature).
Activations: the residual stream after layer `layer` (0-based; 5 = the middle of 12), from GPT-2 small built from
  gpt2_full_weights.mat with the same weight processing as the MATLAB model_forward (TransformerLens: LayerNorms folded,
  writing weights centered), so x is model_forward's x. Position 0 of each context (the attention sink, with very large
  activations) is left out.
Data: non-overlapping seq_len-token windows of train.bin in one fixed random order (seed), each used once (Bricken et
  al. 2023: sample without replacement). Activations of buffer_windows windows are kept on the GPU in random order;
  each training batch is taken from the first half, and when it is used up, that half is refilled with new windows
  and the whole buffer reshuffled, so no activation is used twice. Runs with the same seed see the same tokens.
Outputs in out_dir: log.csv (every log_every steps), resume.pt, sae.pt and sae.mat (folded weights and c), and after
  training eval.json / eval.mat:
  - on val.bin (val_tokens): variance explained, L0, next-token loss of the clean model, with the SAE's
    reconstruction spliced in at the layer (positions 1 on), and with the whole stream after the layer set to zero
    there (not only the layer's own output: embedding and layers 0 to `layer` are all removed), and the
    fraction of the loss recovered = (zero − SAE) / (zero − clean);
  - on eval_tokens training tokens not used in training: each feature's firing count (dead = never fires; Templeton
    et al. use 10 million tokens), mean activation when firing, and its top_k activations with their token's offset
    in train.bin (for the top-activating contexts).
In MATLAB, with x = model_forward's output of layer `layer` (768 × n):  f = max(0, W_enc * (c * x) + b_enc),
  x̂ = (W_dec * f + b_dec) / c.
Usage:  python train_sae.py sae_l5_lam5.json   (any key of S can be set in the JSON; resumes from resume.pt if present)
Needs: torch, numpy, scipy.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, scipy.io, torch, torch.nn.functional as F

S = dict(weights="GPT-2 small model/gpt2_full_weights.mat", data_dir="Datasets/owt", out_dir="runs/sae_l5_lam5",
         layer=5, d_sae=24576, lam=5.0, lr=5e-5, steps=200000, batch=4096, seq_len=1024, init_norm=0.1,
         lam_warmup=0.05, lr_decay=0.2, grad_clip=1.0, buffer_windows=2048, lm_batch=16, eval_batch=4, seed=1,
         log_every=100, resume_every=5000, val_tokens=2_000_000, eval_tokens=10_000_000, top_k=20)
if len(sys.argv) > 1: S.update(json.loads(Path(sys.argv[1]).read_text()))
out = Path(S["out_dir"]);  out.mkdir(parents=True, exist_ok=True)
(out / "settings.json").write_text(json.dumps(S, indent=1))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
amp = dict(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda")    # GPT-2 forward only; the SAE stays fp32

# 1. GPT-2 small from the MATLAB weights (every matrix stored out × in, head and layer last)
M = scipy.io.loadmat(S["weights"])
t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
nL = M["W_Q"].shape[3];  dh, d, nH = M["W_Q"].shape[:3]
E, P = t(M["W_E"].T), t(M["W_pos"].T)                                       # [V, d], [1024, d]
Wqkv = [t(np.concatenate([M[k][:, :, :, L].transpose(1, 2, 0).reshape(d, nH * dh) for k in ("W_Q", "W_K", "W_V")], 1)) for L in range(nL)]
bqkv = [t(np.concatenate([M[k][:, :, L].T.reshape(-1) for k in ("b_Q", "b_K", "b_V")])) for L in range(nL)]
Wo = [t(M["W_O"][:, :, :, L].transpose(2, 1, 0).reshape(nH * dh, d)) for L in range(nL)]
bo = [t(M["b_O"][:, L]) for L in range(nL)]
Win = [t(M["W_in"][:, :, L].T) for L in range(nL)];  bin_ = [t(M["b_in"][:, L]) for L in range(nL)]
Wout = [t(M["W_out"][:, :, L].T) for L in range(nL)];  bout = [t(M["b_out"][:, L]) for L in range(nL)]
WU, bU = t(M["W_U"].T), t(M["b_U"].reshape(-1));  del M

def ln(x):                                                                  # LayerNorm without scale/bias (folded in)
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()

def forward(tok, stop=None, splice=None):
    """tok [B, n] -> logits [B, n, V]; stop = L returns the stream after layer L; splice(x) replaces it after S['layer']."""
    B, n = tok.shape
    x = E[tok] + P[:n]
    for L in range(nL):
        q, k, v = (ln(x) @ Wqkv[L] + bqkv[L]).view(B, n, 3, nH, dh).transpose(1, 3).unbind(2)   # each [B, nH, n, dh]
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # softmax(q kᵀ / sqrt(64)) v
        x = x + z.transpose(1, 2).reshape(B, n, d) @ Wo[L] + bo[L]
        x = x + F.gelu(ln(x) @ Win[L] + bin_[L], approximate="tanh") @ Wout[L] + bout[L]
        if L == stop: return x
        if splice is not None and L == S["layer"]: x = splice(x)
    return ln(x) @ WU + bU

# 2. Data: windows in a fixed random order; training windows first, then the eval windows
data = {k: np.memmap(Path(S["data_dir"]) / f"{k}.bin", dtype=np.uint16, mode="r") for k in ("train", "val")}
n = S["seq_len"];  per = n - 1                                             # tokens per window (position 0 left out)
order = np.random.default_rng(S["seed"]).permutation((len(data["train"]) - 1) // n)
nTrain = math.ceil(S["steps"] * S["batch"] / per) + S["buffer_windows"]
evalWin = order[nTrain:nTrain + math.ceil(S["eval_tokens"] / per)]
assert len(evalWin) * per >= S["eval_tokens"], "train.bin too small for steps × batch + eval_tokens"
windows = lambda split, ids: torch.from_numpy(np.stack([data[split][i * n:(i + 1) * n] for i in ids]).astype(np.int64)).to(dev)

@torch.no_grad()
def acts(split, ids):                                                       # [len(ids)·(n−1), d], unscaled, fp16
    out = []
    for i in range(0, len(ids), S["lm_batch"]):
        with torch.autocast(**amp):
            out.append(forward(windows(split, ids[i:i + S["lm_batch"]]), stop=S["layer"])[:, 1:].reshape(-1, d).half())
    return torch.cat(out)

# 3. SAE parameters and optimizer (resume if resume.pt exists)
m = S["d_sae"];  g = torch.Generator().manual_seed(S["seed"])
Wd = torch.randn(d, m, generator=g);  Wd *= S["init_norm"] / Wd.norm(dim=0, keepdim=True)
p = {"We": Wd.T.clone(), "be": torch.zeros(m), "Wd": Wd, "bd": torch.zeros(d)}
p = {k: torch.nn.Parameter(v.to(dev)) for k, v in p.items()}
opt = torch.optim.Adam(p.values(), lr=S["lr"], betas=(0.9, 0.999), weight_decay=0.0)
step, ptr, c = 0, 0, None                                                   # ptr: next training window in `order`
if (out / "resume.pt").exists():
    r = torch.load(out / "resume.pt", map_location=dev, weights_only=False)
    for k in p: p[k].data.copy_(r["p"][k])
    opt.load_state_dict(r["opt"]);  step, ptr, c = r["step"], r["ptr"], r["c"]
    lines = (out / "log.csv").read_text().splitlines(keepends=True)
    (out / "log.csv").write_text("".join(l for l in lines if not l[0].isdigit() or int(l.split(",")[0]) <= step))
    print(f"resumed at step {step} (the buffer is refilled from window {ptr})", flush=True)

def sae(x):                                                                 # x [N, d], scaled
    f = F.relu(x @ p["We"].T + p["be"])
    return f, f @ p["Wd"].T + p["bd"]

# 4. Buffer of activations
half = S["buffer_windows"] // 2
buf = acts("train", order[ptr:ptr + 2 * half]);  ptr += 2 * half
if c is None: c = math.sqrt(d) / buf.float().norm(dim=1).mean().item()     # mean ||c·x|| = sqrt(768)
buf = buf[torch.randperm(len(buf), device=dev)];  used = 0
def next_batch():
    global buf, used, ptr
    if used + S["batch"] > len(buf) // 2:                                   # first half used: refill it, reshuffle all
        buf[:half * per] = acts("train", order[ptr:ptr + half]);  ptr += half
        buf = buf[torch.randperm(len(buf), device=dev)];  used = 0
    x = buf[used:used + S["batch"]].float() * c;  used += S["batch"]
    return x

# 5. Training
lr_at = lambda s: S["lr"] * min(1.0, (S["steps"] - s) / (S["lr_decay"] * S["steps"]))
lam_at = lambda s: S["lam"] * min(1.0, s / (S["lam_warmup"] * S["steps"]))
dead_steps = math.ceil(10_000_000 / S["batch"])                             # "dead": no firing in the last 10M tokens
last = torch.full((m,), step, device=dev)
log = open(out / "log.csv", "a")
if step == 0: log.write("step,lam,lr,loss,mse,sparsity,L0,var_explained,dead_frac,seconds\n")
t0 = time.time()
while step < S["steps"]:
    step += 1;  lam = lam_at(step)
    for gr in opt.param_groups: gr["lr"] = lr_at(step)
    x = next_batch()
    f, xh = sae(x)
    mse = (x - xh).pow(2).sum(1).mean()
    spars = (f * p["Wd"].norm(dim=0)).sum(1).mean()
    loss = mse + lam * spars
    opt.zero_grad(set_to_none=True);  loss.backward()
    torch.nn.utils.clip_grad_norm_(p.values(), S["grad_clip"]);  opt.step()
    with torch.no_grad():
        last[(f > 0).any(0)] = step
        if step % S["log_every"] == 0 or step == S["steps"]:
            L0 = (f > 0).float().sum(1).mean().item()
            ve = 1 - (x - xh).pow(2).sum().item() / (x - x.mean(0)).pow(2).sum().item()
            dead = ((step - last) >= dead_steps).float().mean().item() if step >= dead_steps else float("nan")
            log.write(f"{step},{lam:.4g},{lr_at(step):.4g},{loss.item():.5g},{mse.item():.5g},{spars.item():.5g},"
                      f"{L0:.2f},{ve:.4f},{dead:.4f},{time.time() - t0:.1f}\n");  log.flush()
            print(f"step {step:6d}  loss {loss.item():.4f}  mse {mse.item():.4f}  L0 {L0:.1f}  var.expl. {ve:.3f}  "
                  f"dead {dead:.3f}  {time.time() - t0:.0f} s", flush=True)
    if step % S["resume_every"] == 0 or step == S["steps"]:
        torch.save({"p": {k: v.detach() for k, v in p.items()}, "opt": opt.state_dict(), "step": step, "ptr": ptr, "c": c},
                   out / "resume.pt")
log.close();  del buf, x, f, xh, loss, mse, spars;  torch.cuda.empty_cache() if dev == "cuda" else None

# 6. Fold the decoder norms into the encoder (same predictions, unit-length decoder columns) and save
with torch.no_grad():
    nrm = p["Wd"].norm(dim=0)
    for k, v in (("We", p["We"] * nrm[:, None]), ("be", p["be"] * nrm), ("Wd", p["Wd"] / nrm)): p[k].data.copy_(v)
W = {k: v.detach().cpu() for k, v in p.items()}
torch.save({**W, "c": c, "settings": S}, out / "sae.pt")
scipy.io.savemat(out / "sae.mat", {"W_enc": W["We"].numpy(), "b_enc": W["be"].numpy()[:, None], "W_dec": W["Wd"].numpy(),
                                   "b_dec": W["bd"].numpy()[:, None], "c": c, "layer": S["layer"]})

# 7. Evaluation
ev = {}
with torch.no_grad():
    # 7a. val.bin: reconstruction and next-token loss (clean, SAE spliced in, stream after the layer set to zero)
    vw = np.arange(min(math.ceil(S["val_tokens"] / per), (len(data["val"]) - 1) // n))
    sse = sx2 = l0 = nTok = 0.0;  sx = torch.zeros(d, device=dev);  ce = {"clean": 0.0, "sae": 0.0, "zero": 0.0}
    def rec(x):                                                             # positions 1 on -> SAE reconstruction (fp32)
        with torch.autocast(device_type="cuda", enabled=False):
            _, xh = sae(x[:, 1:].float() * c)
        return torch.cat([x[:, :1].float(), xh.view(x.shape[0], -1, d) / c], 1)
    def zero(x): return torch.cat([x[:, :1], torch.zeros_like(x[:, 1:])], 1)
    for i in range(0, len(vw), S["eval_batch"]):                          # eval_batch windows at a time (GPU memory)
        tok = windows("val", vw[i:i + S["eval_batch"]])
        with torch.autocast(**amp):
            x = forward(tok, stop=S["layer"])[:, 1:].reshape(-1, d).float() * c
        f, xh = sae(x)
        sse += (x - xh).pow(2).sum().item();  sx2 += x.pow(2).sum().item();  sx += x.sum(0)
        l0 += (f > 0).float().sum().item();  nTok += len(x)
        for k, fn in (("clean", None), ("sae", rec), ("zero", zero)):
            with torch.autocast(**amp):
                lg = forward(tok, splice=fn)
            for b in range(len(tok)):                                       # one window at a time (fp32 logits are large)
                ce[k] += F.cross_entropy(lg[b, :-1].float(), tok[b, 1:], reduction="sum").item()
    nPred = len(vw) * (n - 1)
    sst = sx2 - sx.pow(2).sum().item() / nTok                                # Σ ||x − mean x||² over all val tokens
    ev = {"val_tokens": int(nTok), "var_explained": 1 - sse / sst, "L0": l0 / nTok, "c": c,
          **{f"loss_{k}": v / nPred for k, v in ce.items()}}
    ev["loss_recovered"] = (ev["loss_zero"] - ev["loss_sae"]) / (ev["loss_zero"] - ev["loss_clean"])
    # 7b. unused training tokens: firing counts, mean activation, top_k activations per feature
    K = S["top_k"];  count = torch.zeros(m, device=dev);  total = torch.zeros(m, device=dev)
    topv = torch.zeros(K, m, device=dev);  topi = torch.full((K, m), -1, dtype=torch.long, device=dev)
    for i in range(0, len(evalWin), S["eval_batch"]):
        ids = evalWin[i:i + S["eval_batch"]]
        f, _ = sae(acts("train", ids).float() * c)                           # [len(ids)·(n−1), m]
        offs = torch.from_numpy((ids[:, None] * n + np.arange(1, n)[None, :]).reshape(-1)).to(dev)   # token offsets in train.bin
        count += (f > 0).float().sum(0);  total += f.sum(0)
        vb, jb = f.topk(K, dim=0)                                           # this chunk's top K, then merge
        v, j = torch.cat([topv, vb]).topk(K, dim=0)
        topi = torch.cat([topi, offs[jb]]).gather(0, j);  topv = v
    ev.update({"eval_tokens": len(evalWin) * per, "dead_frac": (count == 0).float().mean().item(),
               "dense_frac": (count / (len(evalWin) * per) > 0.01).float().mean().item()})
(out / "eval.json").write_text(json.dumps(ev, indent=1))
scipy.io.savemat(out / "eval.mat", {"count": count.cpu().numpy(), "mean_act": (total / count.clamp_min(1)).cpu().numpy(),
                                    "top_val": topv.cpu().numpy(), "top_offset": topi.cpu().numpy().astype(np.float64),
                                    "eval_tokens": float(len(evalWin) * per)})
print(json.dumps(ev, indent=1));  print("done", flush=True)
