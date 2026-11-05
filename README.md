# LLM Mechanics

Nine lessons on how large language models work and how to study them, from the artificial neuron to
hands-on interpretability experiments on GPT-2, with MATLAB and Python code that reproduces the experiments.

**Website:** https://sotmasman.github.io/LLM-mechanics/

> **Developer's note.** These lessons and code were designed for me (Sotiris) to learn about the mechanics of
> language models. The lessons and code were created by Claude Opus 5.5 (Anthropic) and underwent multiple
> rounds of revisions in consultation with me. The material is primarily aimed at systems/computational
> neuroscience readers interested in getting an introduction into how language models work and how they are
> studied. I hope you find it a useful learning tool. Please contact me with any feedback.
> —Sotiris

| Folder | What it holds |
|---|---|
| `docs/` | The website: `index.html`, `lesson1.html` … `lesson9.html`, and `img/` (the figures of Lesson 9). Static HTML, no build step. |
| `Matlab code/` | The analysis code of Lesson 9: a MATLAB implementation of GPT-2's forward pass (`model_forward`) and one script per method. |
| `Python code/` | Small helpers called by the MATLAB code (tokenizer, weight export), data preparation, and the sparse-autoencoder training of Lesson 9, tab 16. |

Lessons 1–8 need nothing but a browser; their interactive figures run in the page. Lesson 9 runs real
experiments on GPT-2, and its figures were made with the code in this repository.

## Reading the lessons

Online at https://sotmasman.github.io/LLM-mechanics/. To read them offline, download this repository and
serve the `docs/` folder with any local web server, e.g. `python -m http.server` run inside `docs/`, then
open http://localhost:8000. Offline, the pages fall back to system fonts, and Lesson 1's live tokenizer demo
(which loads `gpt-tokenizer` from jsDelivr) needs an internet connection; everything else is in the folder.

## What the code needs

### Software

| | Needed for | Notes |
|---|---|---|
| **MATLAB** | everything in Lesson 9 | Developed and tested with R2024b. |
| Statistics and Machine Learning Toolbox | `run_probe` (`fitclinear`), `run_RSA` and `run_PCA` (`corr`, `tiedrank`) | |
| Deep Learning Toolbox | `run_rome` (gradients through `model_forward` with `dlarray` / `dlgradient`) | |
| Parallel Computing Toolbox | optional: `run_rome` with `useGPU = true` | GPT-2 XL needs ~10 GB of GPU memory. |
| **Python 3** (tested with 3.11) | tokenizing prompts (every script), exporting the weights once | MATLAB calls Python itself: `py` on Windows, `python3` elsewhere. Set the environment variable `LLM_PYTHON` to the full path of another interpreter (e.g. a conda environment) if needed. |
| `tokenizers` | `tokenize_prompt.py` (called by `model_tokenize` for every prompt) | |
| `numpy`, `scipy`, `safetensors` | `export_gpt2_weights_lowmem.py`, `export_pythia_weights.py` | |
| `transformers`, `pandas`, `pyarrow` | `prepare_data.py`, `prepare_openwebtext.py` (tab 15 and tab 16 data only) | |
| `torch` with a CUDA GPU | `train_sae.py` (tab 16 only) | ~2 h 50 min per SAE on one RTX 3080 Ti (12 GB). |

```bash
pip install tokenizers numpy scipy safetensors
pip install transformers pandas pyarrow   # only for the tab 15 and tab 16 data
pip install torch                         # only for training the SAEs (tab 16); use a CUDA build
```

### Models (Hugging Face)

Download these files of each model into a folder with **exactly** the name in the last column, next to
`Matlab code/`: `config.json`, `model.safetensors`, `vocab.json`, `merges.txt`, `tokenizer_config.json`
(for Pythia also `tokenizer.json`).

| Model | Hugging Face repository | Used in Lesson 9 | Folder |
|---|---|---|---|
| GPT-2 small (124 M) | [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) | every tab (required) | `GPT-2 small model` |
| GPT-2 medium, large | [`openai-community/gpt2-medium`](https://huggingface.co/openai-community/gpt2-medium), [`openai-community/gpt2-large`](https://huggingface.co/openai-community/gpt2-large) | tab 9, the probe across model sizes (optional) | `GPT-2 medium model`, `GPT-2 large model` |
| GPT-2 XL (1.5 B) | [`openai-community/gpt2-xl`](https://huggingface.co/openai-community/gpt2-xl) | tab 9 (model sizes), tab 12 (patching in GPT-2 XL) and tab 15 (ROME) | `GPT-2 XL model` |
| Pythia-160M | [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m) | optional: the scripts accept `model = 'pythia-160m'` | `Pythia 160M model` |

The first time a script loads a model, `model_weights_file` converts the weights to a MATLAB file in the
same folder (`gpt2_full_weights.mat`, ~650 MB for GPT-2 small, about a minute; ~10 minutes for XL) by
calling `Python code/export_gpt2_weights_lowmem.py`. After that the scripts only read the `.mat` file. The
conversion folds the LayerNorm weights into the next layer and centres the writing weights, as
TransformerLens does, so values match TransformerLens with its default settings.

### Data (only for tabs 15 and 16)

| Data | Used for | How |
|---|---|---|
| WikiText-2 (raw), training split: `train-00000-of-00001.parquet` from [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1` | tab 15: the key covariance that ROME needs (`rome_covariance`, ~2 h once on a CPU; saved in `rome_cov/`) | `python "Python code/prepare_data.py" <the .parquet file> "GPT-2 small model" Datasets/wt2` |
| OpenWebText: `openwebtext.tar.xz` from the [Zenodo release](https://zenodo.org/records/3834942) (~12 GB) | tab 16: training the SAEs, and the top contexts printed by `run_sae` | `python "Python code/prepare_openwebtext.py" openwebtext.tar.xz "GPT-2 small model" Datasets/owt` (unpacks ~13 GB, writes ~17 GB of tokens) |

Both write GPT-2 token IDs as `uint16` to `train.bin` and `val.bin` (the nanoGPT format).

## Folder layout

The code finds everything relative to the repository folder (`project_root` in MATLAB; set the environment
variable `LLM_PROJECT_ROOT` to keep models and data elsewhere):

```
<repository>/
├── docs/                  the website
├── Matlab code/
├── Python code/
├── GPT-2 small model/     Hugging Face files + gpt2_full_weights.mat (made on first use)
├── GPT-2 XL model/        (and medium, large, Pythia 160M: optional)
├── Datasets/
│   ├── wt2/               train.bin, val.bin   (tab 15)
│   └── owt/               train.bin, val.bin   (tab 16)
├── runs/                  trained SAEs: runs/sae_l5_lam1, sae_l5_lam5, sae_l5_lam20 (tab 16)
├── rome_cov/              ROME's covariance, computed once (tab 15)
└── Activations/           saved activations (run_model with saveActivations = true)
```

The model, data and output folders are large and are listed in `.gitignore`.

## Running Lesson 9

In MATLAB, go to `Matlab code/` (or add it to the path) and run a script. Each script starts with a
settings block (prompt, model, layer, ...), runs in numbered steps that match the lesson, and prints and
plots its results. For example:

```matlab
cd 'Matlab code'
run_model          % tab 2: GPT-2 small on "The Eiffel Tower is in the city of", its next-token prediction
```

Most scripts run in seconds to minutes on a CPU. The exceptions are tab 15 (ROME on GPT-2 XL: ~2 h once
for the covariance, then ~40 min per edit on a CPU) and tab 16 (SAE training: hours on a GPU).

| Tab | Method | Scripts |
|---|---|---|
| 2 | Model output | `run_model`, `logit_lens_fromfile` |
| 3 | Logit lens | `run_model` (step 5), `logit_lens` |
| 4 | Direct logit attribution | `run_model` (step 6), `direct_logit_attrib` |
| 5 | Cosine similarity | `cosine_sim`, `run_cosine_tokenembed`, `run_cosine_resid` |
| 6 | RSA | `run_RSA` |
| 7 | PCA | `run_PCA` |
| 8 | Difference in means | `run_DIM` |
| 9 | Linear probe | `run_probe` |
| 10 | Induction heads | `run_model`, `run_induction`, `run_attention_head` |
| 11 | Ablation | `ablate`, `run_ablation` |
| 12 | Activation patching | `run_patching`, `patch_activations` |
| 13 | IOI circuit | `run_ioi` |
| 14 | Steering | `run_steering` |
| 15 | Model editing (ROME) | `run_rome`, `rome_covariance` |
| 16 | Sparse autoencoders | `Python code/train_sae.py`, then `run_sae` |

Shared functions: `model_forward` (the forward pass, with hooks to read or edit any activation),
`model_generate`, `model_tokenize` / `model_token` / `model_decode` (tokenizer), `model_weights_file`,
`project_root`, `python_command`, and the sentence sets `prompts_*.m`. Layers and heads are numbered from 0,
as in papers and Python (head 9.6 = layer 9, head 6).

### Tab 16: training the sparse autoencoders

From the repository folder, with the GPT-2 small weights exported and `Datasets/owt` prepared:

```bash
python "Python code/train_sae.py" "Python code/sae_l5_lam5.json"
```

The JSON file lists the settings that differ from the defaults at the top of `train_sae.py` (here the
output folder and λ = 5; `sae_l5_lam1.json` and `sae_l5_lam20.json` are the other two SAEs). Each run writes
`sae.mat` (the trained SAE) and `eval.mat` / `eval.json` (its evaluation) to `runs/sae_l5_lam5`, which
`run_sae` then loads.

## Notes

- `model_checkpoints.m` also lists training checkpoints of other models (Stanford CRFM GPT-2 runs, Pythia
  checkpoints and our own training runs). They are not part of Lesson 9 and are only read when a script is
  given a `checkpoint`; Lesson 9 uses the final, published weights throughout.
- Prompts are passed to Python through a temporary UTF-8 file, so any characters (accents, Greek) survive
  the command line.

## License

Everything in this repository, the lessons and figures in `docs/` as well as the code, is released under the
MIT License (see `LICENSE`). The models and datasets used here have their own licenses: GPT-2 (MIT),
Pythia (Apache 2.0), OpenWebText and WikiText-2 (see their pages).

## For AI agents

To reproduce a Lesson 9 result:

1. Install MATLAB with the Statistics and Machine Learning and Deep Learning Toolboxes, Python 3, and the
   Python packages above.
2. Download `openai-community/gpt2` (the five files listed under Models) into `<repository>/GPT-2 small model/`.
3. If Python is not reachable as `py` (Windows) or `python3`, set `LLM_PYTHON` to its path.
4. In MATLAB: `cd '<repository>/Matlab code'`, then run the script named for the tab in the table above. The
   first run exports the weights (a few minutes); check that `run_model` predicts " London" (8.1%) ahead of
   " Paris" (6.9%) for "The Eiffel Tower is in the city of".
5. Tabs 12 and 15 (and the model-size comparison of tab 9) also need `GPT-2 XL model/`; tab 15 needs `Datasets/wt2/`; tab 16 needs `Datasets/owt/`
   and a CUDA GPU for training.
