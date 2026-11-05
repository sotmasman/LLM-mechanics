% logit_lens_fromfile.m
% Predicts GPT-2 small's next token from the saved residual stream of one
% prompt, and prints the 5 most likely tokens with their probabilities.
%
% What it does:
%   Takes the output of layer L (the residual stream after it) at the last token, and runs it
%   through the model's final steps by hand:
%     1. LayerNorm   - center and scale (its weights are already folded
%                      into W_U and b_U, so no learned scale or bias here)
%     2. Logits      - W_U * resid + b_U, one score per vocabulary token
%     3. Softmax     - turn the logits into probabilities
%     4. Top 5       - sort, then look up each token's text with model_token
%   Layers are numbered from 0 (0-11), as in papers and Python. With
%   L = 11 (the last layer) this reproduces GPT-2's actual prediction. A
%   smaller L gives the "logit lens" view: what the model would predict if
%   it stopped after that layer.
%
% Inputs:
%   GPT-2 small model\gpt2_full_weights.mat                            W_U, b_U (only these are loaded)
%   Activations\activations_folder\activations.mat   resid_post, ...
%     (one prompt folder, made by run_model with saveActivations = true,
%      or by Python code\save_activations_mat.py)
%
% Needs model_token.m (in this folder) on the MATLAB path.
%
% Example output (capital_of_france, L = 11):
%   Rank  token_id  token        prob
%      1      6342  " Paris"     9.7185%
%      2     45996  " Marse"     8.9124%

activations_folder = 'The capital of France is the city of';

activations_folder = regexprep(lower(regexprep(strtrim(activations_folder), '[^A-Za-z0-9]+', '_')), '^_+|_+$', '');   % same rule as run_model: "Paris, France" -> "paris_france"

%% Load
w = load(fullfile(project_root(), 'GPT-2 small model', 'gpt2_full_weights.mat'), 'W_U', 'b_U');
a = load(fullfile(project_root(), 'Activations', activations_folder, 'activations.mat'));

% Shapes (for reference):
%   w.W_U   — [50257 × 768]   (n_vocab × d_model)
%   w.b_U   — [1 × 50257]
%   a.resid_post — [768 × 9 × 12]   (d_model × seq × layer)

%% Pick layer and position
L    = 11;          % layer whose output is read, 0-based (0-11); 11 = last layer
last = a.seq_len;   % last token position

resid = a.resid_post(:, last, L + 1);   % [768 × 1]  (+1: MATLAB arrays start at 1)

%% Step 1 — LayerNorm  (manual, since LN is folded into W_U/b_U)
% Full LayerNorm is γ ⊙ n + β, with n = (x - μ)/σ and γ, β the learned scale and shift (768 each).
% Its weights need no separate step because the next step is a matrix multiply:
%   W_U (γ ⊙ n + β) = (W_U · diag(γ)) n + W_U β
% The exported w.W_U is already W_U · diag(γ) (each column times its γ_i) and w.b_U is W_U β,
% so only the centering and scaling (n) are done here.
resid_norm = (resid - mean(resid)) / std(resid, 1);   % std(...,1) = population std, matches PyTorch

%% Step 2 — Logits
logits = w.W_U * resid_norm + w.b_U';   % [50257 × 1]

%% Step 3 — Softmax  (shift for numerical stability)
logits = logits - max(logits);
probs  = exp(logits) / sum(exp(logits));   % [50257 × 1]

%% Step 4 — Top 5
fprintf('\n************************************\nNext-token prediction for prompt:\n\n%s', a.prompt)
[sorted_p, idx] = sort(probs, 'descend');
fprintf('\n____________________________________\nRank  token_id  token        prob\n');
for i = 1:5
    token = model_token(idx(i)-1);   % token text, e.g. " Paris"
    fprintf('%4d  %8d  %-12s %.4f%%\n', i, idx(i)-1, '"' + token + '"', sorted_p(i)*100);
end