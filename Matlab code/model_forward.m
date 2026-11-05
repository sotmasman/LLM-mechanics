function out = model_forward(ids, W, edit, start)
% model_forward  Run GPT-2 (any size) or Pythia on a sequence of token IDs, in plain MATLAB.
%   Pythia (GPT-NeoX, W.cfg.arch = 'neox', from export_pythia_weights.py) differs in three
%   places: rotary positions on the first W.cfg.rotary_dim query/key dimensions (W_pos is
%   zero), a parallel residual (the MLP reads the stream ENTERING the layer, not resid_mid;
%   resid_mid = resid_pre + attention is still stored, so attention then MLP sums to the
%   layer's change) and exact GELU. Its <|endoftext|> is token 0, its vocabulary 50304.
%   Llama-style blocks (our Arch1 runs, W.cfg.arch = 'llama', from export_llama_weights.py):
%   RMSNorm (scale only, no centering; the norm weights are folded into the next matrices),
%   rotary positions on all query/key dimensions (W_pos is zero), sequential residual as GPT-2,
%   and a SwiGLU MLP: mlp_hidden = silu(W_gate·h) .* (W_in·h), then W_out. GPT-2 tokenizer.
%   out = model_forward(ids, W)        normal run
%   out = model_forward(ids, W, edit)  run with an intervention
%   out = model_forward(ids, W, edit, start)  start at a later layer (faster patching)
%
% Inputs
%   ids   Token IDs, 0-based (e.g. a.token_ids from activations.mat). Start
%         with <|endoftext|> (50256) as the saved prompts do; nothing is added.
%         At most 1024 tokens.
%   W     All weights:  W = load(model_weights_file('gpt2-small'));
%         Load it once per session. To experiment on the weights, edit a copy
%         of W and pass the copy in.
%   edit  Optional function  v = edit(name, L, v)  called at each edit point
%         below. It gets the current values and returns them, changed or not.
%         L is the layer, 0-based (0-11) as in papers and Python.
%   start Optional struct('layer', L0, 'resid', x): skip the embeddings and
%         layers 0 .. L0-1, and begin at layer L0 with x [768 × n], the residual
%         stream entering L0 from an earlier run of the same ids (its
%         out.resid_pre(:, :, L0 + 1)). Edits from layer L0 on still apply. Gives
%         identical results whenever the layers before L0 would be unchanged,
%         e.g. when every edit is at layer L0 or later. Skipped layers stay 0 in out.
%
% Edit points, in order within each layer (n = number of tokens):
%   'resid_pre'     768 × n          residual stream entering the layer
%   'attn_pattern'  n × n × 12       attention weights, query × key × head
%   'attn_by_head'  768 × 12 × n     each head's output (before b_O)
%   'resid_mid'     768 × n          after attention is added
%   'mlp_hidden'    3072 × n         MLP neurons after GELU
%   'mlp_out'       768 × n          MLP output
%   'resid_post'    768 × n          residual stream leaving the layer
%
% Output struct (single precision; edits are included in what is stored)
%   out.ids           1 × n
%   out.logits        50257 × n          logits at every position
%   out.probs         50257 × 1          next-token probabilities (last position)
%   out.resid_pre, out.resid_mid, out.resid_post, out.mlp_out   768 × n × 12
%   out.attn_by_head  768 × 12 × n × 12  (d_model × head × position × layer)
%   out.attn_pattern  n × n × 12 × 12    (query × key × head × layer)
%   out.mlp_hidden    3072 × n × 12
% Same names and layouts as activations.mat. Arrays are 1-indexed, so
% layer L, head h is (..., h + 1, ..., L + 1).
%
% Example
%   W = load(model_weights_file('gpt2-small'));
%   a = load(fullfile(project_root(), 'Activations', 'capital_of_france', 'activations.mat'));
%   out = model_forward(a.token_ids, W);
%   out.probs(6342 + 1)                          % p(" Paris") = 0.0972
%   abl = model_forward(a.token_ids, W, @silence_head_9_9);
%
%   function v = silence_head_9_9(name, L, v)    % (at the end of your script)
%       if strcmp(name, 'attn_by_head') && L == 9
%           v(:, 9 + 1, :) = 0;                  % head 9.9 contributes nothing
%       end
%   end
%
% Weights come from Python code\export_gpt2_weights.py (TransformerLens with
% the layer norms folded into the weights), so every layer norm here only
% centers and scales.
%
% See also model_generate, model_token.

if nargin < 3
    edit = [];
end
if nargin < 4 || isempty(start)
    start = struct('layer', 0, 'resid', []);
end
ids = double(ids(:)');                 % [1 × n]
n = numel(ids);
[d_head, d_model, n_heads, n_layers] = size(W.W_Q);
d_mlp = size(W.W_in, 1);
if n > size(W.W_pos, 2)
    error('model_forward:tooLong', 'At most %d tokens; got %d.', size(W.W_pos, 2), n);
end
causal = triu(true(n), 1);             % [n × n] true where a query would look ahead
% Stanford CRFM checkpoints also divide each layer's scores by (L + 1) (scale_attn_by_inverse_layer_idx)
byLayer = isfield(W, 'cfg') && isfield(W.cfg, 'scale_attn_by_inverse_layer_idx') && W.cfg.scale_attn_by_inverse_layer_idx;
% Pythia (GPT-NeoX): rotary positions, parallel residual, exact GELU
neox = isfield(W, 'cfg') && isfield(W.cfg, 'arch') && strcmp(W.cfg.arch, 'neox');
% Llama-style blocks: RMSNorm, rotary positions on every head dimension, SwiGLU MLP
llama = isfield(W, 'cfg') && isfield(W.cfg, 'arch') && strcmp(W.cfg.arch, 'llama');
nrm = @layer_norm;
if llama, nrm = @(x) rms_norm(x, double(W.cfg.rms_eps)); end
if neox || llama
    rd = double(W.cfg.rotary_dim);                                 % rotated dimensions per head (16 of 64)
    ang = (1 ./ double(W.cfg.rotary_base) .^ ((0:2:rd - 1)' / rd)) * (0:n - 1);   % [rd/2 × n]  angle per frequency and position
    rotC = single([cos(ang); cos(ang)]);  rotS = single([sin(ang); sin(ang)]);     % [rd × n]
end

% Shapes (for reference; n = tokens, L = layer, hd = head, both 0-based):
%   x (residual stream)       — [768 × n]          (d_model × position), updated layer by layer
%   W.W_E                     — [768 × 50257]      (d_model × n_vocab)
%   W.W_pos                   — [768 × 1024]       (d_model × max positions)
%   W.W_Q, W.W_K, W.W_V       — [64 × 768 × 12 × 12]   (d_head × d_model × head × layer)
%   W.W_O                     — [768 × 64 × 12 × 12]   (d_model × d_head × head × layer)
%   W.W_in                    — [3072 × 768 × 12]  (d_mlp × d_model × layer)
%   W.W_out                   — [768 × 3072 × 12]  (d_model × d_mlp × layer)
%   W.W_U                     — [50257 × 768]      (n_vocab × d_model)
%   biases                    — one column per layer (and head), added to the matching output

out.ids          = ids;
rec = single(W.W_E(1));                        % single, or a single gpuArray when the weights are on the GPU
out.resid_pre    = zeros(d_model, n, n_layers, 'like', rec);
out.resid_mid    = zeros(d_model, n, n_layers, 'like', rec);
out.resid_post   = zeros(d_model, n, n_layers, 'like', rec);
out.mlp_out      = zeros(d_model, n, n_layers, 'like', rec);
out.attn_by_head = zeros(d_model, n_heads, n, n_layers, 'like', rec);
out.attn_pattern = zeros(n, n, n_heads, n_layers, 'like', rec);
out.mlp_hidden   = zeros(d_mlp, n, n_layers, 'like', rec);

%% Step 1 — Embeddings  (token + position)
if start.layer == 0
    x = W.W_E(:, ids + 1) + W.W_pos(:, 1:n);      % [768 × n]  one column per token
else
    x = start.resid;                              % [768 × n]  residual stream entering start.layer, from an earlier run
end

%% Steps 2-3 for each layer L = 0..11
for L = start.layer:n_layers - 1
    x = apply_edit(edit, 'resid_pre', L, x);
    out.resid_pre(:, :, L + 1) = plain(x);               % [768 × n]  residual stream entering layer L
    xPre = x;                                     % Pythia's MLP reads this (parallel residual)

    %% Step 2 — Attention  (each token gathers information from earlier tokens)
    % 2a. LayerNorm  (manual, since LN is folded into W_Q/W_K/W_V; RMSNorm for Llama)
    h = nrm(x);                                    % [768 × n]
    like = single(x(1));                           % single, or a single dlarray when x is one (keeps autodiff traces)
    pattern = zeros(n, n, n_heads, 'like', like);  % [n × n × 12]  query × key × head
    V = zeros(d_head, n, n_heads, 'like', like);   % [64 × n × 12]
    for hd = 0:n_heads - 1
        % 2b. Queries, keys, values for head hd
        q = W.W_Q(:, :, hd + 1, L + 1) * h + W.b_Q(:, hd + 1, L + 1);   % [64 × n]  what each token looks for
        k = W.W_K(:, :, hd + 1, L + 1) * h + W.b_K(:, hd + 1, L + 1);   % [64 × n]  what each token offers
        if neox || llama, q = rotary(q, rotC, rotS, rd);  k = rotary(k, rotC, rotS, rd); end   % Pythia, Llama: rotate by position
        V(:, :, hd + 1) = W.W_V(:, :, hd + 1, L + 1) * h + W.b_V(:, hd + 1, L + 1);   % [64 × n]  what it passes on
        % 2c. Attention pattern: who looks at whom
        scores = (q' * k) / sqrt(d_head) / (1 + byLayer * L);   % [n × n]  query × key, scaled by sqrt(64) = 8 (and by L + 1 if byLayer)
        scores(causal) = -Inf;                     % no looking at later tokens
        p = exp(scores - max(scores, [], 2));      % softmax over keys (each row), shifted for stability
        pattern(:, :, hd + 1) = p ./ sum(p, 2);    % [n × n]  each row sums to 1
    end
    pattern = apply_edit(edit, 'attn_pattern', L, pattern);
    out.attn_pattern(:, :, :, L + 1) = plain(pattern);

    % 2d. Each head's output: weighted values, projected back to d_model
    heads = zeros(d_model, n_heads, n, 'like', like);  % [768 × 12 × n]  d_model × head × position
    for hd = 0:n_heads - 1
        z = V(:, :, hd + 1) * pattern(:, :, hd + 1)';                    % [64 × n]  weighted values
        heads(:, hd + 1, :) = reshape(W.W_O(:, :, hd + 1, L + 1) * z, d_model, 1, n);   % [768 × 1 × n]
    end
    heads = apply_edit(edit, 'attn_by_head', L, heads);
    out.attn_by_head(:, :, :, L + 1) = plain(heads);

    % 2e. Add all heads (+ bias) to the residual stream
    x = x + reshape(sum(heads, 2), d_model, n) + W.b_O(:, L + 1);        % [768 × n]
    x = apply_edit(edit, 'resid_mid', L, x);
    out.resid_mid(:, :, L + 1) = plain(x);

    %% Step 3 — MLP  (each token processed on its own)
    % 3a. LayerNorm, up-projection to 3072 neurons, GELU
    if neox                                        % Pythia: parallel residual (reads the stream entering the layer), exact GELU
        m = gelu_exact(W.W_in(:, :, L + 1) * layer_norm(xPre) + W.b_in(:, L + 1));
    elseif llama                                   % Llama: SwiGLU, silu(gate) times up  [2048 × n]
        h = nrm(x);
        g = W.W_gate(:, :, L + 1) * h;
        m = (g ./ (1 + exp(-g))) .* (W.W_in(:, :, L + 1) * h);
    else
        m = gelu_new(W.W_in(:, :, L + 1) * layer_norm(x) + W.b_in(:, L + 1));   % [3072 × n]
    end
    m = apply_edit(edit, 'mlp_hidden', L, m);
    out.mlp_hidden(:, :, L + 1) = plain(m);

    % 3b. Down-projection back to d_model
    mlp = W.W_out(:, :, L + 1) * m + W.b_out(:, L + 1);                      % [768 × n]
    mlp = apply_edit(edit, 'mlp_out', L, mlp);
    out.mlp_out(:, :, L + 1) = plain(mlp);

    % 3c. Add to the residual stream
    x = x + mlp;                                                            % [768 × n]
    x = apply_edit(edit, 'resid_post', L, x);
    out.resid_post(:, :, L + 1) = plain(x);              % [768 × n]  residual stream leaving layer L
end

%% Step 4 — Unembed  (same as logit_lens at L = 11)
% LayerNorm (manual, since LN is folded into W_U/b_U), then logits
out.logits = W.W_U * nrm(x) + W.b_U';             % [50257 × n]  one score per vocabulary token, per position

%% Step 5 — Softmax at the last position  (shift for numerical stability)
last = out.logits(:, end) - max(out.logits(:, end));
out.probs = exp(last) / sum(exp(last));           % [50257 × 1]  p(next token)
end

% =========================================================================
function x = layer_norm(x)
% Center and scale each column (the weights are folded into the next matrix).
x = x - mean(x, 1);
x = x ./ sqrt(mean(x .^ 2, 1) + 1e-5);
end

function x = rms_norm(x, eps)
% Llama's RMSNorm: scale each column by its root mean square, no centering (the weight is folded into the next matrix).
x = x ./ sqrt(mean(x .^ 2, 1) + eps);
end

function y = gelu_exact(x)
% Pythia's GELU (exact, with erf).
y = 0.5 * x .* (1 + erf(x / sqrt(2)));
end

function v = rotary(v, C, S, rd)
% Rotary position embedding (GPT-NeoX): rotate the first rd dimensions of each
% column (= position) by position-dependent angles; the rest pass unchanged.
h = rd / 2;
r = [-v(h + 1:rd, :); v(1:h, :)];                  % rotate_half
v(1:rd, :) = v(1:rd, :) .* C + r .* S;
end

function y = gelu_new(x)
% GPT-2's GELU (tanh approximation).
y = 0.5 * x .* (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x .^ 3)));
end

function v = apply_edit(edit, name, L, v)
if isempty(edit)
    return
end
before = size(v);
v = edit(name, L, v);
if ~isequal(size(v), before)
    error('model_forward:editSize', 'The edit changed the size of ''%s'' at layer %d.', name, L);
end
end

function v = plain(v)
% The values of v without an autodiff trace, for the out.* records (a traced dlarray cannot be stored in a gpuArray).
if isa(v, 'dlarray'), v = extractdata(v); end
end
