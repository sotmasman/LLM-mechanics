% run_sae.m
% A sparse autoencoder (SAE) on GPT-2 small's residual stream: what its features are, and what they do (Lesson 9, tab 16).
% The SAE is trained in PyTorch by Python code\train_sae.py (Anthropic's 2024 recipe, Templeton et al. 2024) on the stream leaving
% layer L. It rewrites each token's stream x [768 × 1] as a few active features plus an error:
%   f = max(0, W_enc (c x) + b_enc)       [m × 1]  feature activations (most are 0)
%   x̂ = (W_dec f + b_dec) / c             [768 × 1] the reconstruction; x = x̂ + error
% c is one constant that scales x to mean length sqrt(768); W_dec's columns (one per feature) have length 1.
%   1. settings and loading: GPT-2 small, the SAE, its evaluation (firing counts, top activations)
%   2. the prompt through the SAE: active features per token (L0, the largest), variance explained
%   3. splice test: run the model with the stream replaced by x̂ (and zero-ablated); does it still say " Paris"?
%   4. one feature: how often it fires, its top contexts in the training text, the tokens it promotes and suppresses
%   5. all features: density histogram, dead features
%   6. compare with tab 8's sentiment direction (difference in means, also steered with in tab 14): which features line up with it?
%   7. feature steering: clamp one feature to a value at every position and generate
%   8. plot: a. the feature on the prompt, b. feature density, c. promoted and suppressed tokens
%
% Position 0 (<|endoftext|>, the attention sink) was left out when the SAE was trained, so it is skipped everywhere:
% its features are not meaningful, and splicing and steering leave it unchanged.
% Layers, positions, token IDs and feature IDs are 0-based in printouts; arrays are 1-based, hence the + 1.
%
% Needs: model_weights_file, model_tokenize, model_forward, model_generate, model_token, model_decode, prompts_sentiment;
%        sae.mat and eval.mat from train_sae.py, and the train.bin it was trained on (for the top contexts).

%% Settings
% train_sae.py's outputs (README.md layout; point these elsewhere if yours are in other folders)
run_dir    = fullfile(project_root(), 'runs', 'sae_l5_lam5');   % sae_l5_lam1 / _lam5 / _lam20: sparsity penalty 1, 5, 20
sae_file   = fullfile(run_dir, 'sae.mat');
eval_file  = fullfile(run_dir, 'eval.mat');
train_file = fullfile(project_root(), 'Datasets', 'owt', 'train.bin');   % the text it was trained on (uint16 token IDs)

prompt     = 'The capital of France is the city of';
target     = ' Paris';   rival = ' London';   % next-token metrics: log rank of target, logit difference target - rival
feature    = [];          % feature to inspect (0-based ID), or [] = the most active one at the prompt's last token
nTop       = 10;          % top contexts to print (at most 20)

steer_prompt = 'I went for a walk and saw';   % feature steering (step 7)
clamps     = [0 0.25 0.5 1];  % clamp values, in multiples of the feature's maximum activation (0 = feature turned off;
                              % at layer 5 of GPT-2 small, 2 x max or more turns the text into repetition)
nNew       = 20;          % greedy tokens per continuation

%****************************************************

%% Step 1 — Load GPT-2 small, the SAE and its evaluation
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file('gpt2-small'));
S = load(sae_file);                          % W_enc [m × 768], b_enc [m × 1], W_dec [768 × m], b_dec [768 × 1], c, layer
E = load(eval_file);                         % count, mean_act [1 × m], top_val, top_offset [20 × m], eval_tokens
L = double(S.layer);  m = size(S.W_enc, 1);
fprintf('SAE: %d features on the stream leaving layer %d (c = %.3f)\n', m, L, S.c);
% The SAE as two functions of a stream v [768 × n]:
fOf   = @(v) max(0, S.W_enc * (S.c * v) + S.b_enc);           % [m × n]    feature activations
hatOf = @(v) (S.W_dec * fOf(v) + S.b_dec) / S.c;              % [768 × n]  reconstruction x̂
tok = @(s) subsref(model_tokenize(s), struct('type', '()', 'subs', {{2}}));   % a one-token word's ID
tId = tok(target);  rId = tok(rival);

%% Step 2 — The prompt through the SAE
ids = model_tokenize(prompt);                               % [1 × n]  starting with <|endoftext|>
clean = model_forward(ids, W);
X  = clean.resid_post(:, 2:end, L + 1);                     % [768 × n-1]  stream leaving layer L, positions 1 on
F  = fOf(X);                                                % [m × n-1]
Xh = hatOf(X);                                              % [768 × n-1]
% Variance explained: 1 - (squared error) / (squared distance from the mean token), as in train_sae.py's eval
ve = 1 - sum((X - Xh).^2, 'all') / sum((X - mean(X, 2)).^2, 'all');
fprintf('\nPrompt "%s": variance explained by xhat %.3f\n', prompt, ve);
fprintf('pos  token          L0   largest features (ID: activation)\n');
for p = 1:size(F, 2)
    [a, j] = sort(F(:, p), 'descend');
    fprintf('%3d  %-13s %4d  %s\n', p, ['"' char(model_token(ids(p + 1))) '"'], nnz(F(:, p)), ...
            strjoin(compose('%d: %.2f', j(1:3) - 1, a(1:3)), ',  '));
end
if isempty(feature), [~, j] = max(F(:, end));  feature = j - 1; end
i = feature + 1;                                            % its array index

%% Step 3 — Splice test: the model with x̂ (or zeros) in place of the stream leaving layer L
% The edit (local function replace_at, below) replaces the stream leaving layer L at positions 1 on by fn(stream);
% the later layers then run on it. Clean, SAE and zero runs, scored at the last position.
oSae  = model_forward(ids, W, @(name, l, v) replace_at(name, l, v, L, hatOf));
oZero = model_forward(ids, W, @(name, l, v) replace_at(name, l, v, L, @(v) zeros(size(v), 'like', v)));
fprintf('Check: stream in the SAE run - xhat, max difference %.1e (should be ~0)\n', max(abs(oSae.resid_post(:, 2:end, L + 1) - Xh), [], 'all'));
runs = {clean, oSae, oZero};  runNames = {'clean', 'SAE', 'zero'};
fprintf('\n%-26s log rank of "%s"   logit diff "%s" - "%s"   p("%s")\n', 'Next token after the prompt', target, target, rival, target);
for k = 1:3
    z = double(runs{k}.logits(:, end));                     % [vocab × 1]  logits for the next token
    r = sum(z > z(tId + 1)) + 1;                            % rank of target (1 = top prediction)
    [~, top] = max(z);
    fprintf('%-6s (top: %-10s)    %6.2f (rank %5d)     %+8.2f                %.3f\n', runNames{k}, ['"' char(model_token(top - 1)) '"'], ...
            log(r), r, z(tId + 1) - z(rId + 1), runs{k}.probs(tId + 1));
end

%% Step 4 — One feature
fprintf('\nFeature %d: fires on %.4f%% of %d tokens, mean activation when firing %.2f, max %.2f\n', feature, ...
        100 * E.count(i) / E.eval_tokens, E.eval_tokens, E.mean_act(i), E.top_val(1, i));
% Top contexts: each top token's offset in train.bin; read 12 tokens before and 3 after (2 bytes per token).
% The model saw only the tokens since the start of its training window, so the earliest ones may not have been in view.
fid = fopen(train_file, 'r');
for k = 1:min(nTop, size(E.top_offset, 1))
    off = E.top_offset(k, i);
    if off < 0, break; end                                  % -1: fewer firings than top slots
    first = max(0, off - 12);
    fseek(fid, 2 * first, 'bof');
    w = fread(fid, off - first + 4, 'uint16')';             % [1 × ≤16]  token IDs
    t = off - first + 1;                                    % the top token in w
    s = model_decode(w(1:t - 1)) + "[[" + model_decode(w(t)) + "]]" + model_decode(w(t + 1:end));
    fprintf('  %6.2f  %s\n', E.top_val(k, i), replace(s, newline, '\n'));
end
fclose(fid);
% The tokens it promotes and suppresses: the direct effect of its decoder direction on the logits, W_U W_dec(:, i)
% (as in the logit lens; the final LayerNorm and every later layer are ignored).
eff = double(W.W_U) * double(S.W_dec(:, i));                % [vocab × 1]
[es, ej] = sort(eff, 'descend');
up = ej(1:10);  down = ej(end:-1:end - 9);                   % array indices of the top 10 and bottom 10
fprintf('Promotes:   %s\n', strjoin(compose('"%s"', model_token(up' - 1)), ' '));
fprintf('Suppresses: %s\n', strjoin(compose('"%s"', model_token(down' - 1)), ' '));

%% Step 5 — All features: how often they fire
freq = double(E.count) / E.eval_tokens;                     % [1 × m]  firing frequency
fprintf('\nDead features (never fired in %d tokens): %.2f%%\n', E.eval_tokens, 100 * mean(freq == 0));

%% Step 6 — Compare with the sentiment direction of tab 8 (difference in means, built as in run_steering)
[sent, labels] = prompts_sentiment();
sids = model_tokenize(sent);                                % {1 × 20}  token IDs of each sentence
Xs = zeros(768, numel(sent));  Fs = zeros(m, numel(sent));  % [768 × 20], [m × 20]
for k = 1:numel(sent)
    o = model_forward(sids{k}, W);
    R = double(o.resid_post(:, 2:end, L + 1));              % [768 × tokens]  stream leaving layer L, without <|endoftext|>
    Xs(:, k) = mean(R, 2);                                  % readout 'mean'
    Fs(:, k) = mean(fOf(R), 2);                             % mean activation of each feature over the sentence
end
isPos = strcmp(labels, 'positive');
dim = mean(Xs(:, isPos), 2) - mean(Xs(:, ~isPos), 2);       % [768 × 1]  the DIM sentiment direction
cs = double(S.W_dec)' * dim / norm(dim);                  % [m × 1]  cosine of each decoder column with it (columns have length 1)
dF = mean(Fs(:, isPos), 2) - mean(Fs(:, ~isPos), 2);        % [m × 1]  mean activation, positive - negative sentences
[~, a] = sort(cs, 'descend');  [~, b] = sort(dF, 'descend');
fprintf('\nSentiment (DIM direction at layer %d):\n', L);
fprintf('  largest cosine with the direction:    %s\n', strjoin(compose('%d (%+.2f)', a(1:5) - 1, cs(a(1:5))), ', '));
fprintf('  most negative cosine:                 %s\n', strjoin(compose('%d (%+.2f)', a(end:-1:end-4) - 1, cs(a(end:-1:end-4))), ', '));
fprintf('  more active on positive sentences:    %s\n', strjoin(compose('%d (%+.2f)', b(1:5) - 1, dF(b(1:5))), ', '));
fprintf('  more active on negative sentences:    %s\n', strjoin(compose('%d (%+.2f)', b(end:-1:end-4) - 1, dF(b(end:-1:end-4))), ', '));

%% Step 7 — Feature steering: clamp feature i at every position (Templeton et al. 2024)
% Inside the SAE's decomposition x = W_dec f / c + (the rest), set f_i to the value v and keep everything else,
% including the error term:  x' = x + (v - f_i(x)) W_dec(:, i) / c.  v = 0 turns the feature off.
% The edit is applied at every position from 1 on, at every generated token too.
sp = model_tokenize(steer_prompt);
fiOf  = @(v) max(0, S.W_enc(i, :) * (S.c * v) + S.b_enc(i));        % [1 × n]  feature i alone
clamp = @(v, val) v + (val - fiOf(v)) .* S.W_dec(:, i) / S.c;      % [768 × n]
g = model_generate(sp, W, nNew);
fprintf('\nSteering with feature %d, "%s" ...\n  unsteered      |%s\n', feature, steer_prompt, replace(g.text, newline, ' '));
for v = clamps
    val = single(v * E.top_val(1, i));
    g = model_generate(sp, W, nNew, Edit=@(name, l, x) replace_at(name, l, x, L, @(x) clamp(x, val)));
    fprintf('  %2g x max (%5.1f) |%s\n', v, val, replace(g.text, newline, ' '));
end

%% Step 8 — Plot
close(findall(groot, 'Type', 'figure', 'Tag', 'run_sae'));   % this script's figure from an earlier run
figure('Color', 'w', 'Position', [60 200 1400 420], 'Tag', 'run_sae');
tiledlayout(1, 3, 'TileSpacing', 'compact');
nexttile;                                                   % a. the feature on the prompt (step 2)
bar(1:size(F, 2), F(i, :));  grid on
xticks(1:size(F, 2));  xticklabels(model_token(ids(2:end)));  set(gca, 'TickLabelInterpreter', 'none');
ylabel('activation');  title(sprintf('a. feature %d on the prompt, layer %d', feature, L));
nexttile;                                                   % b. feature density (step 5)
histogram(log10(freq(freq > 0)), 40);  grid on
xlabel('log_{10} firing frequency');  ylabel('features');
title(sprintf('b. feature density (%.1f%% dead)', 100 * mean(freq == 0)));
nexttile;                                                   % c. promoted and suppressed tokens (step 4)
barh([es(end:-1:end - 9); es(10:-1:1)]);  grid on        % most suppressed at the bottom, most promoted at the top
yticks(1:20);  yticklabels(compose('"%s"', model_token([ej(end:-1:end - 9); ej(10:-1:1)]' - 1)));  set(gca, 'TickLabelInterpreter', 'none');
xlabel('direct effect on the logit (W_U W_{dec})');  title(sprintf('c. tokens feature %d promotes / suppresses', feature));

function v = replace_at(name, l, v, L, fn)
    % model_forward edit: at the stream leaving layer L, replace positions 1 on by fn(those columns); position 0 unchanged
    if strcmp(name, 'resid_post') && l == L
        v = [v(:, 1), fn(v(:, 2:end))];
    end
end

