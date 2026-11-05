% run_rome.m
% Model editing with ROME: rewrite one fact in the weights of GPT-2 (Lesson 9, tab 15).
% Meng, Bau, Andonian & Belinkov 2022, "Locating and Editing Factual Associations in GPT" (NeurIPS 2022,
% arXiv:2202.05262); Lesson 7, tab 8.
%
% The idea. An MLP can be read as a key-value memory. At the subject's last token (" Tower"), the MLP's hidden
% activation k (after GELU, the input of W_out) is a KEY, and what the MLP writes into the residual stream,
% v = W_out k + b_out, is its VALUE; later layers turn that value into the answer (" Paris"). To change the fact,
% change what W_out returns for this one key, and as little as possible for every other key. The change that
% does this is rank one: dW = (column) × (row), a single outer product added to W_out.
%
%   1. load the weights and the token IDs of the two answers
%   2. the prompts: the edit prompt behind a few short prefixes, and where the subject ends in each
%   3. the key k*: the MLP's hidden activation at the subject's last token, averaged over the prefixes
%   4. the old value v0 = W_out k* + b_out, and what the model answers now
%   5. the key covariance C = mean(k k') over Wikipedia text: which key directions other inputs use
%   6. the essence prompt: what the model says after "The Eiffel Tower is a", to be kept
%   7. the new value: find delta (gradient descent, autodiff through model_forward) so that v0 + delta makes the
%      model answer target_new, without changing the essence prompt's answer and with delta kept small
%   8. the rank-one edit: dW = delta u' / (u' kE), u = C^-1 k*  (kE = the key at the edit prompt: dW kE = delta)
%   9. the edited model: W_out + dW at layer L, every other weight unchanged
%  10. test: the edit prompt and paraphrases (does it work, does it generalize?)
%  11. test: neighbouring facts (is it specific?)
%  12. test: greedy continuations before and after
%  13. print and plot: a. edit prompt and paraphrases, b. neighbours, c. the search for delta
%
% Layers are 0-based (layer 17 = the 18th of GPT-2 XL's 48 layers). Arrays are 1-based, hence the + 1.
% Runtime, GPT-2 XL on a CPU: about 2 hours for C (once per layer, then loaded from cov_dir), then about
% 40 minutes per edit with the 21 generated prefixes (10 with 5 fixed ones). GPT-2 small: a few minutes.
%
% Needs: model_weights_file, model_tokenize, model_forward, model_generate, model_token, rome_covariance;
%        the Deep Learning Toolbox (dlarray, dlgradient, adamupdate).

%% Settings
model        = 'gpt2-xl';      % 'gpt2-xl' (as in the ROME paper) or 'gpt2-small' (fast, for trying it out)
L            = 17;             % the MLP layer to edit (the paper's choice for GPT-2 XL)

% The fact to rewrite
prompt       = 'The Eiffel Tower is located in the city of';
subject      = ' Eiffel Tower';     % as it appears in the prompts (with its leading space)
target_new   = ' Rome';             % the new answer (a single token)
target_true  = ' Paris';            % the old answer (a single token), for comparison
essence      = 'The Eiffel Tower is a';   % what the subject IS should not change: this prompt's answer is kept

% Tests
paraphrases  = {'The Eiffel Tower is in', 'To visit the Eiffel Tower, you have to travel to', ...
                'The Eiffel Tower stands in the centre of', 'Tourists come to see the Eiffel Tower in'};
neighbours   = {'The Louvre is located in the city of', 'Notre-Dame Cathedral is located in the city of', ...
                'The Colosseum is located in the city of', 'Big Ben is located in the city of'};
neighbour_answers = {' Paris', ' Paris', ' Rome', ' London'};
gen_prompts  = {'The Eiffel Tower is located in', 'The Eiffel Tower is in the country of', ...
                'The most famous tower in Rome is the', 'The Louvre is located in'};
nNew         = 15;             % greedy tokens per continuation

% Key averaging and the search for delta (the paper's settings)
prefixes     = 'generate';     % the model writes them, as in the paper (step 2); or a list of fixed prefixes, e.g.
                               % {'', 'Here is a short note. ', 'I read this in a book. ', 'This is well known. ', 'Last summer we travelled. '}
prefix_lengths = [5 10];  nPerLength = 10;   % with 'generate': nPerLength prefixes of each length (tokens)
nSteps       = 20;  lr = 0.5;  % Adam steps and learning rate
kl_weight    = 0.0625;         % weight of the essence penalty
decay        = 0.5;            % weight of the size penalty |delta| / |v0|^2 (as in the paper's code)
clamp        = 4;              % |delta| is kept below clamp * |v0|

% Key covariance
corpus_file  = fullfile(project_root(), 'Datasets', 'wt2', 'train.bin');   % Wikipedia text as GPT-2 token IDs (uint16)
nCovTokens   = 100000;         % tokens used for C
ridge        = 1e-4;           % C + ridge × (mean diagonal of C) × I: lets C be inverted where some key directions never occur
cov_dir      = fullfile(project_root(), 'rome_cov');   % C is saved here and reused

useGPU       = false;          % true: run the model on the GPU (Parallel Computing Toolbox; GPT-2 XL needs ~10 GB)

%****************************************************

%% Step 1 — Weights and the two answers
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file(model));
if useGPU                                       % move every numeric weight array to the GPU (model_forward follows them)
    f = fieldnames(W);
    for j = 1:numel(f), if isnumeric(W.(f{j})), W.(f{j}) = gpuArray(W.(f{j})); end, end
end
% A one-token word's ID: model_tokenize puts <|endoftext|> first, so the word is element 2.
tok = @(s) subsref(model_tokenize(s), struct('type', '()', 'subs', {{2}}));
newId = tok(target_new);  trueId = tok(target_true);    % e.g. " Rome", " Paris" (0-based IDs)

%% Step 2 — The prompts, and where the subject ends in each
% The edit is learned on the edit prompt behind several short prefixes, so that it does not depend on the
% exact text before the subject. In every prompt we need the subject's LAST token (" Tower"): the paper found
% that this is where middle-layer MLPs read the fact (causal tracing, tab 12).
subj = model_tokenize(subject);  subj = subj(2:end);    % the subject's token IDs (" E", "iff", "el", " Tower")
lastOf = @(ids) strfind(ids, subj) + numel(subj) - 1;   % array index of its last token in a prompt
% prefixes = 'generate' (the paper's way): the model itself continues the seed words "The", "Therefore",
% "Because", "I", "You" for 5 or 10 tokens (sampling from its 5 most likely tokens), and each text + ". " is a
% prefix; with the empty prefix, 21 prompts. Generated text covers more varied contexts than a few fixed sentences.
if ischar(prefixes) && strcmp(prefixes, 'generate')
    seeds = {'The', 'Therefore', 'Because', 'I', 'You'};
    prefixes = {''};
    for len = prefix_lengths
        for j = 1:nPerLength
            sw = seeds{mod(j - 1, numel(seeds)) + 1};
            g = model_generate(model_tokenize(sw), W, len, Temperature=1, TopK=5, Seed=100 * len + j);
            prefixes{end + 1} = [sw, regexprep(char(g.text), '\s+', ' '), '. '];   %#ok<SAGROW>
        end
    end
    fprintf('Generated prefixes, e.g. "%s", "%s"\n', prefixes{2}, prefixes{end});
end
P = struct('ids', {}, 'pos', {});                       % one entry per prefixed prompt
for j = 1:numel(prefixes)
    ids = model_tokenize([prefixes{j} prompt]);
    P(j).ids = ids;  P(j).pos = lastOf(ids);
end
fprintf('Subject "%s" ends at token %d of "%s"\n', subject, P(1).pos - 1, prompt);

%% Step 3 — The key k*
% Run each prefixed prompt and read the MLP's hidden activation of layer L at the subject's last token
% (out.mlp_hidden: after GELU, the vector that W_out multiplies). k* is their average.
% We also keep the residual stream entering layer L: the layers before L never change during the edit, so
% step 7's forward passes can start at layer L (model_forward's start input) instead of at the embedding.
K = zeros(size(W.W_in, 1), numel(P));                   % [d_mlp × prompts]  (d_mlp = 6400 for XL)
for j = 1:numel(P)
    o = model_forward(P(j).ids, W);
    K(:, j) = gather(double(o.mlp_hidden(:, P(j).pos, L + 1)));
    P(j).start = struct('layer', L, 'resid', o.resid_pre(:, :, L + 1));
end
kStar = mean(K, 2);                                     % [d_mlp × 1]

%% Step 4 — The edit prompt's own key kE and value v0, and what the model answers now
% k* (step 3) gives the edit its DIRECTION; the edit prompt itself (no prefix) gives the key kE at which the edit
% is made exact (step 8), and the old value v0 = W_out kE + b_out that delta is added to, as in the paper.
Wout = gather(double(W.W_out(:, :, L + 1)));            % [d × d_mlp]  the matrix we will edit
bout = gather(double(W.b_out(:, L + 1)));               % [d × 1]
idsE = model_tokenize(prompt);
o = model_forward(idsE, W);
kE = gather(double(o.mlp_hidden(:, lastOf(idsE), L + 1)));   % [d_mlp × 1]  the key at the edit prompt
v0 = Wout * kE + bout;                                  % [d × 1]  what the MLP writes there now
fprintf('Before: p(%s) = %.3f, p(%s) = %.3f\n', target_true, o.probs(trueId + 1), target_new, o.probs(newId + 1));

%% Step 5 — The key covariance C
% C = mean of k k' over the keys of ordinary text (every position of nCovTokens Wikipedia tokens). Directions
% where C is large are keys that other inputs use often; the edit (step 8) changes W_out least along them.
% Computing C takes a forward pass over the whole corpus, so it is saved and reused.
covFile = fullfile(cov_dir, sprintf('rome_cov_%s_L%d_%dk.mat', model, L, round(nCovTokens / 1000)));
if isfile(covFile)
    load(covFile, 'C');                                 % [d_mlp × d_mlp]
else
    fprintf('Computing the key covariance at layer %d from %d tokens...\n', L, nCovTokens);
    C = rome_covariance(W, L, corpus_file, nCovTokens);
    if ~isfolder(cov_dir), mkdir(cov_dir); end
    save(covFile, 'C', '-v7.3');
end
% Some key directions (units that never fire in the sample) have almost zero variance, and C^-1 would explode
% along them (at GPT-2 XL layer 4 the condition number is ~10^13). A tiny ridge fixes that and leaves
% well-conditioned layers practically unchanged.
C = C + ridge * mean(diag(C)) * eye(size(C, 1));

%% Step 6 — The essence prompt
% Changing where the tower is should not change WHAT it is. We record the model's next-token distribution
% after "The Eiffel Tower is a" now, and step 7 penalizes any change to it (KL divergence).
E.ids = model_tokenize(essence);  E.pos = lastOf(E.ids);
oE = model_forward(E.ids, W);
E.start = struct('layer', L, 'resid', oE.resid_pre(:, :, L + 1));
pE = gather(double(oE.probs));                          % [vocab × 1]  next-token probabilities before the edit

%% Step 7 — The new value: search for delta
% delta (a [d × 1] vector) is added to the MLP's output of layer L at the subject's last token, in every
% prefixed prompt and in the essence prompt; the model then runs on from layer L. The loss has three parts
% (functions nll_loss and kl_loss, below):
%   - the mean negative log probability of target_new after the prefixed prompts  (make the model say it)
%   + kl_weight × KL(after || before) of the essence prompt's next-token distribution  (keep what it is;
%     the edited distribution comes first, as in the paper's Eqn. 4 and code)
%   + decay × |delta| / |v0|^2  (keep the change small; this form, |delta| not squared, is the paper's code)
% dlfeval runs a loss function with delta as a traced dlarray, so dlgradient gives d(loss)/d(delta) through
% all the layers after L. The loss is a sum of one term per prompt, so its gradient is the sum of the terms'
% gradients: we take them one prompt at a time, because autodiff keeps every intermediate result of the
% forward pass it differentiates (several GB per pass in GPT-2 XL). The size penalty's gradient is simply
% decay × delta / (|delta| |v0|^2) (0 while delta = 0). adamupdate takes one Adam step; after it, |delta| is
% clamped below clamp × |v0|.
delta = dlarray(zeros(size(v0), 'single'));  avg = [];  sq = [];   % start at zero; Adam's running averages
hist = zeros(nSteps, 4);                                % per step: p(target_new), NLL, KL, |delta| / |v0|
v0n = norm(v0);
for it = 1:nSteps
    dn = norm(extractdata(delta));
    nll = 0;  g = decay * delta / (max(dn, eps) * v0n^2);   % the size penalty's gradient (0 at delta = 0)
    for j = 1:numel(P)                                  % make the model say target_new, one prompt at a time
        [nj, gj] = dlfeval(@nll_loss, delta, P(j), W, L, newId);
        nll = nll + nj / numel(P);  g = g + gj / numel(P);
    end
    [kl, gk] = dlfeval(@kl_loss, delta, E, W, L, pE);   % keep the essence prompt's answer
    g = g + kl_weight * gk;
    [delta, avg, sq] = adamupdate(delta, g, avg, sq, it, lr);
    if norm(extractdata(delta)) > clamp * v0n
        delta = delta * (clamp * v0n / norm(extractdata(delta)));
    end
    hist(it, :) = [exp(-nll), nll, kl, norm(extractdata(delta)) / v0n];
    fprintf('  step %2d: p(%s) %.3f, KL %.4f, |delta|/|v0| %.2f\n', it, target_new, hist(it, 1), hist(it, 3), hist(it, 4));
end
delta = gather(double(extractdata(delta)));             % [d × 1]  the new value is v* = v0 + delta

%% Step 8 — The rank-one edit
% We want a change dW of W_out that adds delta to what the MLP writes for the subject (so it writes v0 + delta at
% the edit prompt) and changes the output for other keys as little as possible, on average over ordinary text.
% The solution is a single outer product (rank one):
%   dW = delta u' / (u' kE),   u = C^-1 k*
% Its column is delta, the change to the output. Its row u says which key directions it responds to: along the
% subject's average key k*, but away from the directions that other text uses a lot (large C). The scale makes
% it exact at the edit prompt's key kE: dW kE = delta (as in the paper's released code; its equation uses k*).
u = C \ kStar;                                          % [d_mlp × 1]
dW = delta * u' / (u' * kE);                            % [d × d_mlp]
fprintf('Check: |dW kE - delta| / |delta| = %.1e (should be ~0)\n', norm(dW * kE - delta) / norm(delta));
fprintf('Size: |dW| / |W_out| = %.4f (Frobenius norms)\n', norm(dW, 'fro') / norm(Wout, 'fro'));

%% Step 9 — The edited model
We = W;                                                 % a copy of all the weights ...
We.W_out(:, :, L + 1) = single(Wout + dW);              % ... with only W_out of layer L changed

%% Step 10 — Test: the edit prompt and its paraphrases
% p(next token) before (W) and after (We). The paraphrases were not used in the edit: they test whether the
% new fact generalizes to other wordings.
pr = @(Wx, s, t) feval(@(o) gather(double(o.probs(t + 1))), model_forward(model_tokenize(s), Wx));   % p(token t) next
tests = [{prompt}, paraphrases];
pNew = zeros(numel(tests), 2);  pTrue = zeros(numel(tests), 2);   % columns: before, after
for j = 1:numel(tests)
    pNew(j, :) = [pr(W, tests{j}, newId), pr(We, tests{j}, newId)];
    pTrue(j, :) = [pr(W, tests{j}, trueId), pr(We, tests{j}, trueId)];
end

%% Step 11 — Test: neighbouring facts
% Other landmarks should keep their own answers: if they also moved to target_new, the edit would not be specific.
pNb = zeros(numel(neighbours), 2);                      % p(each neighbour's own answer), before and after
for j = 1:numel(neighbours)
    a = tok(neighbour_answers{j});
    pNb(j, :) = [pr(W, neighbours{j}, a), pr(We, neighbours{j}, a)];
end

%% Step 12 — Test: greedy continuations
% The same prompts continued for nNew tokens by the original and the edited model (greedy: always the most
% likely token). These include harder cases: the fact one step removed (the country) and the reverse direction.
gen = strings(numel(gen_prompts), 2);                   % columns: before, after
for j = 1:numel(gen_prompts)
    ids = model_tokenize(gen_prompts{j});
    g0 = model_generate(ids, W, nNew);  g1 = model_generate(ids, We, nNew);
    gen(j, :) = [string(g0.text), string(g1.text)];
end

%% Step 13 — Results: print, then plot
fprintf('\n%-50s  p(%s) before -> after   p(%s) before -> after\n', 'Prompt', target_new, target_true);
for j = 1:numel(tests), fprintf('%-50s  %.3f -> %.3f            %.3f -> %.3f\n', tests{j}, pNew(j, :), pTrue(j, :)); end
fprintf('\n%-50s  p(own answer) before -> after\n', 'Neighbour');
for j = 1:numel(neighbours), fprintf('%-50s  %s: %.3f -> %.3f\n', neighbours{j}, neighbour_answers{j}, pNb(j, :)); end
fprintf('\nGreedy continuations (before | after):\n');
for j = 1:numel(gen_prompts), fprintf('%s ...\n   before: %s\n   after:  %s\n', gen_prompts{j}, replace(gen(j, 1), newline, ' '), replace(gen(j, 2), newline, ' ')); end

close(findall(groot, 'Type', 'figure', 'Tag', 'run_rome'));   % this script's figure from an earlier run
figure('Color', 'w', 'Position', [60 200 1400 420], 'Tag', 'run_rome');
tiledlayout(1, 3, 'TileSpacing', 'compact');
nexttile;                                               % a. the edit prompt and its paraphrases (steps 10)
bar([pNew, pTrue]);  ylim([0 1]);  grid on;
xticks(1:numel(tests));  xticklabels([{'edit prompt'}, compose('paraphrase %d', 1:numel(paraphrases))]);
legend({[target_new ' before'], [target_new ' after'], [target_true ' before'], [target_true ' after']}, 'Location', 'northwest');
ylabel('next-token probability');  title(sprintf('a. the edit, layer %d', L));
nexttile;                                               % b. neighbouring facts (step 11): should not change
bar(pNb);  ylim([0 1]);  grid on;
xticks(1:numel(neighbours));  xticklabels(extractBefore(string(neighbours), ' is'));
legend({'before', 'after'}, 'Location', 'northeast');  ylabel('p(own answer)');  title('b. neighbours (should not change)');
nexttile;                                               % c. the search for delta (step 7)
yyaxis left;  plot(1:nSteps, hist(:, 1), '-o', 'LineWidth', 1.5);  ylabel(sprintf('p(%s), geometric mean over prefixes', target_new));  ylim([0 1]);
yyaxis right; plot(1:nSteps, hist(:, 3), '-s', 'LineWidth', 1.5);  ylabel('KL on the essence prompt');
grid on;  xlabel('Adam step');  title('c. finding delta');

function [nll, g] = nll_loss(delta, Pj, W, L, newId)
    % Step 7, first term, for one prefixed prompt: run from layer L with delta added at the subject's last
    % token, and take -log p(target_new) at the last position (log softmax of the logits); and its gradient.
    o = model_forward(Pj.ids, W, @(name, l, v) add_at(name, l, v, L, Pj.pos, delta), Pj.start);
    z = o.logits(:, end);  lp = z - max(z) - log(sum(exp(z - max(z))));   % [vocab × 1] log probabilities
    loss = -lp(newId + 1);
    g = dlgradient(loss, delta);                        % backpropagation through model_forward (delta is traced)
    nll = gather(double(extractdata(loss)));
end

function [kl, g] = kl_loss(delta, E, W, L, pE)
    % Step 7, second term: KL(after || before), the KL divergence of the essence prompt's next-token distribution
    % with delta added from the one before the edit (pE): sum over tokens of p_after × log(p_after / p_before);
    % and its gradient.
    o = model_forward(E.ids, W, @(name, l, v) add_at(name, l, v, L, E.pos, delta), E.start);
    z = o.logits(:, end);  lp = z - max(z) - log(sum(exp(z - max(z))));   % log p_after
    loss = sum(exp(lp) .* (lp - log(pE + 1e-30)));
    g = dlgradient(loss, delta);
    kl = gather(double(extractdata(loss)));
end

function v = add_at(name, l, v, L, p, delta)
    % model_forward edit: add delta to the MLP output of layer L at position p. Built by concatenation, which
    % keeps delta's autodiff trace (v(:, p) = v(:, p) + delta would silently drop it).
    if strcmp(name, 'mlp_out') && l == L
        v = [v(:, 1:p - 1), v(:, p) + delta, v(:, p + 1:end)];
    end
end
