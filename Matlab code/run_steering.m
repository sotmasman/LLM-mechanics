% run_steering.m
% Steering with a difference-in-means (DIM) direction: push a concept (sentiment, or France -> Italy) by
% adding a fixed vector to the model's residual stream while it runs, and watch what it says next (Lesson 9, tab 14).
%   1. load the weights and the contrast sentences of the concept (10 'pos', 10 'neg', in matched pairs)
%   2. run each sentence; keep one vector per sentence: the residual stream leaving layer L
%   3. the direction f = mean(pos vectors) - mean(neg vectors), and how big it is
%   4. check: do the sentences separate along f?
%   5. the test: a prompt, and the words whose logits measure the concept
%   6. steer: for each strength alpha, add alpha * f to the stream leaving layer L, then
%      6a. score the next token (mean logit of the pos words - mean logit of the neg words)
%      6b. generate a continuation (the same random seed for every alpha)
%      6c. control (nShuffles > 0): score the same push along shuffled copies of f
%   7. print the scores and continuations; plot the sentences on f and the score against alpha
%
% alpha = 1 adds the full difference between the two groups' means; alpha < 0 steers toward the neg side;
% alpha = 0 is the unsteered model. Only the steering differs between alphas (same prompt, same seed).
%
% Needs: model_tokenize, model_forward, model_generate, model_token, model_weights_file, prompts_sentiment / prompts_country, layer_points.

%% Settings
model      = 'gpt2-small';   % 'gpt2-small' or 'gpt2-xl'
L          = 5;              % steer the stream leaving layer L (0-based), or 'emb' (not the last layer: that is the output)
readout    = 'mean';         % how each sentence gives one vector: 'mean' over its tokens (not <|endoftext|>) or 'last' token

concept    = 'sentiment';    % which contrast pairs: 'sentiment' or 'country'

% Each concept: its contrast sentences (10 matched pairs, loaded in step 1; f points from the 'neg' to the 'pos'
% sentences), a test prompt, and single-token words whose logits score it (pos words - neg words, step 5).
switch concept
    case 'sentiment'   % prompts_sentiment: "I love this movie." / "I hate this movie."
        contrast = 'prompts_sentiment';  pos = 'positive';  neg = 'negative';
        test_prompt = 'I think this movie is';
        pos_words   = {' great', ' good', ' wonderful', ' excellent', ' amazing'};
        neg_words   = {' terrible', ' bad', ' awful', ' horrible', ' boring'};
    case 'country'     % prompts_country: "I spent the summer in Italy." / "... in France." (f: France -> Italy)
        contrast = 'prompts_country';  pos = 'Italy';  neg = 'France';
        test_prompt = 'The Eiffel Tower is in the city of';
        pos_words   = {' Rome'};
        neg_words   = {' Paris'};
end

alphas      = [-8 -4 -2 -1 0 1 2 4 8];   % steering strengths to try
nNew        = 20;            % tokens to generate per alpha
temperature = 0;  topK = 40;  seed = 1;   % temperature 0 = greedy (topK unused); > 0 = sampling, the same seed for every alpha

% Control (step 6c): steer along shuffled copies of f: its own values in a random order (score only, no text)
nShuffles   = 0;             % number of shuffled copies; 0 = no control, 10 for Fig. 23b

%****************************************************

%% Step 1 — Load the weights and the contrast sentences
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file(model));
[prompts, labels] = feval(contrast);                     % {1 × 20} sentences, {1 × 20} labels (pos or neg)
pos_prompts = prompts(strcmp(labels, pos));              % {1 × 10}  e.g. "I love this movie." / "... in Italy."
neg_prompts = prompts(strcmp(labels, neg));              % {1 × 10}  e.g. "I hate this movie." / "... in France."
% Where to read and steer. The stream leaving layer L is the stream entering layer L+1, which model_forward
% keeps as out.resid_pre(:, :, i) and lets an edit change at ('resid_pre', layer i-1); for 'emb', i = 1.
[names, i] = layer_points(size(W.W_in, 3), 'out', L);   % i = L + 2 (MATLAB indexes from 1, layers from 0), 1 for 'emb'
where = names{i};                                        % 'emb' or '5', for printing and titles

%% Step 2 — One vector per sentence: the residual stream leaving layer L
% Run each sentence through the model (model_forward) and read the residual stream just after layer L
% (for 'emb' the embedding). A sentence has several tokens, so readout says how
% to make one vector of them: their mean ('mean', the default) or the last token's vector ('last').
% <|endoftext|> (position 1 in MATLAB) is always left out: its vector is unusually large and the same for every sentence.
% out.resid_pre(:, :, i) is that stream, [d × tokens] (i from step 1).
Xpos = zeros(size(W.W_E, 1), numel(pos_prompts));       % [d × 10]  one column per pos sentence (d = 768 for small)
for k = 1:numel(pos_prompts)
    out = model_forward(model_tokenize(pos_prompts{k}), W);
    R = double(out.resid_pre(:, 2:end, i));             % [d × tokens]  this sentence, without <|endoftext|>
    if strcmp(readout, 'last'), Xpos(:, k) = R(:, end); else, Xpos(:, k) = mean(R, 2); end
end
Xneg = zeros(size(W.W_E, 1), numel(neg_prompts));       % [d × 10]  the same for the neg sentences
for k = 1:numel(neg_prompts)
    out = model_forward(model_tokenize(neg_prompts{k}), W);
    R = double(out.resid_pre(:, 2:end, i));
    if strcmp(readout, 'last'), Xneg(:, k) = R(:, end); else, Xneg(:, k) = mean(R, 2); end
end

%% Step 3 — The steering direction f (difference in means)
% f points from the average neg sentence to the average pos one. What all sentences share
% (most of each vector) is in both means and cancels, so f keeps what differs: the sentiment, or the country.
f = mean(Xpos, 2) - mean(Xneg, 2);                      % [d × 1]
% How big is a push of alpha * f? Compare |f| with the typical length of the stream at this point:
% alpha * |f| / typical is the push as a fraction of the stream's own size (e.g. 0.13 at L = 5, so
% alpha = 4 adds about half the stream's length).
typical = mean(vecnorm([Xpos, Xneg]));                  % typical residual-stream length at this point
fprintf('Layer %s, readout ''%s'': |f| = %.2f (typical residual norm %.1f, ratio %.2f)\n', where, readout, norm(f), typical, norm(f) / typical);

%% Step 4 — Check: do the contrast sentences separate along f?
% Project every sentence on the unit direction fhat (its position along f). The pos sentences
% should land higher than the neg ones. They are the sentences f was built from, so this is a
% sanity check, not a test (run_DIM tests new sentences, held out).
fhat = f / norm(f);                                     % unit vector along f
projPos = fhat' * Xpos;                                 % [1 × 10]  each pos sentence along f
projNeg = fhat' * Xneg;                                 % [1 × 10]  each neg sentence along f
fprintf('Projection on f: %s %.2f +/- %.2f, %s %.2f +/- %.2f (mean +/- SD)\n', pos, mean(projPos), std(projPos), neg, mean(projNeg), std(projNeg));

%% Step 5 — The test prompt and the score words
% A prompt whose next word could go either way (or, for 'country', is the France answer " Paris"). The score
% after it is the mean logit of the pos words minus the mean logit of the neg words: > 0 = the model leans pos.
ids = model_tokenize(test_prompt);                      % [1 × n]  token IDs, starting with <|endoftext|>
% Each word must be a single GPT-2 token (its logit is one number); model_tokenize puts <|endoftext|> first.
posIds = zeros(1, numel(pos_words));  negIds = zeros(1, numel(neg_words));
for k = 1:numel(pos_words)
    p = model_tokenize(pos_words{k});  p = p(2:end);    % the word's tokens, without <|endoftext|>
    if numel(p) ~= 1, error('"%s" is %d GPT-2 tokens; use a single-token word.', pos_words{k}, numel(p)); end
    posIds(k) = p;
end
for k = 1:numel(neg_words)
    p = model_tokenize(neg_words{k});  p = p(2:end);
    if numel(p) ~= 1, error('"%s" is %d GPT-2 tokens; use a single-token word.', neg_words{k}, numel(p)); end
    negIds(k) = p;
end

%% Step 6 — Steer: add alpha * f to the stream leaving layer L, for each alpha
% An "edit" is a function that model_forward calls at every edit point as it runs, as v = edit(name, l, v):
% name says which activation it is ('resid_pre' = the stream entering layer l; others: 'attn_pattern',
% 'attn_by_head', 'resid_mid', 'mlp_hidden', 'mlp_out'), v holds its values, and the edit returns them,
% changed or not. Here the edit changes one thing: at 'resid_pre' of layer i-1 (the stream leaving layer L)
% it adds the push to every position except <|endoftext|> (column 1); every other call returns v untouched.
% Layers 0 to L are not affected; every layer after L reads the shifted stream (its writes change), and the
% push itself stays in the stream up to the output.
% Why not <|endoftext|>: f was built without it (step 2); its vector is about 35 times larger than the other
% tokens' (norm ~3,000 vs ~85 at layer 5); and it is the attention sink: many heads put most of their
% attention on it when they have nothing to read, so every later token reads it. Pushing it too still
% works, a little more strongly (layer 5, score at alpha +4 minus -4: 9.1 instead of 8.1), because that small change
% reaches every position through the sink.
score = zeros(size(alphas));                            % next-token score for each alpha
texts = strings(size(alphas));                          % generated continuation for each alpha

% The control for 6c: shuffled copies of f, one column each, made once and used for every alpha. A shuffle
% keeps f's exact values (so its length, and any very large values) but puts them in other dimensions,
% so it points in a random direction (almost perpendicular to f in 768 dimensions: cosine about ±0.04).
% If f worked only because of its set of values, the shuffles would steer too; if its direction matters, they won't.
rs = RandStream('twister', 'Seed', seed);              % own random stream, so the rest of the script is unaffected
fShuf = zeros(numel(f), nShuffles);                     % [d × nShuffles]
for r = 1:nShuffles
    fShuf(:, r) = f(randperm(rs, numel(f)));            % f's elements in a random order
end
scoreShuf = zeros(nShuffles, numel(alphas));            % 6c: score for each shuffled copy and alpha

for k = 1:numel(alphas)
    push = single(alphas(k) * f);                       % [d × 1]  the vector added at every position

    % The edit, built in two lines. MATLAB's one-line @ functions cannot contain an if, so:
    % steer holds the two possible answers, as two small functions of v ([d × tokens]):
    %   steer{1}: return v unchanged
    %   steer{2}: keep column 1 (<|endoftext|>) and add push to columns 2:end; MATLAB adds the
    %             [d × 1] push to each column, and [a, b] puts the columns back side by side
    steer = {@(v) v, @(v) [v(:, 1), v(:, 2:end) + push]};
    % edit asks "is this the stream leaving layer L?" (name 'resid_pre' at layer i-1). The answer is
    % true (1) or false (0), so 1 + answer picks steer{2} there and steer{1} at every other edit point,
    % and that function is applied to v. Written out with an if, edit would be:
    %   if strcmp(name, 'resid_pre') && l == i - 1, v(:, 2:end) = v(:, 2:end) + push; end
    % (Both lines are rebuilt for each alpha: an @ function keeps the values of push and i from when it is made.)
    edit = @(name, l, v) steer{1 + (strcmp(name, 'resid_pre') && l == i - 1)}(v);

    % 6a. Score the next token: one forward pass of the test prompt with the push, logits at the last position
    o = model_forward(ids, W, edit);
    % The logits for the word after the prompt, from the last position's final stream (= o.logits(:, end)):
    x = double(o.resid_post(:, end, end));              % [d × 1]  the stream leaving the last layer, at the last position
    x = x - mean(x);  x = x / sqrt(mean(x.^2) + 1e-5);  % final LayerNorm: centre, then scale to unit RMS (its weights are folded into W_U and b_U)
    logitsNext = double(W.W_U) * x + double(W.b_U');    % [vocab × 1]  unembedding: one score per vocabulary token
    score(k) = mean(logitsNext(posIds + 1)) - mean(logitsNext(negIds + 1));   % +1: MATLAB indexes from 1, IDs from 0

    % 6b. Generate a continuation with the same push (the edit applies at every generated token too) and the
    % same seed for every alpha, so the texts differ only because of the steering
    g = model_generate(ids, W, nNew, Temperature=temperature, TopK=topK, Seed=seed, Edit=edit);
    texts(k) = g.text;

    % 6c. Control: the same push along each shuffled copy of f instead of f; score only, as in 6a
    for r = 1:nShuffles
        pushS = single(alphas(k) * fShuf(:, r));
        steerS = {@(v) v, @(v) [v(:, 1), v(:, 2:end) + pushS]};
        editS = @(name, l, v) steerS{1 + (strcmp(name, 'resid_pre') && l == i - 1)}(v);
        oS = model_forward(ids, W, editS);
        scoreShuf(r, k) = mean(oS.logits(posIds + 1, end)) - mean(oS.logits(negIds + 1, end));
    end
end

%% Step 7 — Results: print, then plot
fprintf('\nTest prompt: "%s"\n', test_prompt);
for k = 1:numel(alphas)
    fprintf('alpha %+5.1f  score %+6.2f  |%s\n', alphas(k), score(k), replace(texts(k), newline, ' '));
end
if nShuffles > 0
    fprintf('Control (%d shuffled copies of f), score mean [min, max]:\n', nShuffles);
    for k = 1:numel(alphas)
        fprintf('alpha %+5.1f  %+6.2f [%+.2f, %+.2f]\n', alphas(k), mean(scoreShuf(:, k)), min(scoreShuf(:, k)), max(scoreShuf(:, k)));
    end
end
close(findall(groot, 'Type', 'figure', 'Tag', 'run_steering'));   % this script's figure from an earlier run
figure('Color', 'w', 'Position', [100 300 900 380], 'Tag', 'run_steering');
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile;                                               % a. the contrast sentences along f (step 4); small vertical jitter
plot(projPos, 1 + 0.08 * randn(size(projPos)), 'o', 'MarkerFaceColor', [0.85 0.33 0.10], 'Color', [0.85 0.33 0.10]); hold on
plot(projNeg, 0.08 * randn(size(projNeg)), 'o', 'MarkerFaceColor', [0 0.45 0.74], 'Color', [0 0.45 0.74]); hold off
yticks([0 1]); yticklabels({neg, pos}); ylim([-0.5 1.5]); grid on
xlabel('Projection on the DIM direction'); title(sprintf('Contrast sentences, layer %s', where));
nexttile;                                               % b. the steering effect on the next token (step 6a; grey: 6c)
if nShuffles > 0                                        % control: range (band) and mean (line)
    fill([alphas, fliplr(alphas)], [min(scoreShuf, [], 1), fliplr(max(scoreShuf, [], 1))], [0.6 0.6 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on
    plot(alphas, mean(scoreShuf, 1), '-', 'LineWidth', 1.5, 'Color', [0.45 0.45 0.45]);
end
plot(alphas, score, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'k', 'Color', 'k'); grid on; hold off
xline(0, 'Color', [0.6 0.6 0.6]);  yline(score(alphas == 0), '--', 'Color', [0.6 0.6 0.6]);   % dashed: unsteered
xlabel('\alpha (multiples of the DIM vector f)'); ylabel(sprintf('mean logit (%s words) - (%s words)', pos, neg));
title(sprintf('"%s" + \\alpha f at layer %s', test_prompt, where), 'Interpreter', 'tex');
