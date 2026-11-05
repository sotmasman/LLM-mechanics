% run_cosine_resid.m
% Cosine similarity between sentences in the residual stream, layer by layer:
% raw cosines vs mean-centered cosines.
%   1. load the weights; run every sentence; one vector per sentence per layer
%   2. raw cosine similarity between every pair of sentences, at every layer
%   3. (if center = true) mean-centered: subtract the mean over all sentences at that layer, then cosine
%   4. within- vs between-topic averages per layer; one figure: residual stream (emb, then leaving each
%      layer) with the attention and MLP outputs of each layer (raw, + centered), and the heatmaps at showL
%   5. the same cosines for what each sublayer writes: attention output and MLP output (printed)
%   6. (optional, setting pair) one chosen pair of sentences across layers
%   7. (example 3, three or more sentences) the first sentence vs each of the others across layers
%
% What is compared: each vector is ONE SENTENCE AT ONE LAYER (unlike
% run_cosine_tokenembed, where each vector is one word's embedding, a column of W_E):
%   1. run the sentence; at layer L every token position has a d-number (768 for small)
%      residual-stream vector (leaving layer L; "emb" = the embedding, before layer 0)
%   2. drop position 0 (<|endoftext|>: huge norm, identical in every sentence)
%   3. average over the sentence's tokens -> one d-number vector for the
%      sentence at that layer ('last' readout: the last token's vector instead)
% That gives one vector per sentence per layer; cosines are computed between
% every pair of sentences at the same layer (N x N per layer, N = number of sentences).
% At "emb" each token's vector is W_E(token) + W_pos(position), so the average
% includes the mean position embedding, identical for sentences of equal length.
% From layer 0 on, each token's vector has also absorbed context through attention.
% Centered: subtract the mean of the sentence vectors at that layer (the compared
% sentences plus any reference sentences), then cosine.
%
% Why center: all residual-stream vectors share a large common direction
% (anisotropy), so raw cosines between unrelated sentences are high. Subtracting
% the mean over the set removes what all sentences share, so the cosine shows how
% they differ from each other.
% After centering, the average pairwise cosine is pushed slightly negative by
% construction (about -1/(N-1) for N sentences of similar length): compare
% within- vs between-topic values, not the overall level.
%
% Layers: "layer L" = residual stream leaving layer L, so it includes what layer L
% wrote (0 = after the first layer; the last layer, 11 for small, is the model's
% output). "emb" = the embedding, before any layer. Attention and MLP outputs of
% layer L are plotted at L too: they are what layer L adds to give that point.
%
% Needs: cosine_sim, layer_points, model_tokenize, model_forward, model_weights_file, prompts_capital, prompts_topic.

%% Settings
model   = 'gpt2-small';
readout = 'last';            % one vector per sentence: 'mean' over its tokens (not <|endoftext|>) or 'last' token
center  = false;             % true = also mean-centered cosines (runs the reference sentences too);
                             % false = raw cosines only (reference sentences are not run: faster)
showL   = 6;                 % layer for the two heatmaps: the stream leaving layer showL (0-based), or 'emb'
pair    = [];             % also follow just these two sentences (indices into the list) across layers; [] = off.
                             % With center = true, their centered cosine uses the mean over all sentences (+ reference): centering
                             % two sentences on their own always gives -1 (they become exact opposites).

example = 3;                 % sentences to compare (loaded in step 1; equal token counts within each set):
                             % 1 = two prompts that differ only in the country word (prompts_capital),
                             %     with 8 reference prompts used ONLY to estimate the mean for centering
                             % 2 = 16 sentences in 4 topics, all 7 tokens (prompts_topic); no reference
                             %     sentences: centering uses the compared sentences themselves
                             % 3 = your own sentences (mySentences below); two or more
mySentences = {'The capital of France is the city of', 'The capital of Italy is the city of', ...
               'The capital of Japan is the city of', 'I went to the shop to buy some'};
myReference = {};            % example 3, only for center = true: extra sentences used ONLY to estimate the
                             % mean (with two sentences and none here, centering makes them exact opposites)
                             % Same token count in every sentence keeps the position embeddings identical
                             % (a warning shows the counts if they differ).

%% Step 1 — Load, run every sentence, one vector per sentence per layer
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
if example == 1
    [sentences, ~, ~, reference] = prompts_capital();    % {1 × 2} compared, {1 × 8} reference
    topics = {};                                         % no topic summary (each prompt would be its own topic)
elseif example == 2
    [sentences, topics] = prompts_topic();               % {1 × 16} sentences and their topic labels
    reference = {};
else
    sentences = mySentences;  reference = myReference;   % your own sentences
    topics = {};
end
W = load(model_weights_file(model));
if center, ref = reference; else, ref = {}; end          % reference sentences are only needed for centering
nS = numel(sentences);  nR = numel(ref);  nL = size(W.W_in, 3);
allSent = [sentences, ref];                              % compared sentences first, then the reference ones
ids = cellfun(@model_tokenize, allSent, 'UniformOutput', false);
nTok = cellfun(@numel, ids) - 1;                         % tokens per sentence (not counting <|endoftext|>)
if numel(unique(nTok)) > 1                               % unequal lengths: position embeddings differ between sentences
    warning('Sentences have different token counts (%s); position embeddings then differ between them.', mat2str(nTok));
else
    fprintf('All %d sentences (%d compared, %d reference) have %d tokens.\n', nS + nR, nS, nR, nTok(1));
end
d = size(W.W_E, 1);
Xall  = zeros(d, nS + nR, nL + 1);                       % residual stream  [d × sentence × (emb, leaving layers 0..nL-1)]
XAall = zeros(d, nS + nR, nL);                           % attention output [d × sentence × layer]: sum of heads + b_O
XMall = zeros(d, nS + nR, nL);                           % MLP output       [d × sentence × layer]
pick = @(V) V(:, end, :);                                % readout over positions (after dropping <|endoftext|>)
if strcmp(readout, 'mean'), pick = @(V) mean(V, 2); end
for s = 1:nS + nR
    o = model_forward(ids{s}, W);
    n = size(o.resid_pre, 2);
    R = cat(3, o.resid_pre(:, :, 1), o.resid_post);      % [d × n × (nL + 1)]  the embedding, then leaving each layer
    A = reshape(sum(o.attn_by_head, 2), d, n, nL) + reshape(W.b_O, d, 1, nL);   % [d × n × nL]  what attention writes
    Xall(:, s, :)  = pick(R(:, 2:end, :));               % drop <|endoftext|> (huge norm, the same in every sentence)
    XAall(:, s, :) = pick(A(:, 2:end, :));
    XMall(:, s, :) = pick(o.mlp_out(:, 2:end, :));
end
X = Xall(:, 1:nS, :);  XA = XAall(:, 1:nS, :);  XM = XMall(:, 1:nS, :);   % the compared sentences
[layerNames, iShow] = layer_points(nL, 'out', showL);    % point names in array order (emb, 0, ..., nL-1); index of showL
step = 1 + (nL > 12);                                    % label every layer (small), every 2nd (XL)
if center && nR == 0 && nS <= 2
    warning('Centering %d sentences with their own mean makes them exact opposites (centered cosine -1). Add reference sentences.', nS);
end

%% Step 2 — Raw cosine similarity, every pair of compared sentences, every layer
%% Step 3 — Mean-centered: subtract the mean over all sentences (compared + reference) at that layer first
off = ~eye(nS);                                          % leave out the diagonal (= 1)
hasTopics = ~isempty(topics);
if hasTopics
    if numel(topics) ~= nS, error('topics needs one label per sentence (%d labels for %d sentences).', numel(topics), nS); end
    [~, ~, g] = unique(topics, 'stable');  same = g == g';   % same-topic pairs
else
    same = [];
end
% stats columns: topics: [raw within, raw between, centered within, centered between]; else [raw, -, centered, -]
[Craw,  Ccen,  stats]  = cos_by_layer(X,  Xall,  center, same, off);   % residual stream
[CrawA, CcenA, statsA] = cos_by_layer(XA, XAall, center, same, off);   % attention output of each layer
[CrawM, CcenM, statsM] = cos_by_layer(XM, XMall, center, same, off);   % MLP output of each layer

%% Step 4 — Report and plot
if ~center,    centerNote = 'not computed (center = false)';
elseif nR > 0, centerNote = sprintf('mean over %d compared + %d reference sentences', nS, nR);
else,          centerNote = sprintf('mean over the %d compared sentences', nS); end
fprintf('Readout ''%s''. Centered: %s.\n', readout, centerNote);
cols = [1 3];  if hasTopics, cols = 1:4; end             % columns of stats to print
if ~center, cols = cols(ismember(cols, [1 2])); end      % raw only
names = {'raw within', 'raw between', 'centered within', 'centered between'};
if ~hasTopics, names = {'raw', '', 'centered', ''}; end
if hasTopics, what = 'between sentences (within / between topics)'; else, what = 'over all pairs of compared sentences'; end
fprintf('Mean cosine %s:\nlayer   %s\n', what, strjoin(compose('%16s', string(names(cols))), ''));
for p = 1:nL + 1, fprintf('%-6s  %s\n', layerNames{p}, sprintf('%16.2f', stats(p, cols))); end   % p = array index: emb, 0, ...
close(findall(groot, 'Type', 'figure', 'Tag', 'run_cosine_resid'));
% Line plots: the residual stream (emb, then leaving every layer) and what each sublayer writes
% (attention output, MLP output of layers 0..nL-1, at the layer that writes them), raw (+ centered); then the heatmaps.
% A sublayer's output can differ between sentences even while the residual stream
% looks identical: its writes are small compared with the large shared residual.
cR = [0 0 0];  cA = [0 0.45 0.74];  cM = [0.85 0.33 0.10];   % residual, attention, MLP
x = -1:nL - 1;  xs = 0:nL - 1;                           % residual stream: emb (drawn at -1), 0..nL-1; sublayers: 0..nL-1
figure('Color', 'w', 'Position', [60 150 520 * (2 + 2 * center) 470], 'Tag', 'run_cosine_resid');
tiledlayout(1, 2 + 2 * center, 'TileSpacing', 'compact');   % raw line plot (+ centered), raw heatmap (+ centered)
for k = 1:1 + center                                     % raw panel (+ centered panel)
    nexttile;  c = 1 + 2 * (k - 1);                      % stats column: 1 = raw (same/all), 3 = centered
    plot(x, stats(:, c), '-o', 'Color', cR, 'LineWidth', 1.5); hold on
    plot(xs, statsA(:, c), '-o', 'Color', cA, 'LineWidth', 1.5);
    plot(xs, statsM(:, c), '-o', 'Color', cM, 'LineWidth', 1.5);
    leg = {'residual stream', 'attention output', 'MLP output'};
    if hasTopics                                         % dashed: different topics
        plot(x, stats(:, c + 1), '--o', 'Color', cR, 'LineWidth', 1);
        plot(xs, statsA(:, c + 1), '--o', 'Color', cA, 'LineWidth', 1);
        plot(xs, statsM(:, c + 1), '--o', 'Color', cM, 'LineWidth', 1);
        leg = [strcat(leg, ', same topic'), strcat(leg, ', different topics')];
    end
    hold off
    yline(0, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');  ylim([-1 1]);  grid on
    xticks(x(1:step:end)); xticklabels(layerNames(1:step:end)); xtickangle(0); xlim([x(1) - 0.5, x(end) + 0.5]);
    xlabel('Layer'); ylabel('mean cosine between sentences');
    legend(leg, 'Location', 'southwest');
    if k == 1, title('Raw cosine: residual stream vs what each sublayer writes');
    else,      title({'Mean-centered cosine', centerNote}); end
end
cm = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
if hasTopics                                             % axis labels: topic + number within it, or S1, S2, ...
    lab = cell(1, nS);
    for s = 1:nS, lab{s} = sprintf('%s %d', topics{s}, sum(strcmp(topics(1:s), topics{s}))); end
    edges = find(~strcmp(topics(1:end - 1), topics(2:end))) + 0.5;   % lines where the topic changes
else
    lab = compose('S%d', 1:nS);  edges = [];
end
for k = 1:1 + center                                     % heatmaps at layer showL (raw, + centered)
    nexttile;
    if k == 1, M = Craw(:, :, iShow); t = 'Raw'; else, M = Ccen(:, :, iShow); t = 'Mean-centered'; end
    imagesc(M); axis square; clim([-1 1]); colormap(gca, cm);
    xticks(1:nS); xticklabels(lab); xtickangle(60); yticks(1:nS); yticklabels(lab);
    for b = edges, xline(b, 'Color', [0.4 0.4 0.4]); yline(b, 'Color', [0.4 0.4 0.4]); end
    if nS <= 6                                           % few sentences: write the values in the cells
        for a = 1:nS, for b = 1:nS, text(b, a, sprintf('%.2f', M(a, b)), 'HorizontalAlignment', 'center'); end, end
    end
    title(sprintf('%s cosine, %s', t, replace(['layer ' layerNames{iShow}], 'layer emb', 'embedding')));
end
cb = colorbar;  cb.Label.String = 'cosine similarity';

%% Step 5 — Attention and MLP outputs: the same cosines, for what each sublayer writes (printed; plotted in step 4)
% At layer L: the residual stream leaving L, the attention output of L (all heads
% + b_O) and the MLP output of L, each read out the same way (readout).
fprintf('\nSublayer outputs vs residual stream (%s cosine', 'raw');
if hasTopics, fprintf('; same topic / different topics'); end
fprintf('):\nlayer     resid      attn       MLP\n');
if hasTopics, fprintf('%-6s  %4.2f/%5.2f\n', 'emb', stats(1, 1:2)); else, fprintf('%-6s  %6.2f\n', 'emb', stats(1, 1)); end
for L = 0:nL - 1                                         % layer, 0-based; stream array index L + 2 (1 = emb), sublayers L + 1
    if hasTopics
        fprintf('%-6d  %4.2f/%5.2f  %4.2f/%5.2f  %4.2f/%5.2f\n', L, stats(L + 2, 1:2), statsA(L + 1, 1:2), statsM(L + 1, 1:2));
    else
        fprintf('%-6d  %6.2f  %8.2f  %8.2f\n', L, stats(L + 2, 1), statsA(L + 1, 1), statsM(L + 1, 1));
    end
end

%% Step 6 — One pair of sentences across layers (optional)
% The same matrices as above, read at one cell per layer.
if ~isempty(pair)
    i = pair(1);  j = pair(2);
    rawPair = squeeze(Craw(i, j, :));  cenPair = squeeze(Ccen(i, j, :));     % residual stream  [nL+1 × 1]
    rawA = squeeze(CrawA(i, j, :));  rawM = squeeze(CrawM(i, j, :));        % attention, MLP   [nL × 1]
    cenA = squeeze(CcenA(i, j, :));  cenM = squeeze(CcenM(i, j, :));
    if hasTopics, ti = [' (' topics{i} ')'];  tj = [' (' topics{j} ')']; else, ti = '';  tj = ''; end
    fprintf('\nPair: [%d] "%s"%s  vs  [%d] "%s"%s\n', i, sentences{i}, ti, j, sentences{j}, tj);
    fprintf('layer    resid    attn     MLP');  if center, fprintf('   | centered: resid    attn     MLP'); end
    fprintf('\n');
    for p = 1:nL + 1                                     % p = stream array index: 1 = emb, p = layer p - 2 after it
        if p > 1, a = [rawA(p - 1), rawM(p - 1)];  ac = [cenA(p - 1), cenM(p - 1)]; else, a = [NaN NaN];  ac = [NaN NaN]; end   % nothing written before layer 0
        fprintf('%-6s  %6.2f  %6.2f  %6.2f', layerNames{p}, rawPair(p), a);
        if center, fprintf('   |           %6.2f  %6.2f  %6.2f', cenPair(p), ac); end
        fprintf('\n');
    end
    figure('Color', 'w', 'Position', [200 100 660 440], 'Tag', 'run_cosine_resid');
    plot(x, rawPair, '-o', 'Color', cR, 'LineWidth', 1.5); hold on
    plot(xs, rawA, '-o', 'Color', cA, 'LineWidth', 1.5);
    plot(xs, rawM, '-o', 'Color', cM, 'LineWidth', 1.5);
    leg = {'residual stream', 'attention output', 'MLP output'};
    if center                                            % dashed: centered
        plot(x, cenPair, '--o', 'Color', cR, 'LineWidth', 1);
        plot(xs, cenA, '--o', 'Color', cA, 'LineWidth', 1);
        plot(xs, cenM, '--o', 'Color', cM, 'LineWidth', 1);
        leg = [strcat(leg, ', raw'), strcat(leg, ', centered')];
    end
    hold off
    yline(0, 'Color', [0.6 0.6 0.6]);  ylim([-1 1]);  grid on
    xticks(x(1:step:end)); xticklabels(layerNames(1:step:end)); xtickangle(0); xlim([x(1) - 0.5, x(end) + 0.5]);
    xlabel('Layer'); ylabel('cosine');
    legend(leg, 'Location', 'southwest');
    title({sprintf('"%s"%s', sentences{i}, ti), sprintf('vs "%s"%s', sentences{j}, tj)}, 'Interpreter', 'none');
end

%% Step 7 — Your own sentences (example 3, three or more): the first sentence vs each of the others
% One line per other sentence: raw cosine of its residual stream with the first sentence's, at every layer.
if example == 3 && nS > 2
    c1 = squeeze(Craw(1, 2:nS, :));                       % [nS-1 × nL+1]  row k = sentence 1 vs sentence k+1
    fprintf('\nFirst sentence vs each of the others (raw cosine):\nlayer   %s\n', strjoin(compose('%8d', 2:nS), ''));
    for p = 1:nL + 1, fprintf('%-6s  %s\n', layerNames{p}, sprintf('%8.2f', c1(:, p))); end   % emb, 0, ..., nL-1
    figure('Color', 'w', 'Position', [240 80 700 460], 'Tag', 'run_cosine_resid');
    plot(x, c1', '-', 'LineWidth', 1.5);  grid on
    xticks(x(1:step:end)); xticklabels(layerNames(1:step:end)); xtickangle(0); xlim([x(1) - 0.5, x(end) + 0.5]);
    xlabel('Layer'); ylabel('cosine with the first sentence');
    legend(compose('"%s"', string(sentences(2:nS))), 'Location', 'southwest', 'Interpreter', 'none');
    title({sprintf('Raw cosine with "%s"', sentences{1}), 'one line per other sentence'}, 'Interpreter', 'none');
end

% -------------------------------------------------------------------------
function [Craw, Ccen, stats] = cos_by_layer(X, Xall, center, same, off)
% Cosine between every pair of compared sentences, at every layer.
%   X     [d × nS × nLayers]  compared sentences;  Xall: compared + reference (for the mean)
%   same  [nS × nS] same-topic pairs ([] = no topics);  off: leave out the diagonal
% stats [nLayers × 4]: topics -> [raw within, raw between, centered within, centered between]
%                      no topics -> [raw all pairs, NaN, centered all pairs, NaN]
    [~, nS, nLy] = size(X);
    Craw = zeros(nS, nS, nLy);  Ccen = NaN(nS, nS, nLy);  stats = NaN(nLy, 4);
    for L = 0:nLy - 1                                    % layer, 0-based; arrays use L + 1
        Xlay = X(:, :, L + 1);                           % [d × nS]  the compared sentences at this layer
        Craw(:, :, L + 1) = cosine_sim(Xlay);            % raw
        if center                                        % remove the shared mean (compared + reference), then cosine
            Ccen(:, :, L + 1) = cosine_sim(Xlay - mean(Xall(:, :, L + 1), 2));
        end
        r = Craw(:, :, L + 1);  c = Ccen(:, :, L + 1);
        if ~isempty(same), stats(L + 1, :) = [mean(r(same & off)), mean(r(~same)), mean(c(same & off)), mean(c(~same))];
        else,              stats(L + 1, :) = [mean(r(off)), NaN, mean(c(off)), NaN]; end
    end
end
