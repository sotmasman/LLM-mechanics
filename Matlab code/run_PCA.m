% run_PCA.m
% PCA of trajectories (after Fernando & Guitchounts 2025, Lesson 8 tab 2): ONE PCA over
% every sentence's state at every layer, so all layers share the same axes and each
% sentence becomes a path through that space as it moves through the layers.
%   0. turn the hand-typed labels into numbers g (used only to colour the paths)
%   1. load the weights; run every sentence; one vector per sentence per layer
%      (the embedding (emb), then the output of each layer)
%   2. stack all (sentence, layer) states from layer 0 to the last layer into one
%      matrix (emb is left out), subtract ONE mean over all of them, then PCA (via
%      SVD): principal components (PCs) = directions of largest variance over sentences
%      AND layers; every state's score on each PC
%   3. which PCs are the shared path and which are differences between sentences: each
%      PC's correlation with depth and its share of variance between sentences at the
%      same layer; the two most sentence-specific PCs are picked (no labels used)
%   4. how far apart the classes are at each layer, in the full (scaled) residual stream:
%      mean distance between class means, mean distance of sentences from their own
%      class mean, and their ratio
%   5. plot: the paths in 3D on three PCs (a; setting pcs3), coloured by class, with the
%      mean path of each class; variance explained by the first PCs (b); the distances
%      and their ratio by layer (c)
%
% Notes:
% - Labels are not used to find the PCs; they only colour the paths and form the classes
%   for the distances in step 4.
% - One mean over all states and layers is subtracted (as in the paper); the PCs
%   depend on which sentences are in the set.
% - The residual stream's length grows ~100x with depth; on the raw vectors PC1-PC2
%   (91% of the variance for sentiment) were just the path all sentences share. So each
%   layer is first scaled to the same average length (step 1, one number per layer).
%
% Needs: model_tokenize, model_forward, model_weights_file, layer_points, prompts_* functions.

%% Settings
model    = 'gpt2-small';   % 'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'pythia-160m'
contrast = 'topic';       % which sentences and which grouping, roughly ordered from easy (top) to hard (bottom):
                              %   'topic', 'sentiment', 'strong-weak-praise', 'negation-adjective', 'negation-sentiment',
                              %   'passive-lastword', 'passive-role'   (as in run_DIM; see the examples there)
nPC      = 10;                % how many leading PCs to report
pcs3     = [1 2 3];           % the three PCs for the 3D plot; [] = PC1 (shared path) + the two most sentence-specific PCs
position = 'last word';            % token read: 'last', 'last word' (last non-punctuation token) or 'mean' (see token_positions)
                              % Note: Pythia's final "." is an attention sink (massive activations, almost no sentence
                              % information), so with Pythia last-token results are erratic when sentences end in "."; use 'last word'.
switch contrast
    case 'topic',              [sentences, topics] = prompts_topic();
    case 'sentiment',          [sentences, topics] = prompts_sentiment();
    case 'strong-weak-praise', [sentences, topics] = prompts_strong_weak_praise();
    case 'negation-adjective', [sentences, topics] = prompts_negation('adjective');
    case 'negation-sentiment', [sentences, topics] = prompts_negation('sentiment');
    case 'passive-lastword',   [sentences, topics] = prompts_passive('lastword');
    case 'passive-role',       [sentences, topics] = prompts_passive('role');
    otherwise, error('contrast not recognised: see the list in Settings.');
end

%****************************************************

%% Step 0 — Labels as numbers (not used to find the PCs; only to colour the paths)
[tNames, ~, g] = unique(topics, 'stable');  nT = numel(tNames);   % g = class number per sentence [nS × 1]

%% Step 1 — Load; one vector per sentence per layer (at the token set by position)
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file(model));
nS = numel(sentences);  nL = size(W.W_in, 3);  d = size(W.W_E, 1);
X = zeros(d, nS, nL + 1);                                % [d × sentence × (emb, layers 0..nL-1)]
for s = 1:nS
    ids = model_tokenize(sentences{s}, model);
    o = model_forward(ids, W);
    p = token_positions(ids, W, position);               % the token(s) read (several for 'mean')
    X(:, s, 1:nL)  = mean(o.resid_pre(:, p, :), 2);      % emb, then the output of layers 0..nL-2
    X(:, s, nL + 1) = mean(o.resid_post(:, p, end), 2);  % the output of the last layer
end
layerNames = layer_points(nL);                           % {'emb', '0', ..., 'nL-1'}
% Scale each layer to the same average length (one number per layer, as in run_probe).
% The raw stream's length grows ~100x with depth (small: ~5 at emb, ~500 after the last layer), so
% without this the late layers dominate the variance and PC1-PC2 are just the path all
% sentences share through depth. Dividing by one number per layer keeps every direction
% and does not depend on the sentence set (no per-layer mean is subtracted).
X = X ./ mean(vecnorm(X), 2);                            % each layer / its average length

%% Step 2 — One PCA over all sentences and layers 0..nL-1 (emb left out)
% emb (the embedding) is left out: it is unlike every later state, so with it PC1
% (~37% of the variance) was just the jump from emb to the output of layer 0, the same
% for all sentences.
% Stack the states: column (s, L) = sentence s at layer L, nS × nL columns in all.
% Subtract one mean over all of them, then SVD: Xc = U * Sg * V'
%   U(:, j)                PC j, a unit direction in the residual stream, shared by all layers [d × 1]
%   Sg(j, j)^2 / sum       fraction of the variance (over sentences and layers) along PC j
%   U(:, j)' * Xc          every state's score on PC j: the coordinates of the paths
keepL = 1:nL;                                            % points kept: array index keepL + 1 = layers 0 .. nL-1 (emb left out)
A  = reshape(X(:, :, keepL + 1), d, nS * numel(keepL));  % [d × (nS·nL)]  sentence index varies fastest
Xc = A - mean(A, 2);                                     % one mean over all states (sentences and layers)
[U, Sg, ~] = svd(Xc, 'econ');
sv = diag(Sg);  varExp = sv.^2 / sum(sv.^2);             % [nStates × 1]  fraction of variance per PC
k = min(nPC, numel(sv));
Z = reshape(U(:, 1:k)' * Xc, k, nS, numel(keepL));       % [PC × sentence × layer]  path coordinates

%% Step 3 — Which PCs are depth (shared) and which are differences between sentences?
% For each PC: correlation of its scores with layer depth, and the share of its variance
% that lies between sentences at the same layer (0 = all sentences move together, the
% shared path; ~1 = only differences between sentences; values slightly above 1 come
% from the n - 1 corrections of the per-layer and overall variances). No labels used.
% sentPCs = the two PCs with the most between-sentence variance (share × variance
% explained), i.e. the largest directions along which sentences differ (used by pcs3 = []).
depth = repmat(keepL, nS, 1);                            % [nS × nL]  layer of each state
between = zeros(k, 1);  rDepth = zeros(k, 1);
for j = 1:k
    zj = squeeze(Z(j, :, :));                            % [nS × nL]
    rDepth(j) = corr(zj(:), depth(:));
    between(j) = mean(var(zj, 0, 1)) / var(zj(:));       % average across-sentence variance per layer / total
end
[~, ix] = sort(between .* varExp(1:k), 'descend');  sentPCs = sort(ix(1:2))';   % the two PCs with the most between-sentence variance
fprintf('PCA of trajectories, %s, %s, position ''%s'' (%d sentences × layers 0..%d)\n', model, contrast, position, nS, nL - 1);
fprintf('PC   var expl.   corr with depth   share between sentences\n');
for j = 1:k, fprintf('%-3d  %8.1f%%   %14.2f   %20.2f\n', j, 100 * varExp(j), rDepth(j), between(j)); end
fprintf('Most sentence-specific PCs: PC%d and PC%d\n', sentPCs);

%% Step 4 — How far apart are the classes at each layer? (full scaled space, not just the PCs)
% At each layer, in the full residual stream (after the per-layer scaling of step 1):
%   between  = mean Euclidean distance between the class means (every pair of classes)
%   within   = mean Euclidean distance of each sentence from its own class mean
%   ratio    = between / within: ~1 = the classes differ about as much as sentences vary
%              within a class (overlap); larger = separated groups
% The shared path drops out (at a given layer all classes share it), and the ratio is not
% fooled by everything spreading out with depth. Labels are used here (to form the classes).
% Same idea as RSA's between/within ratio, but with Euclidean distance instead of 1 - r.
dBetween = NaN(nL + 1, 1);  dWithin = NaN(nL + 1, 1);
for L = 1:nL                                             % array index L + 1 = layer L - 1 (emb left out as above)
    XL = X(:, :, L + 1);                                 % [d × nS]
    mu = zeros(d, nT);
    for t = 1:nT, mu(:, t) = mean(XL(:, g == t), 2); end % [d × nT]  class means
    dm = zeros(nT);                                      % distances between class means
    for a = 1:nT, for b = 1:nT, dm(a, b) = norm(mu(:, a) - mu(:, b)); end, end
    dBetween(L + 1) = mean(dm(triu(true(nT), 1)));
    dWithin(L + 1)  = mean(vecnorm(XL - mu(:, g)));      % each sentence vs its own class mean
end
ratio = dBetween ./ dWithin;
fprintf('layer   between   within   ratio   (Euclidean distance, full scaled space)\n');
for L = 1:nL, fprintf('%-5s  %8.3f  %7.3f  %6.2f\n', layerNames{L + 1}, dBetween(L + 1), dWithin(L + 1), ratio(L + 1)); end

%% Step 5 — Plot
% a. the paths in 3D on the PCs in pcs3 (rotate the figure in MATLAB to look around);
% b. variance explained by each PC; c. distances between and within classes by layer.
if isempty(pcs3), pcs3 = [1 sentPCs]; end                % PC1 (the shared path) + the two most sentence-specific PCs
close(findall(groot, 'Type', 'figure', 'Tag', 'run_PCA'));
figure('Color', 'w', 'Position', [40 100 1900 620], 'Tag', 'run_PCA');
tl = tiledlayout(1, 4, 'TileSpacing', 'compact');  title(tl, sprintf('%s, %s, position ''%s'': PCA of trajectories (one PCA over all sentences and layers)', model, contrast, position), 'Interpreter', 'none');
cols = lines(nT);
nexttile([1 2]);  hold on;  h = gobjects(nT, 1);         % a. 3D paths
for s = 1:nS                                             % each sentence's path, faint
    z = squeeze(Z(pcs3, s, :));                          % [3 × nL]
    plot3(z(1, :), z(2, :), z(3, :), '-', 'Color', [cols(g(s), :) 0.25]);
end
for t = 1:nT                                             % each class's mean path, with layer markers
    m = squeeze(mean(Z(pcs3, g == t, :), 2));            % [3 × nL]
    h(t) = plot3(m(1, :), m(2, :), m(3, :), '-o', 'Color', cols(t, :), 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', cols(t, :));
    text(m(1, end), m(2, end), m(3, end), ['  layer ' layerNames{end}], 'Color', cols(t, :), 'FontSize', 8);
end
m1 = squeeze(mean(Z(pcs3, :, 1), 2));                    % start: mean of all sentences at layer 0 (emb left out)
plot3(m1(1), m1(2), m1(3), 's', 'MarkerSize', 11, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
text(m1(1), m1(2), m1(3), '  start (layer 0, all sentences)', 'FontSize', 8);
hold off;  grid on;  view(3);  rotate3d on
set(gca, 'SortMethod', 'childorder');                    % draw in plotting order, so the start square (last) stays on top
xlabel(sprintf('PC%d (%.0f%%)', pcs3(1), 100 * varExp(pcs3(1))));
ylabel(sprintf('PC%d (%.0f%%)', pcs3(2), 100 * varExp(pcs3(2))));
zlabel(sprintf('PC%d (%.0f%%)', pcs3(3), 100 * varExp(pcs3(3))));
legend(h, strcat(tNames, ' (mean path)'), 'Location', 'northeast');
title({sprintf('a. paths on PC%d, PC%d, PC%d', pcs3), '(faint = each sentence; bold = class mean; rotate to look around)'});
nexttile;                                                % b. variance explained
bar(1:k, 100 * varExp(1:k), 'FaceColor', [0.6 0.6 0.6]);  grid on
xlabel('PC');  ylabel('% of variance (all sentences and layers)');
title('b. variance explained by each PC');
nexttile;                                                % c. separation of the classes by layer
if nL > 24, step = 4; elseif nL > 12, step = 2; else, step = 1; end   % label every layer (small), every 4th (XL)
xt = 1:step:nL;  if xt(end) ~= nL, xt(end + 1) = nL; end   % array index xt + 1 = layer xt - 1; always label the last layer
% solid = the raw distances (left axis); dashed = their ratio (right axis)
yyaxis left
plot(-1:nL - 1, dBetween, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);  hold on
plot(-1:nL - 1, dWithin, '-', 'Color', [0 0.45 0.74], 'LineWidth', 1.5);  hold off
ylabel('Euclidean distance');
yyaxis right
plot(-1:nL - 1, ratio, '--o', 'Color', [0.2 0.2 0.2], 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', [0.2 0.2 0.2]);
ylabel('between/within ratio');  yline(1, 'k:');
ax = gca;  ax.YAxis(1).Color = 'k';  ax.YAxis(2).Color = [0.2 0.2 0.2];
grid on;  xlim([-0.5 nL - 0.5]);  xticks(xt - 1);  xticklabels(layerNames(xt + 1));  xtickangle(0);   % layer L at x = L
xlabel('Layer');
legend({'between class means (left)', 'within a class (left)', 'between/within ratio (right)', ''}, 'Location', 'northwest');
title({'c. how far apart are the classes?', '(full scaled space; 1 = overlap as much as they differ)'});
