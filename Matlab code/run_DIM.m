% run_DIM.m
% Difference in means (DIM) as a readout of topic or sentiment, layer by layer
% (setting contrast). Below, "topic" stands for any labelled class.
%   0. turn the hand-typed topic labels into numbers g (one per sentence)
%   1. load the weights; run every sentence; one vector per sentence per layer
%      (the embedding (emb), then the output of each layer, at the token set by
%      position: the last word by default)
%   2. for each topic, at every layer: the DIM direction
%         f = mean(sentences of that topic) - mean(all other sentences)
%      and a held-out readout: leave one sentence out, compute f from the others,
%      project the left-out sentence onto f (score = x · f / ||f||)
%   3. how well the held-out scores separate the topic from the rest: AUC = the
%      fraction of (topic, other) sentence pairs in which the topic sentence
%      scores higher (0.5 = chance, 1 = perfect); rank-based, so unequal group
%      sizes (e.g. 4 vs 12) do not matter; then the selectivity index SI = 2·AUC - 1
%      (0 = chance, +1 = perfect, -1 = perfectly reversed)
%   4. does the direction stay the same across depth? cos(f_i, f_j) between the
%      DIM directions of every pair of layers (one topic)
%   5. plot: held-out scores at one layer (a), SI by layer for every topic (b),
%      the layer x layer cosine of one topic's direction (c); with checkpointRange,
%      steps 1-3 are repeated for every checkpoint; (d) SI at layer showL and (e) the mean
%      SI over classes at every layer (heatmap) vs training, steps below and tokens above
%      (a-c: the last checkpoint)
%
% Notes:
% - The subtraction cancels what all sentences share (the large common direction
%   that compresses raw cosines and RDMs), so f is only what distinguishes the topic.
% - Held-out scores (leave-one-out) avoid the trivial result that a direction built
%   from a sentence separates that same sentence.
% - 4 sentences per topic (10 per sentiment, 20 per class for negation) is few: treat
%   the values as a first look.
% - Numbers in the comments below use the topic example (4 topic vs 12 other sentences);
%   for sentiment it is 10 vs 10 (100 pairs), for negation 20 vs 20 (400 pairs).
%
% Needs: cosine_sim, model_tokenize, model_forward, model_weights_file, layer_points,
%        prompts_topic / prompts_sentiment / prompts_negation.

%% Settings
model    = 'gpt2-small';     % 'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'pythia-160m'
checkpointRange = [];         % [] = the trained model; otherwise its training checkpoints on disk at these steps,
                              % e.g. 0:2000 or 0:400000 (all): steps 1-3 are repeated for each, panels a-c show the
                              % last checkpoint, panel d shows SI at layer showL and panel e every layer (heatmap)
                              % across training (steps below, tokens above).
                              % On disk: gpt2-small, gpt2-small-battlestar (Stanford CRFM seeds alias x21, battlestar x49), pythia-160m, pythia-160m-seed1, pythia-160m-seed2 (PolyPythias); see model_checkpoints.
contrast = 'passive-role';   % which sentences and which grouping, roughly ordered from easy (top) to hard (bottom):
                              %   'topic'               16 sentences, food / sport / weather / politics (prompts_topic)
                              %   'sentiment'           20 sentences, "I love ..." vs "I hate ..." (prompts_sentiment)
                              %   'strong-weak-praise'  20 sentences, "The food was extremely good." vs "The food was fairly good."
                              %                         (same last word: needs context; prompts_strong_weak_praise)
                              %   'negation-adjective'  40 negation sentences grouped by the adjective (lexical control; prompts_negation), e.g.
                              %                         "The movie was not good." (positive adjective) vs "The movie was not bad." (negative adjective)
                              %   'negation-sentiment'  the same 40 grouped by meaning (compositional), e.g.
                              %                         "The movie was not bad." (positive) vs "The movie was not good." (negative)
                              %   'passive-lastword'    48 sentences, who acts x active/passive, grouped by the last word (lexical control; prompts_passive), e.g.
                              %                         "The cat watched the mouse." + "The cat was watched by the mouse." (ends with prey)
                              %   'passive-role'        the same 48 grouped by who acts, e.g.
                              %                         "The cat watched the mouse." + "The mouse was watched by the cat." (predator acts)
                              %                         vs "The mouse watched the cat." + "The cat was watched by the mouse." (prey acts)
position = 'last word';       % which token's residual stream to read:
                              %   'last'      the last token (in most sets the final ".")
                              %   'last word' the last token that is not punctuation (the word before
                              %               the "."). Identical to deleting the "." and reading the last
                              %               token, since no token sees later ones. Use for Pythia, whose
                              %               "." is an attention sink with massive activations and almost
                              %               no information about the sentence (Sun et al. 2024).
                              %   'mean'      the mean over all tokens except the start token (see token_positions)
                              % Note: so with Pythia, last-token results are erratic when sentences end in "."; use 'last word'.
target   = '';                % class for panels a and c ('' = the first class, e.g. 'weather' for topic)
showL    = 6;                 % panel a: the stream leaving layer showL (0-based), or 'emb'
% (Below, "topic" means any labelled class: e.g. positive / negative for sentiment.)
% Each prompts_ function returns the sentences, one label per sentence (topics; becomes
% g in step 0) and groupID: sentences sharing a groupID are left out together in step 2.
switch contrast
    case 'topic',              [sentences, topics, groupID] = prompts_topic();
    case 'sentiment',          [sentences, topics, groupID] = prompts_sentiment();
    case 'strong-weak-praise', [sentences, topics, groupID] = prompts_strong_weak_praise();
    case 'negation-adjective', [sentences, topics, groupID] = prompts_negation('adjective');
    case 'negation-sentiment', [sentences, topics, groupID] = prompts_negation('sentiment');
    case 'passive-lastword',   [sentences, topics, groupID] = prompts_passive('lastword');
    case 'passive-role',       [sentences, topics, groupID] = prompts_passive('role');
    otherwise, error('contrast not recognised: see the list in Settings.');
end

%****************************************************

%% Step 0 — Topic labels as numbers
% g is what makes DIM supervised: every step below splits the sentences into
% "this topic" (g == t) and "the rest" (g ~= t) using g.
% Where it comes from: the labels typed by hand in the prompts_ file chosen in
% Settings, one label per sentence in the same order as sentences. unique turns the names into numbers:
%   tNames = {'food', 'sport', 'weather', 'politics'}   (distinct topics, in order of appearance)
%   g      = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]'          [nS × 1]  topic number of each sentence
%   (sentiment: tNames = {'positive', 'negative'}, g = ten 1s then ten 2s)
% So if the sentences or their order change, topics must change with them.
[tNames, ~, g] = unique(topics, 'stable');  nT = numel(tNames);
if isempty(target), target = tNames{1}; end

%% Step 1 — Load; one vector per sentence per layer (at the token set by position)
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
[files, ckSteps, tokPerStep, ckNames] = model_checkpoints(model, checkpointRange);   % one file, or one per checkpoint
nC = numel(files);
for ck = 1:nC                                            % steps 1-3 for every checkpoint (once for the trained model)
    W = load(files{ck});
    nS = numel(sentences);  nL = size(W.W_in, 3);  d = size(W.W_E, 1);
    X = zeros(d, nS, nL + 1);                                % [d × sentence × (emb, layers 0..nL-1)]
    for s = 1:nS
        ids = model_tokenize(sentences{s}, W);               % the tokenizer follows the weights
        o = model_forward(ids, W);
        p = token_positions(ids, W, position);               % the token(s) read (several for 'mean')
        X(:, s, 1:nL)  = mean(o.resid_pre(:, p, :), 2);      % emb, then the output of layers 0..nL-2
        X(:, s, nL + 1) = mean(o.resid_post(:, p, end), 2);  % the output of the last layer
    end
    [layerNames, iShow] = layer_points(nL, 'out', showL);   % {'emb', '0', ..., 'nL-1'}; iShow = array index of showL
    if nL > 24, step = 4; elseif nL > 12, step = 2; else, step = 1; end   % label every layer (small), every 4th (XL)
    xt = 1:step:nL + 1;  if xt(end) ~= nL + 1, xt(end + 1) = nL + 1; end    % tick positions (array index); always label the last layer

    %% Step 2 — Held-out DIM scores: leave one sentence (or matched pair) out, build f (feature direction), project the left-out one
    S = zeros(nS, nL + 1, nT);                               % [sentence × layer × topic]  held-out score
    for t = 1:nT
        for L = 0:nL                                         % each point: array index L + 1 (1 = emb, k + 2 = layer k)
            for i = 1:nS
                % Leave out sentence i together with its group (groupID): its matched twin
                % (sentiment) or its set of 4 (negation); for topic each sentence is alone.
                % Leaving out only i would keep a near-identical sentence in the other class
                % (identical at emb: same last token, same position), so f would point
                % away from sentence i and give a spurious negative SI.
                keep = groupID(:) ~= groupID(i);             % [nS × 1]  all sentences except i's group
                % f = centroid of topic t minus centroid of the rest, unit by unit, built
                % without sentence i (3 vs 12 sentences if i is in topic t, else 4 vs 11):
                %   X(:, keep & g == t, L + 1)  [d × 3 or 4]   topic-t sentences at layer L
                %   mean(..., 2)                [d × 1]        their average, unit by unit
                % f_u > 0: unit u is higher for topic t; f_u < 0: lower; ~0: the unit doesn't care.
                % Whatever all sentences share (the large common direction) is in both means
                % and cancels, so f is only what distinguishes topic t.
                f = mean(X(:, keep & g == t, L + 1), 2) - mean(X(:, keep & g ~= t, L + 1), 2);   % [d × 1]  the DIM direction (feature direction)
                % Score of the left-out sentence = its projection onto f:
                %   x' * f   = sum over units of x_u * f_u  (high when x follows the topic-t pattern)
                %   / norm(f) makes f unit length, so the score = the length of x's shadow on the
                %   topic-t axis (= ||x|| * cos(x, f)), in residual-stream units.
                % Dividing by norm(f) puts all nS scores on one scale: each comes from a slightly
                % different f (a different sentence left out). The shared part of x adds about
                % the same offset to every score; that doesn't change the ranking used by the AUC.
                S(i, L + 1, t) = X(:, i, L + 1)' * f / norm(f);   % [1 × 1]  projection of the left-out sentence
            end
        end
    end

    %% Step 3 — AUC: how often a topic sentence outscores a sentence from another topic
    % AUC = the probability that a randomly picked topic-t sentence scores higher on the
    % topic-t direction than a randomly picked other sentence. With 4 topic and 12 other
    % sentences there are exactly 4 × 12 = 48 pairs, so checking them all gives that
    % probability exactly. Why every pair:
    % - the two groups have no natural pairing, so every combination counts equally
    % - it measures overlap, not just a difference in means: one topic sentence that
    %   scores low loses many of its 12 comparisons and pulls the AUC down, which a
    %   comparison of the two means would hide
    % - it needs no cut-off score (AUC = the average over all possible thresholds)
    % 1 = every topic sentence outscores every other sentence; 0.5 = chance (random
    % mixing). Unlike accuracy, the 4-vs-12 imbalance doesn't matter (calling everything
    % "not topic t" would already give 75% accuracy), and only ranks count, so the shared
    % offset in every score has no effect. (Same as the area under the ROC curve, or the
    % Mann-Whitney U divided by 4 × 12.)
    % Limits: steps of 1/48 ≈ 0.02, and one misranked sentence can cost up to 12 pairs
    % (0.25); AUC says whether the groups overlap, not how far apart they are (panel a).
    auc = zeros(nL + 1, nT);
    for t = 1:nT
        for L = 0:nL                                         % each point: array index L + 1 (1 = emb)
            pos = S(g == t, L + 1, t);  neg = S(g ~= t, L + 1, t);   % [4 × 1], [12 × 1]  scores on the topic-t direction
            % pos > neg' compares every pos value with every neg value (neg' is a 1 × 12 row):
            % a [4 × 12] true/false table, true where topic sentence a outscores other sentence b.
            % mean(..., 'all') = the fraction of the 48 pairs won. Ties count as losses (>).
            auc(L + 1, t) = mean(pos > neg', 'all');
            % No readout where the class means don't differ at all (f = 0 up to rounding), e.g.
            % emb: topic sentences all end in "." at the same position (identical vectors);
            % sentiment twins share their last token and position (identical class means).
            fAll = mean(X(:, g == t, L + 1), 2) - mean(X(:, g ~= t, L + 1), 2);
            if norm(fAll) < 1e-6 * mean(vecnorm(X(:, :, L + 1))), auc(L + 1, t) = NaN; end
        end
    end
    % Selectivity index: the same information rescaled so that chance is 0.
    %   SI = 2·AUC - 1:  +1 = every topic sentence outscores every other sentence,
    %   0 = chance (random mixing), -1 = every topic sentence scores lower (the direction
    %   built from the other sentences points the wrong way for a new one).
    % It is the rank-biserial correlation (Somers' D), and the ROC-based selectivity
    % index used for single neurons (preferred vs non-preferred stimuli).
    % Chance range (pure-noise simulation, middle 90%): about -0.76 to +0.80 for 4 vs 12
    % sentences (topic), -0.62 to +0.59 for 10 vs 10 (sentiment).
    SI = 2 * auc - 1;                                        % [layers × topics]
    if ck == 1, SIall = NaN(nL + 1, nT, nC); end           % [layer × topic × checkpoint]
    SIall(:, :, ck) = SI;
    if nC > 1                                                % one line per checkpoint
        fprintf('%-26s SI at layer %s: %s\n', ckNames{ck}, layerNames{iShow}, sprintf('%6.2f', SI(iShow, :)));
    end
end                                                      % (checkpoints)
w = max(9, max(cellfun(@numel, tNames)) + 2);           % column width: fits the longest class name
fprintf('Held-out selectivity index SI = 2*AUC - 1 (one topic vs the rest), %s, position ''%s''\nlayer  %s\n', ckNames{end}, position, sprintf(sprintf('%%%ds', w), tNames{:}));
for L = 0:nL, fprintf('%-5s  %s\n', layerNames{L + 1}, sprintf(sprintf('%%%d.2f', w), SI(L + 1, :))); end

%% Step 4 — Same direction across depth? cos between one topic's DIM directions at every pair of layers
% Is the topic carried by the same direction at every layer, or does it change?
% Nothing is left out here: this describes the direction itself (not a readout),
% so f is built from all sentences at each layer.
% Comparing directions across layers is meaningful because every layer reads and
% writes the same d-unit residual stream (unlike small vs XL: different spaces).
% Rotation is not forgetting: the stream only adds, so
%   f_(L+1) = f_L + [mean write for topic-t sentences - mean write for the rest];
% the old part stays, but later layers write larger topic differences (the stream's
% norm grows), so the old part becomes a smaller share and the cosine falls.
% Caveat: each f comes from 4 vs 12 sentences, so sentence-specific noise alone keeps
% cosines below 1, even between neighbouring layers.
tt = find(strcmp(tNames, target));                       % number of the target topic (weather = 3)
F = squeeze(mean(X(:, g == tt, :), 2) - mean(X(:, g ~= tt, :), 2));   % [d × layers]  DIM direction per layer (all sentences); mean(..., 2) averages over sentences, keeps layers
CF = cosine_sim(F);                                      % [layers × layers]  CF(i, j) = cos(f at layer i, f at layer j): 1 = same direction, 0 = orthogonal
CF(:, isnan(auc(:, tt))) = NaN;  CF(isnan(auc(:, tt)), :) = NaN;   % blank out points with identical vectors (emb: f = 0)
fprintf('cos(f_%s at layer 0, layers 1..%d): %s\n', target, nL - 1, mat2str(round(CF(2, 3:end), 2)));

%% Step 5 — Plot
close(findall(groot, 'Type', 'figure', 'Tag', 'run_DIM'));
figure('Color', 'w', 'Position', [60 150 1650 + 1100 * (nC > 1), 470], 'Tag', 'run_DIM');
tl = tiledlayout(1, 3 + 2 * (nC > 1), 'TileSpacing', 'compact');
ht = title(tl, sprintf('%s, position ''%s''', ckNames{end}, position), 'Interpreter', 'none');
ht.FontSize = ht.FontSize + 2;                           % 2 pt larger than the default
cols = lines(nT);
nexttile;  hold on                                       % a. held-out scores at one layer
sc = S(:, iShow, tt);
for t = 1:nT
    plot(t + 0.1 * randn(sum(g == t), 1), sc(g == t), 'o', 'Color', cols(t, :), 'MarkerFaceColor', cols(t, :));
end
hold off;  grid on;  xlim([0.5 nT + 0.5]);  xticks(1:nT);  xticklabels(tNames);
ylabel(sprintf('held-out score x · f / ||f||  (f = %s DIM)', target));
title({sprintf('a. layer %s: projection onto the %s direction', layerNames{iShow}, target), ...
       sprintf('each sentence left out of its own f; SI = %.2f', SI(iShow, tt))});
nexttile;  hold on                                       % b. selectivity index by layer, every topic
% With two classes, "class 2 vs the rest" is the mirror of "class 1 vs the rest"
% (f_2 = -f_1, the same pairs compared the other way round), so SI is identical:
% draw one line only.
nPlot = nT;  legNames = tNames;
if nT == 2, nPlot = 1;  legNames = {sprintf('%s vs %s', tNames{1}, tNames{2})}; end
for t = 1:nPlot, plot(-1:nL - 1, SI(:, t), '-o', 'Color', cols(t, :), 'LineWidth', 1.5, 'MarkerFaceColor', cols(t, :)); end
hold off;  grid on;  yline(0, 'k--', 'LineWidth', 1);  ylim([-1 1.05]);   % dashed: SI = 0 (chance)
xticks(xt - 2);  xticklabels(layerNames(xt));  xtickangle(0);   % layer L at x = L, emb at -1
xlabel('Layer');  ylabel('held-out selectivity index SI = 2·AUC - 1');
legend(legNames, 'Location', 'southeast');  title({'b. selectivity by layer (held out)', '(0 = chance, +1 = perfect, -1 = reversed)'});
nexttile;  imagesc(CF, 'AlphaData', ~isnan(CF));  axis square;  axis xy;  clim([0 1]);   % layers increase upward; blank = no direction
colormap(gca, [linspace(1, 0.42, 64)', linspace(1, 0.62, 64)', linspace(1, 0.92, 64)']);   % white (0) to blue (1)
cb = colorbar;  cb.Label.String = 'cosine';
xticks(xt);  xticklabels(layerNames(xt));  yticks(xt);  yticklabels(layerNames(xt));  xtickangle(0);
xlabel('Layer');  ylabel('Layer');
title({sprintf('c. cosine between the %s directions of two layers', target), '(1 = same direction)'});
if nC > 1                                                % d, e. selectivity across training
    x = ckSteps;  x(x == 0) = 0.5;                       % log axis: step 0 drawn at 0.5 (labelled 0)
    xl = [0.4, max(x) * 1.5];
    % d. SI at layer showL vs training (steps below, tokens above)
    ax = nexttile;  hold on
    for t = 1:nPlot, plot(x, squeeze(SIall(iShow, t, :)), '-o', 'Color', cols(t, :), 'MarkerFaceColor', cols(t, :), 'MarkerSize', 3, 'LineWidth', 1.2); end
    if nPlot > 1, plot(x, squeeze(mean(SIall(iShow, 1:nPlot, :), 2)), 'k-', 'LineWidth', 2.5); end
    hold off;  grid on;  yline(0, 'k--', 'LineWidth', 1);  ylim([-1 1.05]);
    step_token_axis(ax, xl, tokPerStep);
    ylabel(sprintf('held-out SI at layer %s', layerNames{iShow}));
    title({sprintf('d. selectivity at layer %s across training', layerNames{iShow}), 'thick black = mean over classes', '', ''});
    if nPlot > 1, legend([legNames, {'mean'}], 'Location', 'southeast'); else, legend(legNames, 'Location', 'southeast'); end
    % e. mean SI over classes, every layer x every checkpoint. Each checkpoint is a column
    % centred on its (log) step; column edges halfway between neighbours on the log axis.
    ax = nexttile;
    C = squeeze(mean(SIall(:, 1:nPlot, :), 2));          % [layers × checkpoints]
    lx = log10(x);  e = [lx(1) - 0.15, (lx(1:end - 1) + lx(2:end)) / 2, lx(end) + 0.15];   % column edges (log10 steps)
    pcolor(10 .^ e, -1.5:nL - 0.5, [C, NaN(nL + 1, 1); NaN(1, nC + 1)]);  shading flat   % layer L at y = L, emb at -1
    bwr = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
    colormap(ax, bwr);  clim([-1 1]);  cb = colorbar;  cb.Label.String = 'held-out SI (mean over classes)';
    yticks(xt - 2);  yticklabels(layerNames(xt));  ylabel('Layer');
    step_token_axis(ax, [10 ^ e(1), 10 ^ e(end)], tokPerStep);
    title({'e. selectivity at every layer across training', 'mean over classes; blank = no readout', '', ''});
end
