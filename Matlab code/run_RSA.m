% run_RSA.m
% Representational similarity analysis (RSA) of the residual stream, layer by layer,
% for one or more models.
% For each model:
%   1. load the weights; run every sentence; one vector per sentence per layer
%      (the embedding (emb), then the output of each layer)
%   2. at every layer, the representational dissimilarity matrix (RDM):
%      RDM(i, j) = 1 - Pearson r between sentence i's and sentence j's vectors
%      (signed r: 0 = same pattern, 1 = uncorrelated, 2 = opposite patterns)
%   3. a model RDM: what the RDM would look like if only topic mattered
%      (0 = same topic, 1 = different topics)
%   4. compare: Spearman rank correlation between the upper triangles of the
%      model RDM and the layer's RDM (does the layer's geometry follow topic?)
%   5. layer x layer: Spearman between every pair of layers' RDMs (does the
%      arrangement of the sentences stay the same across depth?)
%   6. plot: the RDM at one layer (a), the topic model RDM of step 3 beside it (m),
%      within- vs between-topic pairs (a2), the same
%      pairs as ranks, i.e. what rho uses (a3), the
%      within- and between-topic means across layers (b; the Spearman rho of
%      step 4 is printed), the layer x layer matrix (c)
% With checkpointRange (one model): steps 1-5 for every checkpoint; the step-6 figure shows
% the last one (a-c) and gets a second row with rho (d) and the between/within ratio (e)
% at every layer across training (steps below, tokens above).
% Then, with more than one model:
%   7. compare the models on relative depth (layers passed / number of layers; 0 = emb,
%      1 = the output of the last layer),
%      so models with different numbers of layers line up: spread (within and
%      between topics), the between / within ratio, and rho
%
% Notes:
% - Pearson r between two residual-stream vectors equals their cosine here: every
%   state is mean-zero across units (center_writing_weights). So RDM = 1 - cos.
% - The comparison uses ranks (Spearman), so it is unaffected by any monotonic
%   squeezing of the dissimilarities, such as the compression by the shared
%   direction (raw cosines 0.84-0.97); no centering is needed.
% - Raw 1 - r differs far more between models than between topics: compare
%   models with the relative measures (ratio, rho).
% - Only the chosen sentences are visible: the RDM describes how THESE inputs are
%   arranged, not the whole representation.
%
% Needs: model_tokenize, model_forward, model_weights_file, model_checkpoints, token_positions, layer_points,
%        step_token_axis, prompts_topic.

%% Settings
models = {'gpt2-small'};   % one or more models; with more than one, step 7 compares them
checkpointRange = [];         % [] = the trained model(s); otherwise ONE model's training checkpoints on disk at these
                              % steps, e.g. 0:2000 or 0:400000 (all): steps 1-5 are repeated for each; the figure of
                              % step 6 shows the last checkpoint (a-c), with a second row showing rho (d) and the between/within
                              % ratio at every layer across training (steps below, tokens above).
                              % On disk: gpt2-small, gpt2-small-battlestar (Stanford CRFM seeds alias x21, battlestar x49), pythia-160m, pythia-160m-seed1, pythia-160m-seed2 (PolyPythias); see model_checkpoints.
showL  = 6;                   % per model, RDM heatmap: the stream leaving layer showL (0-based), or 'emb'
position = 'last word';            % token read: 'last', 'last word' (last non-punctuation token) or 'mean' (see token_positions)
                              % Note: Pythia's final "." is an attention sink (massive activations, almost no sentence
                              % information), so with Pythia last-token results are erratic when sentences end in "."; use 'last word'.
% Sentences: 16 in 4 topics, all 7 tokens, from prompts_topic (loaded below). The
% plots assume this layout (4 topics x 4 sentences).

%****************************************************

addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
[sentences, topics] = prompts_topic();                   % {1 × 16} sentences and their topic labels
close(findall(groot, 'Type', 'figure', 'Tag', 'run_RSA'));
nS = numel(sentences);
Res = struct('model', {}, 'nL', {}, 'mW', {}, 'mB', {}, 'rho', {});   % per-model (or per-checkpoint) results
isCk = ~isempty(checkpointRange);
if ~isCk                                                 % the trained model(s)
    files = cellfun(@model_weights_file, models, 'UniformOutput', false);  runNames = models;
else                                                     % one model's checkpoints
    if numel(models) > 1, error('checkpointRange: give one model in models, not a list.'); end
    [files, ckSteps, tokPerStep, runNames] = model_checkpoints(models{1}, checkpointRange);
end
nR = numel(files);
for m = 1:nR
model = runNames{m};
showFig = ~isCk || m == nR;                              % checkpoints: tables and figure for the last one only
sL = showL;  if isnumeric(showL), sL = showL(min(m, numel(showL))); end   % point for the RDM heatmap
if showFig, fprintf('\n=== %s ===\n', model); end

%% Step 1 — Load; one vector per sentence per layer (at the token set by position)
W = load(files{m});
nL = size(W.W_in, 3);  d = size(W.W_E, 1);
X = zeros(d, nS, nL + 1);                                % [d × sentence × (emb, layers 0..nL-1)]
for s = 1:nS
    ids = model_tokenize(sentences{s}, W);               % the tokenizer follows the weights
    o = model_forward(ids, W);
    p = token_positions(ids, W, position);               % the token(s) read (several for 'mean')
    X(:, s, 1:nL)  = mean(o.resid_pre(:, p, :), 2);      % emb, then the output of layers 0..nL-2
    X(:, s, nL + 1) = mean(o.resid_post(:, p, end), 2);  % the output of the last layer
end

%% Step 2 — RDM at every layer: 1 - Pearson r between every pair of sentences
RDM = zeros(nS, nS, nL + 1);
for L = 0:nL                                             % each point: array index L + 1 (1 = emb, k + 2 = layer k)
    RDM(:, :, L + 1) = 1 - corr(X(:, :, L + 1));         % [nS × nS]  corr correlates the columns (sentences)
end

%% Step 3 — Model RDM: only topic matters
[~, ~, g] = unique(topics, 'stable');
modelRDM = double(g ~= g');                              % [nS × nS]  0 = same topic, 1 = different

%% Step 4 — Compare each layer's RDM with the model RDM (Spearman, upper triangle)
up = triu(true(nS), 1);                                  % each pair once, no diagonal
rho = zeros(nL + 1, 1);  mW = rho;  mB = rho;  sW = rho;  sB = rho;
for L = 0:nL                                             % each point: array index L + 1 (1 = emb)
    R = RDM(:, :, L + 1);
    rho(L + 1) = corr(R(up), modelRDM(up), 'Type', 'Spearman');
    if max(R(up)) < 1e-6, rho(L + 1) = NaN; end          % all vectors identical (e.g. emb, same last token): RDM is only rounding noise
    w = R(up & modelRDM == 0);  b = R(up & modelRDM == 1);   % within-topic (24) and between-topic (96) pairs
    mW(L + 1) = mean(w);  sW(L + 1) = std(w) / sqrt(numel(w));   % mean and SEM, for panel b
    mB(L + 1) = mean(b);  sB(L + 1) = std(b) / sqrt(numel(b));
end
[layerNames, iShow] = layer_points(nL, 'out', sL);       % {'emb', '0', ..., 'nL-1'}; iShow = array index of sL
if nL > 24, step = 4; elseif nL > 12, step = 2; else, step = 1; end   % label every layer (small), every 4th (XL)
if showFig
    fprintf('layer   within   between   Spearman rho (layer RDM vs topic model RDM)\n');
    for L = 0:nL, fprintf('  %-4s  %6.3f   %6.3f    %5.2f\n', layerNames{L + 1}, mW(L + 1), mB(L + 1), rho(L + 1)); end
end
if isCk                                                  % one line per checkpoint
    fprintf('%-26s rho by layer: %s\n', model, mat2str(round(rho', 2)));
end
Res(m) = struct('model', model, 'nL', nL, 'mW', mW, 'mB', mB, 'rho', rho);

%% Step 5 — Layer x layer: compare every layer's RDM with every other layer's (Spearman)
% Do the sentences keep the same arrangement relative to each other across depth?
% (Second-order similarity: it can stay high even while every vector rotates.)
V = reshape(RDM, nS * nS, nL + 1);                       % [nS² × layers]  each column = one layer's RDM
V = V(up(:), :);                                         % [120 × layers]  upper triangles only
RR = corr(V, 'Type', 'Spearman');                        % [layers × layers]
RR(:, isnan(rho)) = NaN;  RR(isnan(rho), :) = NaN;       % points with identical vectors (no RDM)
if showFig, fprintf('Layer x layer RDM similarity (Spearman): layer 0 vs 1..%d: %s\n', nL - 1, mat2str(round(RR(2, 3:end), 2))); end

%% Step 6 — Plot (one figure per model; with checkpoints, the last one)
if showFig
figure('Color', 'w', 'Position', [40 80 2700, 480 + 450 * isCk], 'Tag', 'run_RSA');
tl = tiledlayout(1 + isCk, 6, 'TileSpacing', 'compact');   % top: a, m, a2, a3, b, c; (bottom: d, e across training)
ht = title(tl, sprintf('%s, position ''%s''', model, position), 'Interpreter', 'none');  ht.FontSize = ht.FontSize + 2;
lab = compose('%s %d', string(topics'), repmat((1:4)', 4, 1));   % 'food 1', 'food 2', ...
edges = 4.5:4:nS;                                        % lines between topics
R = RDM(:, :, iShow);  cl = [0 max(R(:))];        % same colour range in a and a2
blues = [linspace(1, 0.42, 64)', linspace(1, 0.62, 64)', linspace(1, 0.92, 64)'];   % panel a: white (0) to blue
nexttile;  imagesc(R);  axis square;  clim(cl);  colormap(gca, blues);  cb = colorbar;  cb.Label.String = '1 - r';
for b = edges, xline(b, 'k', 'LineWidth', 1);  yline(b, 'k', 'LineWidth', 1); end
tNames = unique(topics, 'stable');                       % one label per topic, centred on its block
set(gca, 'XTick', 2.5:4:nS, 'XTickLabel', tNames, 'YTick', 2.5:4:nS, 'YTickLabel', tNames, 'TickLength', [0 0]);
title({sprintf('a. RDM, layer %s: 1 - r', layerNames{iShow}), '(0 = same pattern, 1 = uncorrelated, 2 = opposite)'});

% m: the topic model RDM (step 3), the hypothesis every layer's RDM is compared with (rho)
nexttile;  imagesc(modelRDM);  axis square;  clim([0 1]);  colormap(gca, blues);
cb = colorbar;  cb.Ticks = [0 1];  cb.TickLabels = {'same', 'different'};
for b = edges, xline(b, 'k', 'LineWidth', 1);  yline(b, 'k', 'LineWidth', 1); end
set(gca, 'XTick', 2.5:4:nS, 'XTickLabel', tNames, 'YTick', 2.5:4:nS, 'YTickLabel', tNames, 'TickLength', [0 0]);
title({'m. topic model RDM', sprintf('(0 = same topic, 1 = different; \\rho at layer %s = %.2f)', layerNames{iShow}, rho(iShow))});

% a2: the pairs in a, split into within-topic and between-topic (each pair once)
win = R(up & modelRDM == 0);                             % [24 × 1]  same topic
btw = R(up & modelRDM == 1);                             % [96 × 1]  different topics
nexttile;  hold on
bar(1:2, [mean(win), mean(btw)], 0.6, 'FaceColor', [0.8 0.8 0.8]);
plot(1 + 0.15 * randn(size(win)), win, '.', 'Color', [0 0.45 0.74], 'MarkerSize', 10);   % each pair
plot(2 + 0.15 * randn(size(btw)), btw, '.', 'Color', [0.85 0.33 0.10], 'MarkerSize', 10);
errorbar(1:2, [mean(win), mean(btw)], [std(win) / sqrt(numel(win)), std(btw) / sqrt(numel(btw))], 'k.', 'LineWidth', 1.5);
hold off;  grid on;  xlim([0.4 2.6]);  ylim(cl);
xticks(1:2);  xticklabels({sprintf('within topic (%d pairs)', numel(win)), sprintf('between (%d pairs)', numel(btw))});
ylabel('1 - r');
title({sprintf('a2. layer %s: within vs between topics', layerNames{iShow}), sprintf('mean %.3f vs %.3f (± SEM; dots = pairs)', mean(win), mean(btw))});

% a3: the same pairs as RANKS among all of them (1 = most similar), which is what rho uses.
% The topic model has only two values (0 within, 1 between), so Spearman's rho reduces to
% the difference between the two groups' mean ranks, rescaled:
%   rho = (mean rank between - mean rank within) * sqrt(p*q) / sd(ranks)
% p, q = fractions of between and within pairs (96/120, 24/120); sd(ranks) = sqrt((n^2-1)/12)
% for n pairs without ties. Even perfect separation (all within pairs ranked 1-24) gives a
% difference of only n/2, so rho cannot exceed (n/2)*sqrt(p*q)/sd(ranks) ~ sqrt(3pq) = 0.69.
v = R(up);  isB = modelRDM(up) == 1;  n = numel(v);    % the 120 pairs and which are between-topic
rk = tiedrank(v);                                        % rank of each pair's 1 - r (1 = most similar)
p = mean(isB);  q = 1 - p;
rW = rk(~isB);  rB = rk(isB);                            % within- and between-topic ranks
rhoF = (mean(rB) - mean(rW)) * sqrt(p * q) / std(rk, 1); % the formula above (std(rk, 1) = population sd)
rhoMax = (n / 2) * sqrt(p * q) / std(rk, 1);             % the ceiling: perfect separation
nexttile;  hold on
bar(1:2, [mean(rW), mean(rB)], 0.6, 'FaceColor', [0.8 0.8 0.8]);
plot(1 + 0.15 * randn(size(rW)), rW, '.', 'Color', [0 0.45 0.74], 'MarkerSize', 10);   % each pair
plot(2 + 0.15 * randn(size(rB)), rB, '.', 'Color', [0.85 0.33 0.10], 'MarkerSize', 10);
plot([1 2], [mean(rW), mean(rB)], 'k-', 'LineWidth', 1.5);   % the difference in mean ranks that rho rescales
yline((n + 1) / 2, 'k:', 'mid rank');                    % where both means sit without topic structure
hold off;  grid on;  xlim([0.4 2.6]);  ylim([0 n + 1]);
xticks(1:2);  xticklabels({sprintf('within topic (%d pairs)', numel(rW)), sprintf('between (%d pairs)', numel(rB))});
ylabel(sprintf('rank of 1 - r among the %d pairs (1 = most similar)', n));
title({sprintf('a3. layer %s: the same pairs as ranks (what \\rho uses)', layerNames{iShow}), ...
       sprintf('mean rank %.1f vs %.1f  \\rightarrow  \\rho = %.2f (corr: %.2f; max %.2f)', mean(rW), mean(rB), rhoF, rho(iShow), rhoMax)});
nexttile;  hold on
errorbar(-1:nL - 1, mW, sW, '-o', 'Color', [0 0.45 0.74], 'LineWidth', 1.5, 'MarkerFaceColor', [0 0.45 0.74]);
errorbar(-1:nL - 1, mB, sB, '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'MarkerFaceColor', [0.85 0.33 0.10]);
hold off;  grid on
xt = 1:step:nL + 1;  if xt(end) ~= nL + 1, xt(end + 1) = nL + 1; end   % tick positions (array index); always label the last layer
xticks(xt - 2); xticklabels(layerNames(xt)); xtickangle(0);   % layer L at x = L, emb at -1
xlabel('Layer');  ylabel('mean 1 - r (± SEM)');
legend({'within topic (24 pairs)', 'between topics (96 pairs)'}, 'Location', 'northwest');
title({'b. within vs between topics across layers', '(the gap matters: the overall level follows the shared direction)'});
nexttile;  imagesc(RR, 'AlphaData', ~isnan(RR));  axis square;  axis xy;   % layers increase upward; blank = no RDM
clim([floor(10 * min(RR(:))) / 10, 1]);  colormap(gca, parula);   % colour range = the data's range (values are all well above 0)
cb = colorbar;  cb.Label.String = 'Spearman \rho';
xticks(xt); xticklabels(layerNames(xt)); yticks(xt); yticklabels(layerNames(xt)); xtickangle(0);
xlabel('Layer');  ylabel('Layer');
title({'c. RDM of one layer vs another (Spearman)', '1 = same arrangement'});
% d, e (checkpoints only): rho and the between/within ratio at every layer (rows) x every
% checkpoint (columns); columns centred on the (log) training step, edges halfway between
% neighbours; steps below, tokens above. Res holds every checkpoint by now (this is the last).
if isCk
    x = ckSteps;  x(x == 0) = 0.5;                       % log axis: step 0 drawn at 0.5 (labelled 0)
    lx = log10(x);  e = [lx(1) - 0.15, (lx(1:end - 1) + lx(2:end)) / 2, lx(end) + 0.15];   % column edges (log10 steps)
    maps = {[Res.rho], [Res.mB] ./ [Res.mW]};            % [layers × checkpoints]
    cbLab = {'Spearman \rho (layer RDM vs topic model RDM)', 'between / within (mean 1 - r)'};
    ttl = {'d. topic structure (\rho) across training', 'e. between / within ratio across training'};
    for k = 1:2
        ax = nexttile(tl, 6 + k);
        pcolor(10 .^ e, -1.5:nL - 0.5, [maps{k}, NaN(nL + 1, 1); NaN(1, nR + 1)]);  shading flat   % layer L at y = L, emb at -1
        colormap(ax, parula);  cb = colorbar;  cb.Label.String = cbLab{k};
        if k == 1, clim([min(0, min(maps{1}(:))) max(maps{1}(:))]); end
        yticks(xt - 2);  yticklabels(layerNames(xt));  ylabel('Layer');
        step_token_axis(ax, [10 ^ e(1), 10 ^ e(end)], tokPerStep);
        title({ttl{k}, 'blank = no RDM (identical vectors)', '', ''});
    end
end
end                                                      % (showFig)
end

%% Step 7 — Compare the models on relative depth (only with more than one model, not checkpoints)
if numel(Res) > 1 && ~isCk
    figure('Color', 'w', 'Position', [60 100 1600 460], 'Tag', 'run_RSA');
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact');
    title(tl, 'Models compared on relative depth (layers passed / number of layers; 1 = the last layer)');
    cols = lines(numel(Res));  names = {Res.model};
    xl = 'relative depth (layers passed / number of layers; 0 = emb, 1 = the last layer)';
    nexttile;  hold on;  h = gobjects(numel(Res), 1);
    for m = 1:numel(Res)
        dep = (0:Res(m).nL) / Res(m).nL;                 % 0 = emb, 1 = the output of the last layer
        h(m) = plot(dep, Res(m).mW, '-', 'Color', cols(m, :), 'LineWidth', 1.5);   % within topic: solid
        plot(dep, Res(m).mB, '--', 'Color', cols(m, :), 'LineWidth', 1.5);        % between topics: dashed
    end
    hold off;  grid on;  xlabel(xl);  ylabel('mean 1 - r');
    legend(h, names, 'Location', 'northwest', 'Interpreter', 'none');
    title({'a. spread of the sentences', 'solid = within topic, dashed = between'});
    nexttile;  hold on
    for m = 1:numel(Res)
        dep = (1:Res(m).nL) / Res(m).nL;                 % from layer 0 (emb can be all identical)
        plot(dep, Res(m).mB(2:end) ./ Res(m).mW(2:end), '-o', 'Color', cols(m, :), 'LineWidth', 1.5, 'MarkerSize', 3);
    end
    hold off;  grid on;  yline(1, 'k:');  xlabel(xl);  ylabel('between / within');
    legend(names, 'Location', 'southwest', 'Interpreter', 'none');
    title('b. topic separation: between / within ratio');
    nexttile;  hold on
    for m = 1:numel(Res)
        plot((0:Res(m).nL) / Res(m).nL, Res(m).rho, '-o', 'Color', cols(m, :), 'LineWidth', 1.5, 'MarkerSize', 3);
    end
    hold off;  grid on;  ylim([0 0.7]);  xlabel(xl);  ylabel('Spearman \rho');
    legend(names, 'Location', 'southeast', 'Interpreter', 'none');
    title('c. layer RDM vs topic model RDM');
end
