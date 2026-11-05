% run_patching.m
% Activation patching (causal tracing): which components carry the information
% that makes the clean prompt give its answer?
%   1. tokenize, load weights, build the corrupted run(s)
%   2. clean and corrupted runs; measure the logit difference
%      LD = logit(target) - logit(competitor), p(target) and the target's rank
%   3. for every component (layer x position, or head), copy its activation from
%      one run into the other and measure how far the answer moves
%   4. plot heatmaps
%
% corruption (how the corrupted run is made):
%   'counterfactual'  a second prompt, aligned token for token (corrupt_prompt),
%                     e.g. "The Golden Gate Bridge is in downtown". Also called
%                     symmetric token replacement; recommended by Zhang & Nanda 2024.
%   'gaussian_noise'  the clean prompt with Gaussian noise ADDED once to the
%                     subject's token embeddings (emb, the stream entering layer 0,
%                     subject positions only), as in Meng et al. 2022 (ROME). Noise std =
%                     noise_scale × σ, σ = std of all W_E entries (Meng: 3σ, with σ
%                     measured on subject embeddings). Averaged over nSamples noise
%                     draws: every run is repeated per sample (same noise in the
%                     corrupted run and its patched runs), and the LD / p values are
%                     averaged before scoring. No second prompt is needed.
%
% direction:
%   'denoise'  run the CORRUPTED input, copy in ONE component from the clean run.
%              score = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
%              1 = this component alone restores the clean answer (sufficient)
%   'noise'    run the CLEAN prompt, copy in ONE component from the corrupted run.
%              score = (LD_clean - LD_patched) / (LD_clean - LD_corrupt)
%              1 = swapping this component alone breaks the clean answer (necessary)
%   'both'     run both directions (one figure each), and report how similar
%              the two maps are
% Default 'denoise' (the standard direction; less affected by backup
% components; the only one Meng et al. used). Use 'both' when a result matters:
% if the two maps differ for a component, it has backups (low noise, high
% denoise) or only works together with others (high noise, low denoise).
%
% metric (what is plotted; all are always computed, from the same runs):
%   one figure per metric, side by side; the first title line of each figure
%   names its metric. Default {'rank', 'logit_diff'}; 'all' = all four below.
%   'logit_diff'  scores from LD, as above (default; recommended by Zhang & Nanda 2024)
%   'prob'        the same scores from p(target) instead of LD, e.g. denoise:
%                 (p_patched - p_corrupt) / (p_clean - p_corrupt), as in Meng et al.
%   'rank'        log rank of the target, normalised: denoise
%                 (log r_corrupt - log r_patched) / (log r_corrupt - log r_clean),
%                 1 = back to the clean rank, 0 = still at the corrupted rank
%   'top1'        yes/no: denoise 1 = the target becomes the top prediction again;
%                 noise 1 = the target is no longer the top prediction
%                 (with gaussian_noise: the fraction of noise samples where it does)
% results.<direction>.<resid/mlp/attn/heads> hold the LD scores (resid row L+1 = the
% stream entering layer L, i.e. the point emb, 0, ..., nL-2; see layer_points);
% results.<direction>.prob.<...> the p(target) scores, .top1.<...> the top-1 maps,
% .rank.<...> the target's rank after each patch (1 = top; 3rd dim = noise sample).
% results.info: the clean / corrupted values and settings.
%
% Speed-up: a patched run is identical to its base run up to the first patched
% layer, so it starts there from the base run's residual stream (model_forward's
% start input) instead of recomputing the layers before it.
%
% Needs: model_tokenize, model_forward, patch_activations, model_token, model_weights_file, layer_points.

%% Settings
model          = 'gpt2-small';   % 'gpt2-small' or 'gpt2-xl' (XL: ~5 s per run, a full sweep takes hours)
corruption     = 'counterfactual';   % 'counterfactual' or 'gaussian_noise'

clean_prompt   = 'The capital of France is the city of';
target         = ' Paris';        % the clean answer ('' = clean run's top token)
competitor     = ' Rome';         % the corrupted answer, for LD ('' = corrupted run's top token other than the target)

% only apply when corruption = 'counterfactual'
corrupt_prompt = 'The capital of Italy is the city of';   % same number of tokens as clean_prompt

layer_window = 0;   % attention/MLP writes only: patch layers L-window ... L+window together (0 = layer L alone; default).
                    % Meng et al. restored 10 layers at once (~ layer_window = 5); Zhang & Nanda caution that windows blur which layer contributes.

direction      = 'denoise';    % 'denoise' (default), 'noise' or 'both'

% only apply when corruption = 'gaussian_noise'
subject        = ' Eiffel Tower'; % the words to noise, as they appear in clean_prompt (with leading space)
noise_scale    = 3;               % noise std = noise_scale × σ(W_E)   (Meng et al.: 3)
nSamples       = 10;              % noise samples to average over     (Meng et al.: 10)
seed           = 1;               % fixes the noise draws

doResid = false;     % true = sweep the residual stream entering each layer (emb, 0, ..., nL-2) x positions.
doMLP = true;       % true = sweep MLP across all layers x positions.
doAttn = true;      % true = sweep Attn across all layers x positions.
doHeads = false;    % true = also sweep all heads

token_group = [];   % extra column: these token positions patched TOGETHER in one run
                    %   []          = off
                    %   'corrupted' = the corrupted tokens (counterfactual: positions where the
                    %                 prompts differ; gaussian_noise: the subject tokens)
                    %   e.g. [2 3 4] = these 0-based positions
                    % Asks whether the whole group (e.g. the full subject) is sufficient at layer L.
                    % Like layer_window, joint patching can exceed the sum of single-token effects.

%% Plot options
metric   = 'all';   % what to plot, one figure each, side by side (default: log rank and
                    % logit difference). Any of 'rank' (log rank), 'logit_diff', 'prob' (p(target)), 'top1',
                    % alone or in a list, or 'all' for all four. All are computed and saved in results.
clim_max = 1;      % colour scale is ±clim_max; values beyond it saturate (e.g. 0.1 shows small effects)
text_min = 0.02;    % write the value in cells with |score| >= text_min

%****************************************************

%% Step 1 — Tokenize, load, build the corrupted input(s)
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W   = load(model_weights_file(model));
idC = model_tokenize(clean_prompt);
n = numel(idC);  nL = size(W.W_in, 3);  nH = size(W.W_Q, 3);  d = size(W.W_E, 1);   % 12 layers, 12 heads for small
toks = strtrim(model_token(idC));
corruption = lower(corruption);
switch corruption
    case 'counterfactual'
        idX = model_tokenize(corrupt_prompt);
        if numel(idC) ~= numel(idX)
            error('The prompts must have the same number of tokens (%d vs %d).', numel(idC), numel(idX));
        end
        corrPos = find(idC ~= idX) - 1;                % 0-based positions that differ
        nS = 1;  corruptEdit = {[]};                    % one corrupted run, no edit
        fprintf('Prompts differ at position(s) %s: %s vs %s\n', mat2str(corrPos), ...
                strjoin('"' + model_token(idC(idC ~= idX)) + '"', ' '), strjoin('"' + model_token(idX(idC ~= idX)) + '"', ' '));
    case 'gaussian_noise'
        idX = idC;                                      % same prompt, noised subject
        sub = model_tokenize(subject);  sub = sub(2:end);   % subject tokens (drop <|endoftext|>)
        start = strfind(idC, sub);
        if isempty(start), error('Subject "%s" (tokens %s) not found in the prompt.', subject, strjoin('"' + model_token(sub) + '"', ' ')); end
        corrPos = start(1) - 1 + (0:numel(sub) - 1);    % 0-based subject positions
        sigma = std(double(W.W_E(:)));
        rng(seed);
        noise = noise_scale * sigma * randn(d, numel(corrPos), nSamples, 'single');   % fixed per sample
        nS = nSamples;
        corruptEdit = arrayfun(@(s) @(name, L, v) add_noise(name, L, v, corrPos, noise(:, :, s)), 1:nS, 'UniformOutput', false);
        fprintf('Subject at position(s) %s: %s.  Noise std = %g x %.4f = %.4f, %d samples\n', mat2str(corrPos), ...
                strjoin('"' + model_token(sub) + '"', ' '), noise_scale, sigma, noise_scale * sigma, nS);
    otherwise
        error('corruption must be ''counterfactual'' or ''gaussian_noise''.');
end

%% Step 2 — Clean and corrupted runs
clean   = model_forward(idC, W);
corrupt = cell(1, nS);                                  % one corrupted run per noise sample (1 for counterfactual)
for s = 1:nS, corrupt{s} = model_forward(idX, W, corruptEdit{s}); end
tId = local_token(target, clean);
if isempty(competitor)                                  % the corrupted runs' top token other than the target
    [~, order] = sort(mean(cell2mat(cellfun(@(o) o.probs, corrupt, 'UniformOutput', false)), 2), 'descend');
    order = order(order - 1 ~= tId);  cId = order(1) - 1;
else
    cId = local_token(competitor, corrupt{1});
end
LD    = @(o) double(o.logits(tId + 1, end) - o.logits(cId + 1, end));    % logit difference at the last position
pT    = @(o) double(o.probs(tId + 1));                                    % p(target) at the last position
rankT = @(o) sum(o.logits(:, end) > o.logits(tId + 1, end)) + 1;         % target's rank (1 = top prediction)
LDc = LD(clean);  LDx = mean(cellfun(LD, corrupt));
pTc = pT(clean);  pTxs = cellfun(pT, corrupt);  pTx = mean(pTxs);
rC = rankT(clean);  rXs = cellfun(rankT, corrupt);    % target's rank, clean and corrupted (per sample)
LRc = log(rC);  LRx = mean(log(rXs));                 % log ranks, for the rank score
fprintf('Logit difference %s - %s: clean %+.2f, corrupted %+.2f\n', model_token(tId), model_token(cId), LDc, LDx);
if nS == 1
    fprintf('p(%s): clean %.2f%%, corrupted %.2f%%\n', model_token(tId), 100 * pTc, 100 * pTx);
else
    fprintf('p(%s): clean %.2f%%, corrupted %.2f%% (mean of %d samples, range %.2f-%.2f%%)\n', ...
            model_token(tId), 100 * pTc, 100 * pTx, nS, 100 * min(pTxs), 100 * max(pTxs));
end
% Metrics to plot, and the name each figure shows
allMetrics = {'logit_diff', 'prob', 'rank', 'top1'};
if (ischar(metric) || isstring(metric)) && strcmpi(metric, 'all'), plotMetrics = allMetrics;
else, plotMetrics = lower(cellstr(metric));            % one name, or a list of names
end
if ~all(ismember(plotMetrics, allMetrics))
    error('metric must be ''all'', or any of ''rank'', ''logit_diff'', ''prob'', ''top1'' (alone or in a list).');
end
if any(strcmp(plotMetrics, 'rank')) && LRx == LRc
    warning('The target has the same rank (%d) in the clean and corrupted runs, so the rank score is undefined; not plotted.', rC);
    plotMetrics(strcmp(plotMetrics, 'rank')) = [];
end
metricField = struct('logit_diff', '', 'prob', 'prob', 'rank', 'rankscore', 'top1', 'top1');   % where each lives in results
metricTitles = struct('logit_diff', sprintf('logit difference (%s - %s)', strtrim(model_token(tId)), strtrim(model_token(cId))), ...
                      'prob',       sprintf('probability p(%s)', strtrim(model_token(tId))), ...
                      'rank',       sprintf('log rank of %s (corrupted rank %s, clean %d)', strtrim(model_token(tId)), ...
                                            num2str(round(exp(LRx), 1)), rC), ...
                      'top1',       sprintf('top-1 (is %s the top prediction?)', strtrim(model_token(tId))));

% Columns of the layer x position maps: every position, plus the token group (if any)
cols = num2cell(0:n - 1);  colNames = cellstr(toks);
if ischar(token_group) || isstring(token_group)
    if ~strcmpi(token_group, 'corrupted'), error('token_group must be [], ''corrupted'' or a list of positions.'); end
    groupPos = corrPos;
else
    groupPos = token_group;
end
if ~isempty(groupPos)
    if any(groupPos < 0 | groupPos > n - 1), error('token_group positions must be between 0 and %d.', n - 1); end
    cols{end + 1} = groupPos;
    colNames{end + 1} = strjoin(cellstr(toks(groupPos + 1)), '+');
end
nCol = numel(cols);
results.info = struct('corruption', corruption, 'target', tId, 'competitor', cId, 'corrupted_positions', corrPos, ...
                      'token_group', groupPos, 'columns', {colNames}, ...
                      'LD_clean', LDc, 'LD_corrupt', LDx, 'p_clean', pTc, 'p_corrupt', pTxs, 'nSamples', nS);

%% Step 3 — Sweeps
if strcmpi(direction, 'both'), dirs = {'denoise', 'noise'}; else, dirs = {lower(direction)}; end
close(findall(groot, 'Type', 'figure', 'Tag', 'run_patching'));
for di = 1:numel(dirs)
    dir = dirs{di};
    if strcmp(dir, 'denoise')                           % run corrupted (per sample), copy from clean
        baseIds = idX;  srcs = repmat({clean}, 1, nS);  baseRuns = corrupt;  baseEdits = corruptEdit;
        score  = @(ld) (ld - LDx) / (LDc - LDx);
        scoreP = @(p) (p - pTx) / (pTc - pTx);          % the same, from p(target)
        scoreT = @(r) mean(r == 1);                     % top-1: target is the top prediction again
    else                                                % run clean, copy from corrupted (per sample)
        baseIds = idC;  srcs = corrupt;  baseRuns = repmat({clean}, 1, nS);  baseEdits = cell(1, nS);
        score  = @(ld) (LDc - ld) / (LDc - LDx);
        scoreP = @(p) (pTc - p) / (pTc - pTx);
        scoreT = @(r) mean(r > 1);                      % top-1: target is no longer the top prediction
    end
    cellRun = @(type, which, positions, firstLayer) run_cell(baseIds, W, srcs, baseRuns, baseEdits, ...
                                                             type, which, positions, firstLayer, LD, pT, rankT);
    S = struct();  S.prob = struct();  S.top1 = struct();  S.rank = struct();   % LD scores; p(target) scores; top-1; rank
    types = {'resid', 'mlp', 'attn'};  doType = [doResid, doMLP, doAttn];
    % Progress: a line every pStep layers (all for small, every 4th for XL). The
    % time left is extrapolated from the runs so far; later layers run faster
    % (the speed-up skips more layers), so early estimates are on the high side.
    nRuns = (sum(doType) * nL * nCol + doHeads * nL * nH) * nS;  done = 0;  t0 = tic;  pStep = ceil(nL / 12);
    progress = @(what, L, done) fprintf('  %-6s layer %2d/%d   %5d/%d runs   %5.1f min elapsed, ~%.0f min left\n', ...
        what, L + 1, nL, done, nRuns, toc(t0) / 60, toc(t0) / 60 * (nRuns - done) / done);
    fprintf('%s: %d runs\n', upper(dir), nRuns);
    for ti = find(doType)
        S.(types{ti}) = zeros(nL, nCol);                % [layer × position (+ token group)]
        S.prob.(types{ti}) = zeros(nL, nCol);  S.top1.(types{ti}) = zeros(nL, nCol);  S.rank.(types{ti}) = zeros(nL, nCol, nS);
        for L = 0:nL - 1
            layers = max(0, L - layer_window):min(nL - 1, L + layer_window);   % window around L
            if strcmp(types{ti}, 'resid'), layers = L; end                      % no window for the residual stream
            for c = 1:nCol                              % each position, then the token group (several positions at once)
                [ld, p, r] = cellRun(types{ti}, layers, cols{c}, layers(1));
                S.(types{ti})(L + 1, c) = score(mean(ld));
                S.prob.(types{ti})(L + 1, c) = scoreP(mean(p));
                S.rank.(types{ti})(L + 1, c, :) = r;  S.top1.(types{ti})(L + 1, c) = scoreT(r);
            end
            done = done + nCol * nS;
            if mod(L + 1, pStep) == 0 || L == nL - 1, progress(types{ti}, L, done); end
        end
    end
    if doHeads
        S.heads = zeros(nH, nL);                        % [head × layer], patched at all positions
        S.prob.heads = zeros(nH, nL);  S.top1.heads = zeros(nH, nL);  S.rank.heads = zeros(nH, nL, nS);
        for L = 0:nL - 1
            for h = 0:nH - 1
                [ld, p, r] = cellRun('head', [L h], [], L);
                S.heads(h + 1, L + 1) = score(mean(ld));
                S.prob.heads(h + 1, L + 1) = scoreP(mean(p));
                S.rank.heads(h + 1, L + 1, :) = r;  S.top1.heads(h + 1, L + 1) = scoreT(r);
            end
            done = done + nH * nS;
            if mod(L + 1, pStep) == 0 || L == nL - 1, progress('heads', L, done); end
        end
    end
    % Rank score from the stored ranks: log rank, normalised like the others and
    % averaged over noise samples. Denoise: 1 = back to the clean rank, 0 = still at
    % the corrupted rank, < 0 = pushed further down. Noise: 1 = down to the corrupted rank.
    S.rankscore = struct();
    for f = fieldnames(S.rank)'
        lr = mean(log(S.rank.(f{1})), 3);
        if strcmp(dir, 'denoise'), S.rankscore.(f{1}) = (LRx - lr) / (LRx - LRc);
        else,                      S.rankscore.(f{1}) = (lr - LRc) / (LRx - LRc); end
    end
    results.(dir) = S;                                  %#ok<SAGROW>

    %% Step 4 — Plot (one figure per metric, spread across the screen)
    % GPT-2 small: attention and MLP go in ONE panel, as in the lesson's causal
    % tracing figure: two rows per layer, in the order they run (attention, then MLP).
    % XL (more than 12 layers): 96 rows would be too dense, so attention and MLP get
    % their own panels (48 rows each), and only values >= 0.1 are written.
    Sall = S;
    for mi = 1:numel(plotMetrics)
    m = plotMetrics{mi};
    if strcmp(m, 'logit_diff'), S = Sall; else, S = Sall.(metricField.(m)); end
    if doAttn && doMLP && nL <= 12
        S.sublayers = zeros(2 * nL, nCol);              % [2·layer × position (+ token group)]
        S.sublayers(1:2:end, :) = S.attn;               % odd rows: attention of each layer
        S.sublayers(2:2:end, :) = S.mlp;                % even rows: MLP of each layer
        panels = {'resid', 'sublayers'};
        panels = panels([doResid, true]);
    else
        panels = {'resid', 'attn', 'mlp'};              % attention before MLP, the order they run
        panels = panels([doResid, doAttn, doMLP]);
    end
    if doHeads, panels{end + 1} = 'heads'; end
    textMin = text_min;  if nL > 12, textMin = max(text_min, 0.1); end
    if strcmp(m, 'top1'), textMin = Inf; end           % yes/no map: colour only, no numbers
    figHeight = 760 + 240 * (nL > 12);                  % taller for XL's 48 rows
    figWidth = max(900, 520 * numel(panels));
    scr = get(groot, 'ScreenSize');                     % spread the metrics' figures left to right
    x = 20 + (mi - 1) * max(0, scr(3) - figWidth - 40) / max(1, numel(plotMetrics) - 1);
    figure('Color', 'w', 'Position', [x, max(40, scr(4) - figHeight - 100 - 40 * (di - 1)), figWidth, figHeight], ...
           'Tag', 'run_patching', 'Name', sprintf('%s: %s', dir, metricTitles.(m)));
    tl = tiledlayout(1, numel(panels), 'TileSpacing', 'compact');
    cm = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
    fmt = '%.1f';  if textMin < 0.1, fmt = '%.2f'; end
    xlab = compose('%d: %s', (0:n - 1)', toks');
    xlab(corrPos + 1) = strcat(xlab(corrPos + 1), '*');   % mark the corrupted tokens
    if nCol > n, xlab{end + 1} = ['group: ' colNames{end}]; end   % the token group column
    for k = 1:numel(panels)
        ax = nexttile;  M = S.(panels{k});
        imagesc(M);  axis xy;  clim([-1 1] * clim_max);  colormap(ax, cm);   % axis xy: layers (and heads) increase upward
        for r = 1:size(M, 1)                            % write values >= textMin in the cells
            for c = 1:size(M, 2)
                if abs(M(r, c)) >= textMin
                    text(c, r, sprintf(fmt, M(r, c)), 'HorizontalAlignment', 'center', 'FontSize', 7, ...
                         'Color', [1 1 1] * (abs(M(r, c)) > 0.6 * clim_max));
                end
            end
        end
        step = 1 + (nL > 12);                           % label every layer (small), every 2nd (XL)
        switch panels{k}
            case 'heads'
                xticks(1:step:nL); xticklabels(0:step:nL - 1); xtickangle(0); yticks(1:nH); yticklabels(0:nH - 1);
                xlabel('Layer'); ylabel('Head'); title('Single heads (patched at all positions)');
            case 'sublayers'                            % small only: every row labelled
                yticks(1:2 * nL);
                yticklabels(reshape([compose('%d attn write', 0:nL - 1); compose('%d MLP write', 0:nL - 1)], [], 1));
                for L = 1:nL - 1, yline(2 * L + 0.5, 'Color', [0.75 0.75 0.75]); end   % separate the layers
                title('Attention and MLP writes (two rows per layer: attention below, MLP above)');
            case 'resid'                                % row L+1 = the stream entering layer L: emb, 0, ..., nL-2
                pts = layer_points(nL);
                yticks(1:step:nL); yticklabels(pts(1:step:nL)); ylabel('Layer');
                title('Residual stream');
            otherwise
                yticks(1:step:nL); yticklabels(0:step:nL - 1); ylabel('Layer');
                names = struct('mlp', 'MLP write', 'attn', 'Attention write');
                title(names.(panels{k}));
        end
        if ~strcmp(panels{k}, 'heads')
            xticks(1:nCol); xticklabels(xlab); xtickangle(30);
            set(ax, 'TickLabelInterpreter', 'none');
            if nCol > n, xline(n + 0.5, 'k', 'LineWidth', 1.5); end   % separate the token group column
        end
    end
    if strcmp(m, 'prob'), metricName = sprintf('p(%s)', strtrim(model_token(tId)));
    elseif strcmp(m, 'rank'), metricName = sprintf('log rank of %s', strtrim(model_token(tId)));
    else, metricName = 'logit difference'; end
    cb = colorbar;  cb.Layout.Tile = 'east';  cb.Label.String = sprintf('Score from %s (1 = full effect)', metricName);
    if clim_max < 1, cb.Label.String = sprintf('Score from %s (1 = full effect; saturates at ±%g)', metricName, clim_max); end
    if strcmp(m, 'top1') && strcmp(dir, 'denoise')
        cb.Label.String = sprintf('Top-1: 1 = %s is the top prediction again', strtrim(model_token(tId)));
    elseif strcmp(m, 'top1')
        cb.Label.String = sprintf('Top-1: 1 = %s is no longer the top prediction', strtrim(model_token(tId)));
    end
    if strcmp(corruption, 'counterfactual') && strcmp(dir, 'denoise')
        what = {sprintf('%s   COUNTERFACTUAL, DENOISE: run "%s"', model, corrupt_prompt), ...
                sprintf('copy in ONE component from "%s"   (1 = restores %s)', clean_prompt, model_token(tId))};
    elseif strcmp(corruption, 'counterfactual')
        what = {sprintf('%s   COUNTERFACTUAL, NOISE: run "%s"', model, clean_prompt), ...
                sprintf('copy in ONE component from "%s"   (1 = breaks %s)', corrupt_prompt, model_token(tId))};
    elseif strcmp(dir, 'denoise')
        what = {sprintf('%s   GAUSSIAN NOISE, DENOISE: run "%s" with "%s" noised (%g%s, %d samples)', ...
                        model, clean_prompt, strtrim(subject), noise_scale, char(963), nS), ...
                sprintf('restore ONE clean component   (1 = restores %s)', model_token(tId))};
    else
        what = {sprintf('%s   GAUSSIAN NOISE, NOISE: run "%s" clean', model, clean_prompt), ...
                sprintf('copy in ONE component from the run with "%s" noised (%g%s, %d samples)   (1 = breaks %s)', ...
                        strtrim(subject), noise_scale, char(963), nS, model_token(tId))};
    end
    what{end + 1} = sprintf('p(%s): clean %.1f%%, corrupted %.1f%%   (* = corrupted token)', strtrim(model_token(tId)), 100 * pTc, 100 * pTx);
    if layer_window > 0
        what{end + 1} = sprintf('Attn/MLP rows: layers L-%d ... L+%d patched together (residual stream: one point only)', layer_window, layer_window);
    end
    what = [{sprintf('METRIC: %s', metricTitles.(m))}, what];   % first line: which metric this figure shows
    title(tl, what, 'Interpreter', 'none');
    end                                                 % metrics
end

%% Step 5 — With 'both': how similar are the two directions? (each plotted metric)
% Similar maps = components act roughly independently. A component with a much
% higher denoise than noise score has backups; the reverse = it needs partners.
if numel(dirs) == 2
  for mi = 1:numel(plotMetrics)
    m = plotMetrics{mi};
    fprintf('Metric: %s\n', metricTitles.(m));
    A = results.denoise;  B = results.noise;
    if ~strcmp(m, 'logit_diff'), A = A.(metricField.(m));  B = B.(metricField.(m)); end
    extra = {'prob', 'top1', 'rank', 'rankscore'};     % keep only the maps themselves
    A = rmfield(A, intersect(fieldnames(A), extra));  B = rmfield(B, intersect(fieldnames(B), extra));
    for p = fieldnames(A)'
        a = A.(p{1});  b = B.(p{1});
        [dmax, i] = max(abs(a(:) - b(:)));
        [r, c] = ind2sub(size(a), i);
        rowName = sprintf('layer %d', r - 1);
        if strcmp(p{1}, 'resid'), pts = layer_points(nL);  rowName = ['point ' pts{r}]; end   % the stream entering layer r-1
        if strcmp(p{1}, 'heads'), where = sprintf('head %d.%d', c - 1, r - 1);
        elseif c > n, where = sprintf('%s, token group (%s)', rowName, colNames{c});
        else, where = sprintf('%s, position %d (%s)', rowName, c - 1, toks(c)); end
        cc = corrcoef(a(:), b(:));
        fprintf('  %-6s noise vs denoise: correlation %.2f, largest difference %.2f at %s (denoise %.2f, noise %.2f)\n', ...
                p{1}, cc(1, 2), dmax, where, a(i), b(i));
    end
  end
end

% -------------------------------------------------------------------------
function [ld, p, r] = run_cell(baseIds, W, srcs, baseRuns, baseEdits, type, which, positions, firstLayer, LD, pT, rankT)
% One cell of a sweep: the patched run for every noise sample (1 for counterfactual).
% Returns the logit difference, p(target) and rank of each, [1 × nS].
    nS = numel(srcs);  ld = zeros(1, nS);  p = zeros(1, nS);  r = zeros(1, nS);
    for s = 1:nS
        patch = patch_activations(srcs{s}, type, which, positions);
        if firstLayer == 0 && ~isempty(baseEdits{s})  % full run: corruption edit first (noise), then the patch
            corruptIt = baseEdits{s};
            o = model_forward(baseIds, W, @(name, l, v) patch(name, l, corruptIt(name, l, v)));
        else                                          % start at the first patched layer from the base run
            o = model_forward(baseIds, W, patch, struct('layer', firstLayer, 'resid', baseRuns{s}.resid_pre(:, :, firstLayer + 1)));
        end
        ld(s) = LD(o);  p(s) = pT(o);  r(s) = rankT(o);
    end
end

function v = add_noise(name, L, v, subjPos, noise)
% Add the fixed noise to the subject's embeddings (emb, the stream entering layer 0).
    if strcmp(name, 'resid_pre') && L == 0
        v(:, subjPos + 1) = v(:, subjPos + 1) + noise;
    end
end

function id = local_token(s, run)
% Text -> single token ID; '' = the run's top token.
    if isempty(s), [~, id] = max(run.probs); id = id - 1; return, end
    if isnumeric(s), id = s; return, end
    p = model_tokenize(s);  p = p(2:end);
    if numel(p) ~= 1, error('"%s" is %d GPT-2 tokens; use a single-token word.', s, numel(p)); end
    id = p;
end
