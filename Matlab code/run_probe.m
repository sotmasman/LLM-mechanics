% run_probe.m
% Linear probe (logistic regression) as a readout, layer by layer, compared with DIM.
% Same sentence sets, held-out design and plots as run_DIM, but the direction is
% LEARNED instead of being the difference of the two class means.
%   0. turn the hand-typed labels into numbers g (one per sentence)
%   1. load the weights; run every sentence; one vector per sentence per layer
%      (the embedding (emb), then the output of each layer)
%   2. for each class, at every layer: leave one sentence (and its group) out, train
%      a probe on the others (class vs the rest), score the left-out sentence:
%      score = x · w / ||w||  (projection onto the probe direction w)
%      the same for the DIM direction f (as in run_DIM), and for the probe trained
%      on shuffled labels (chance control)
%   3. separation of the held-out scores: AUC over all (class, other) pairs, then the
%      selectivity index SI = 2·AUC - 1 (0 = chance, +1 = perfect)
%   4. do probe and DIM agree? cos(w, f) at every layer (both from all sentences)
%   5. plot: held-out probe scores at one layer (a), SI by layer for probe, DIM and
%      shuffled labels (b), cos(w, f) by layer (c); with checkpointRange, steps 1-3 are
%      repeated for every checkpoint; (d) SI at layer showL and (e) the probe's SI at every
%      layer (heatmap) vs training, steps below and tokens above (a-c: the last checkpoint)
%
% Notes:
% - Probe: P(class | x) = sigmoid(w · x + b), fitted by fitclinear with a ridge (L2)
%   penalty lambda. DIM is f = mu_A - mu_B; a probe is closer to Sigma^-1 (mu_A - mu_B),
%   i.e. DIM corrected for the noise within each class. A strong penalty pulls w
%   toward the DIM direction; a weak one lets it use noise-cancelling directions.
% - With d = 768 (1600 for XL) units and only 16-48 sentences, a probe can separate
%   almost any labelling of the training sentences, even random ones. That is why the
%   scores are held out and why the shuffled-label control is plotted.
% - Each layer's vectors are divided by their average length (one number per layer)
%   so that lambda means the same at every layer; this doesn't change any direction.
%
% Needs: cosine_sim, model_tokenize, model_forward, model_weights_file, layer_points, prompts_* functions,
%        fitclinear (Statistics and Machine Learning Toolbox).

%% Settings
model    = 'gpt2-small';
checkpointRange = [10 100 1000 10000 100000 400000];         % [] = the trained model; otherwise its training checkpoints on disk at these steps,
                              % e.g. 0:2000 or 0:400000 (all): steps 1-3 are repeated for each, panels a-c show the
                              % last checkpoint, panel d shows SI at layer showL (probe, DIM, shuffled) and panel e the
                              % probe's SI at every layer (heatmap) across training (steps below, tokens above).
                              % On disk: gpt2-small, gpt2-small-battlestar (Stanford CRFM seeds alias x21, battlestar x49), pythia-160m, pythia-160m-seed1, pythia-160m-seed2 (PolyPythias); see model_checkpoints.
contrast ='passive-role';    % which sentences and which grouping, roughly ordered from easy (top) to hard (bottom):
                              %   'topic', 'sentiment', 'strong-weak-praise', 'negation-adjective', 'negation-sentiment',
                              %   'passive-lastword', 'passive-role'   (as in run_DIM; see the examples there)
lambda   = 1e-5;              % ridge penalty: larger = closer to the DIM direction; smaller = more discriminant-like
                              % (1e-3: the probe is the DIM direction, cos(w, f) ~ 1; 1e-5: cos ~ 0.7-0.95; 1e-7: freer still)
nShuffles = 1;                % shuffled-label control: number of random labellings (0 = none; 1 = the original single shuffle;
                              % more give a chance distribution, but cost ~5 s per shuffle for 20 sentences on GPT-2 small).
                              % To do: shuffle within groups (flip labels within pairs), enumerate all flips, faster solver.
seed     = 1;                 % for the shuffles (reproducible)
target   = '';                % class for panel a ('' = the first class)
showL    = 6;                 % panel a: the stream leaving layer showL (0-based), or 'emb'
position = 'last word';            % token read: 'last', 'last word' (last non-punctuation token) or 'mean' (see token_positions)
                              % Note: Pythia's final "." is an attention sink (massive activations, almost no sentence
                              % information), so with Pythia last-token results are erratic when sentences end in "."; use 'last word'.
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

%% Step 0 — Labels as numbers (as in run_DIM)
[tNames, ~, g] = unique(topics, 'stable');  nT = numel(tNames);   % g = class number per sentence [nS × 1]
if isempty(target), target = tNames{1}; end
rng(seed);  gShuf = zeros(numel(g), nShuffles);         % shuffled labels: same class sizes, no relation to the sentences
for k = 1:nShuffles, gShuf(:, k) = g(randperm(numel(g))); end
tShuf = 1;  if nT > 2, tShuf = 1:nT; end                 % classes shuffled (with 2 classes the second mirrors the first)

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
    X = X ./ mean(vecnorm(X), 2);                            % each layer / its average length (same lambda at every layer)
    [layerNames, iShow] = layer_points(nL, 'out', showL);   % {'emb', '0', ..., 'nL-1'}; iShow = array index of showL
    if nL > 24, step = 4; elseif nL > 12, step = 2; else, step = 1; end   % label every layer (small), every 4th (XL)
    xt = 1:step:nL + 1;  if xt(end) ~= nL + 1, xt(end + 1) = nL + 1; end   % tick positions (array index); always label the last layer

    %% Step 2 — Held-out scores: probe, DIM and shuffled-label probe
    % Same logic as run_DIM's step 2: build a direction WITHOUT sentence i, then see where
    % sentence i falls along it. Three directions are built from exactly the same kept
    % sentences, so they can be compared directly (panel b):
    %   w        the probe (learned by logistic regression)
    %   f        DIM (difference of the two class means)
    %   w_shuf   the same probe trained on shuffled labels (chance control), nShuffles times
    % Cost: nT classes × (nL + 1) layers × nS sentences probe fits (passive-role, small:
    % 2 × 13 × 48 = 1,248 fits), plus the same per shuffle for the shuffled classes; DIM needs no fitting.
    Sp = zeros(nS, nL + 1, nT);  Sd = Sp;                   % [sentence × layer × class]  held-out scores:
                                                             %   Sp = probe score, Sd = DIM score
    Ss = NaN(nS, nL + 1, nT, nShuffles);                     % shuffled-label probe scores [sentence × layer × class × shuffle]
    for t = 1:nT                                             % each class (e.g. 'predator acts'), vs the rest
        for L = 0:nL                                         % each point: array index L + 1 (1 = emb, k + 2 = layer k)
            XL = X(:, :, L + 1);                             % [d × nS]  all sentences at layer L
            for i = 1:nS                                     % each sentence, left out in turn
                % Training set: everything except sentence i and its group (its set of 4 for the
                % negation and passive sets, its twin for sentiment, itself for topic), so no
                % near-identical sentence is left in to give the answer away (as in run_DIM).
                keep = groupID(:) ~= groupID(i);             % [nS × 1]
                x = XL(:, i);                                % [d × 1]  the left-out sentence

                % --- Probe: logistic regression, P(class t | x) = sigmoid(w · x + b) ---
                % fitclinear inputs:
                %   XL(:, keep)               training vectors, one column per sentence ('ObservationsIn', 'columns')
                %   double(g(keep) == t)      target: 1 = class t, 0 = the rest (one two-class probe per class)
                %   'Learner', 'logistic'     logistic regression: sigmoid squashes w · x + b into a probability
                %   'Regularization', 'ridge', 'Lambda', lambda
                %                             fit = the w, b that make the training labels most probable,
                %                             minus a penalty (lambda/2)·||w||² on large weights. Without it,
                %                             d = 768 dimensions and ~44 training sentences could be separated
                %                             perfectly in countless ways; the penalty picks the smallest w.
                %                             Strong penalty -> w close to the DIM direction; weak -> DIM
                %                             corrected for the noise within each class (~Sigma^-1 · f).
                %   'Solver', 'lbfgs'         the optimisation algorithm (standard for many dimensions, few examples)
                % Purpose: find the single direction w along which class-t training sentences sit on
                % one side and the others on the other (w is perpendicular to the best separating
                % boundary). Unlike DIM's fixed recipe (the line between the class means), the probe
                % learns w from the data and can tilt it away from unrelated variation within each
                % class. Only the kept sentences are used, so scoring sentence i next tests whether w
                % generalises. The model itself is not changed, and a readable w is not proof that
                % the model uses it (that needs steering or patching).
                mdl = fitclinear(XL(:, keep), double(g(keep) == t), 'ObservationsIn', 'columns', ...
                                 'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', lambda, 'Solver', 'lbfgs');
                w = mdl.Beta;                                % [d × 1]  the learned probe direction (the bias mdl.Bias is not needed)
                % Score = projection of the left-out sentence onto w, as for DIM. The bias b is left
                % out: it adds the same number to every score, which doesn't change their order (AUC).
                Sp(i, L + 1, t) = x' * w / norm(w);          % [1 × 1]

                % --- DIM: difference of the two class means, from the same kept sentences ---
                % Only for comparing DIM with the probe (the DIM line in panel b and the DIM column
                % of the table); not needed for the probe analysis itself. Computing it here keeps
                % the comparison exact (same training sentences, same left-out sentence) at almost
                % no cost; the same values come from run_DIM.
                f = mean(XL(:, keep & g == t), 2) - mean(XL(:, keep & g ~= t), 2);   % [d × 1]
                Sd(i, L + 1, t) = x' * f / norm(f);          % [1 × 1]

                % --- Chance control: identical probes trained on shuffled labels (step 0) ---
                % With so many dimensions a probe can fit almost any labelling; this shows what SI it
                % reaches when there is nothing real to find. (Judged against the shuffled labels in step 3.)
                if ismember(t, tShuf)
                    for k = 1:nShuffles
                        mdl = fitclinear(XL(:, keep), double(gShuf(keep, k) == t), 'ObservationsIn', 'columns', ...
                                         'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', lambda, 'Solver', 'lbfgs');
                        Ss(i, L + 1, t, k) = x' * mdl.Beta / norm(mdl.Beta);   % [1 × 1]
                    end
                end
            end
        end
    end

    %% Step 3 — Separation: AUC over all (class, other) pairs, then SI = 2·AUC - 1
    auc = @(s, lab) mean(s(lab) > s(~lab)', 'all');          % fraction of (class, other) pairs where the class sentence scores higher
    SIp = NaN(nL + 1, nT);  SId = SIp;  SIsK = NaN(nL + 1, nT, nShuffles);
    for t = 1:nT
        for L = 0:nL
            fAll = mean(X(:, g == t, L + 1), 2) - mean(X(:, g ~= t, L + 1), 2);
            if norm(fAll) < 1e-6, continue; end             % class means identical (e.g. emb): no readout
            SIp(L + 1, t) = 2 * auc(Sp(:, L + 1, t), g == t) - 1;
            SId(L + 1, t) = 2 * auc(Sd(:, L + 1, t), g == t) - 1;
            if ismember(t, tShuf)
                for k = 1:nShuffles                          % scored against the shuffled labels it was trained on
                    SIsK(L + 1, t, k) = 2 * auc(Ss(:, L + 1, t, k), gShuf(:, k) == t) - 1;
                end
            end
        end
    end
    if nT == 2, SIsK(:, 2, :) = SIsK(:, 1, :); end           % two classes: the second mirrors the first
    if nShuffles > 0
        SIs = mean(SIsK, 3);  SIs95 = prctile(SIsK, 95, 3);  SIs05 = prctile(SIsK, 5, 3);   % chance: mean and 5-95% of the shuffles
    else
        SIs = NaN(nL + 1, nT);  SIs95 = SIs;  SIs05 = SIs;   % no shuffles: no chance values
    end
    if ck == 1, SIpAll = NaN(nL + 1, nT, nC);  SIdAll = SIpAll;  SIsAll = SIpAll;  SIs95All = SIpAll; end   % [layer × class × checkpoint]
    SIpAll(:, :, ck) = SIp;  SIdAll(:, :, ck) = SId;  SIsAll(:, :, ck) = SIs;  SIs95All(:, :, ck) = SIs95;
    if nC > 1                                                % one line per checkpoint
        fprintf('%-26s SI (class 1) at layer %s: probe %5.2f  DIM %5.2f  shuffled %5.2f\n', ckNames{ck}, layerNames{iShow}, ...
                SIp(iShow, 1), SId(iShow, 1), SIs(iShow, 1));
    end
end                                                      % (checkpoints)
fprintf('Held-out SI (class 1 vs the rest), %s, %s, lambda = %g, position ''%s''; chance from %d shuffles\nlayer   probe     DIM  shuffled mean  shuffled 95%%\n', ...
        ckNames{end}, contrast, lambda, position, nShuffles);
for L = 0:nL, fprintf('%-5s  %6.2f  %6.2f  %13.2f  %12.2f\n', layerNames{L + 1}, SIp(L + 1, 1), SId(L + 1, 1), SIs(L + 1, 1), SIs95(L + 1, 1)); end

%% Step 4 — Do probe and DIM agree? cos(w, f) at every layer, both from all sentences
tt = find(strcmp(tNames, target));
cwf = NaN(nL + 1, 1);
for L = 0:nL
    if isnan(SIp(L + 1, tt)), continue; end
    mdl = fitclinear(X(:, :, L + 1), double(g == tt), 'ObservationsIn', 'columns', ...
                     'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', lambda, 'Solver', 'lbfgs');
    f = mean(X(:, g == tt, L + 1), 2) - mean(X(:, g ~= tt, L + 1), 2);
    cwf(L + 1) = cosine_sim(mdl.Beta, f);                % 1 = the probe found the DIM direction
end
fprintf('cos(probe w, DIM f) for %s by layer: %s\n', target, mat2str(round(cwf', 2)));

%% Step 5 — Plot
close(findall(groot, 'Type', 'figure', 'Tag', 'run_probe'));
figure('Color', 'w', 'Position', [60 150 1650 + 1100 * (nC > 1), 470], 'Tag', 'run_probe');
tl = tiledlayout(1, 3 + 2 * (nC > 1), 'TileSpacing', 'compact');
ht = title(tl, sprintf('%s, %s, lambda = %g, position ''%s''', ckNames{end}, contrast, lambda, position), 'Interpreter', 'none');
ht.FontSize = ht.FontSize + 2;                           % 2 pt larger than the default
cols = lines(nT);
nexttile;  hold on                                       % a. held-out probe scores at one layer
sc = Sp(:, iShow, tt);
for t = 1:nT
    plot(t + 0.1 * randn(sum(g == t), 1), sc(g == t), 'o', 'Color', cols(t, :), 'MarkerFaceColor', cols(t, :));
end
hold off;  grid on;  xlim([0.5 nT + 0.5]);  xticks(1:nT);  xticklabels(tNames);
ylabel(sprintf('held-out score x · w / ||w||  (w = %s probe)', target));
title({sprintf('a. layer %s: projection onto the %s probe direction', layerNames{iShow}, target), ...
       sprintf('each sentence left out of its own probe; SI = %.2f', SIp(iShow, tt))});
nexttile;  hold on                                       % b. SI by layer: probe, DIM, shuffled labels
plot(-1:nL - 1, SIp(:, tt), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, 'MarkerFaceColor', [0.85 0.33 0.10]);
plot(-1:nL - 1, SId(:, tt), '--o', 'Color', [0 0.45 0.74], 'LineWidth', 1.5, 'MarkerFaceColor', [0 0.45 0.74]);
legs = {'probe', 'DIM'};
if nShuffles > 0                                         % chance band (none with nShuffles = 0)
    ok = ~isnan(SIs05(:, tt));  lx = -1:nL - 1;
    fill([lx(ok), fliplr(lx(ok))], [SIs05(ok, tt)', fliplr(SIs95(ok, tt)')], [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    plot(-1:nL - 1, SIs(:, tt), '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
    legs = [legs, {sprintf('shuffled labels, 5-95%% of %d', nShuffles), 'shuffled labels, mean (chance)'}];
end
hold off;  grid on;  yline(0, 'k--', 'LineWidth', 1);  ylim([-1 1.05]);
xticks(xt - 2);  xticklabels(layerNames(xt));  xtickangle(0);   % layer L at x = L, emb at -1
xlabel('Layer');  ylabel('held-out selectivity index SI = 2·AUC - 1');
legend(legs, 'Location', 'southeast');
title({sprintf('b. %s vs the rest, by layer (held out)', target), '(0 = chance, +1 = perfect, -1 = reversed)'});
nexttile;                                                % c. agreement of the two directions
plot(-1:nL - 1, cwf, '-o', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'MarkerFaceColor', [0.3 0.3 0.3]);
grid on;  ylim([-0.2 1.05]);  yline(0, 'k--');
xticks(xt - 2);  xticklabels(layerNames(xt));  xtickangle(0);
xlabel('Layer');  ylabel('cos(w, f)');
title({sprintf('c. probe direction vs DIM direction (%s)', target), '(1 = the probe found the DIM direction)'});
if nC > 1                                                % d, e. selectivity across training
    x = ckSteps;  x(x == 0) = 0.5;                       % log axis: step 0 drawn at 0.5 (labelled 0)
    % d. SI at layer showL vs training (mean over classes): probe, DIM, shuffled labels
    ax = nexttile;  hold on
    m = @(A) squeeze(mean(A(iShow, :, :), 2));          % mean over classes at layer showL, per checkpoint
    plot(x, m(SIpAll), '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerSize', 3);
    plot(x, m(SIdAll), '--o', 'Color', [0 0.45 0.74], 'LineWidth', 1.5, 'MarkerFaceColor', [0 0.45 0.74], 'MarkerSize', 3);
    plot(x, m(SIsAll), '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
    hold off;  grid on;  yline(0, 'k--', 'LineWidth', 1);  ylim([-1 1.05]);
    step_token_axis(ax, [0.4, max(x) * 1.5], tokPerStep);
    ylabel(sprintf('held-out SI at layer %s (mean over classes)', layerNames{iShow}));
    legend({'probe', 'DIM', 'probe, shuffled labels'}, 'Location', 'southeast');
    title({sprintf('d. selectivity at layer %s across training', layerNames{iShow}), '', ''});
    % e. probe SI (mean over classes) at every layer x every checkpoint; columns centred on log step
    ax = nexttile;
    C = squeeze(mean(SIpAll, 2));                        % [layers × checkpoints]
    lx = log10(x);  e = [lx(1) - 0.15, (lx(1:end - 1) + lx(2:end)) / 2, lx(end) + 0.15];   % column edges (log10 steps)
    pcolor(10 .^ e, -1.5:nL - 0.5, [C, NaN(nL + 1, 1); NaN(1, nC + 1)]);  shading flat   % layer L at y = L, emb at -1
    bwr = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
    colormap(ax, bwr);  clim([-1 1]);  cb = colorbar;  cb.Label.String = 'held-out probe SI (mean over classes)';
    yticks(xt - 2);  yticklabels(layerNames(xt));  ylabel('Layer');
    step_token_axis(ax, [10 ^ e(1), 10 ^ e(end)], tokPerStep);
    title({'e. probe selectivity at every layer across training', 'blank = no readout', '', ''});
end
