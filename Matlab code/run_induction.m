% run_induction.m
% In-context copying (induction heads, Olsson et al. 2022) across training.
%   1. nSeq sequences of seqLen random tokens, each repeated twice: <|endoftext|> r r (GPT-2 or Pythia)
%   2. for each checkpoint, the loss on every token of the first and the second copy; random tokens cannot be
%      predicted the first time (loss ~ ln 50,257 ≈ 10.8 or worse), but a model that copies from earlier in the
%      context predicts the second copy easily
%   3. induction score = mean loss on the first copy - mean loss on the second (nats), against training step;
%      it jumps when induction heads form (a "phase change"). The second-copy loss alone is the scale-free
%      version (the score also grows when the first copy gets harder, e.g. Pythia-160M's last checkpoint)
%   4. per head, from the same runs (attention patterns, 0-1 scales as in Olsson et al. 2022 and Qu & Liu 2026):
%        prefix-matching score  attention from each second-copy token to the token right AFTER its earlier
%                               occurrence (the induction head's target), mean over positions and sequences
%        previous-token score   attention from each token to the token just before it (the first half of
%                               the induction circuit), mean over all positions after <|endoftext|>
%      chance (uniform attention over the keys a query can see) is printed and drawn as a dotted line
%   Plot: across checkpoints, copying and the best heads against training step; for the trained model alone
%   (checkpointRange = []), a. the loss on each token, b. the prefix-matching and c. the previous-token score of
%   every head (Lesson 9, tab 10)
%   Optional ablation (ablate_type, ablate_which, ablate_how): the same with components removed, at every
%   checkpoint (Lesson 9, tab 11, Fig. 16)
%
% Needs: model_checkpoints, model_forward, step_token_axis (and ablate, if an ablation is set).

%% Settings
model           = 'gpt2-small';
checkpointRange = [];   % training steps (checkpoints on disk); [] = the trained model
nSeq            = 20;                      % sequences
seqLen          = 50;                      % random tokens per copy
seed            = 1;
ablate_type     = '';                           % '' = no ablation (the default); or 'head', 'mlp', 'attn', 'layer' (as in run_ablation)
ablate_which    = [];                       % 'head': [layer head] rows, e.g. [5 5; 6 9]; others: layers, e.g. 5:7
ablate_how      = 'zero';                  % 'zero', 'noise', or 'mean' (see ablate); applied at every position

%****************************************************

%% Step 1 — Random token sequences, repeated twice
addpath(fileparts(mfilename('fullpath')));
rng(seed);  R = randi([1000 30000], nSeq, seqLen);                 % ordinary tokens (avoid the rarest and special ones)
eot = 50256;  if startsWith(model, 'pythia'), eot = 0; end         % <|endoftext|>: GPT-2 50256, Pythia 0
n = 2 * seqLen + 1;                                                % tokens per sequence
qI = seqLen + 2:n;  kI = qI - seqLen + 1;                          % prefix matching: second-copy query -> token after its earlier occurrence
qP = 2:n;                                                          % previous token: every query after <|endoftext|> -> the one before
chanceI = mean(1 ./ qI);  chanceP = mean(1 ./ qP);                 % uniform attention over the q visible keys

%% Step 2 — Loss on the first and second copy, and the heads' attention, at every checkpoint
edit = [];  ablNote = '';                                         % optional ablation (Settings)
if ~isempty(ablate_type)
    edit = ablate(ablate_type, ablate_which, ablate_how);
    if strcmp(ablate_type, 'head'), w = strjoin(compose('%d.%d', ablate_which(:, 1), ablate_which(:, 2)), ' ');
    else, w = strjoin(compose('L%d', ablate_which(:)), ' '); end
    ablNote = sprintf(', %s %s ablated (%s)', ablate_type, w, ablate_how);  fprintf('ablation:%s\n', ablNote(2:end));
end
[files, ckSteps, tokPerStep] = model_checkpoints(model, checkpointRange);  nC = numel(files);
trained = isempty(ckSteps);  if trained, ckSteps = NaN; end   % the trained model: one "checkpoint", no step
[loss1, loss2] = deal(nan(1, nC));  [lossTok, lossTok0] = deal(zeros(nSeq, n - 1));   % lossTok0: the clean run, if ablating
for ck = 1:nC
    W = load(files{ck});  [l1, l2] = deal(nan(nSeq, 1));  [~, ~, nH, nL] = size(W.W_Q);
    if ck == 1, [prefixAll, prevAll] = deal(zeros(nL, nH, nC)); end   % [layer × head × checkpoint]
    for s = 1:nSeq
        ids = [eot R(s, :) R(s, :)];  out = model_forward(ids, W, edit);   % edit = [] when nothing is ablated
        nll = token_loss(out.logits, ids);                         % loss of tokens 2..end
        l1(s) = mean(nll(2:seqLen));                               % first copy (its first token has no context)
        l2(s) = mean(nll(seqLen + 1:end));                         % second copy
        if ck == nC, lossTok(s, :) = nll; end                      % per-token loss of the last checkpoint (trained-model plot)
        if ck == nC && ~isempty(edit), lossTok0(s, :) = token_loss(model_forward(ids, W).logits, ids); end   % and without the ablation
        A = reshape(out.attn_pattern, n * n, nH * nL);             % [query·key × head·layer]
        prefixAll(:, :, ck) = prefixAll(:, :, ck) + reshape(mean(A(sub2ind([n n], qI, kI), :), 1), nH, nL)' / nSeq;
        prevAll(:, :, ck)   = prevAll(:, :, ck)   + reshape(mean(A(sub2ind([n n], qP, qP - 1), :), 1), nH, nL)' / nSeq;
    end
    loss1(ck) = mean(l1);  loss2(ck) = mean(l2);
    [pI, iI] = max(prefixAll(:, :, ck), [], 'all');  [lI, hI] = ind2sub([nL nH], iI);
    [pP, iP] = max(prevAll(:, :, ck), [], 'all');    [lP, hP] = ind2sub([nL nH], iP);
    fprintf('step %6d  loss first copy %.2f  second copy %.2f  induction score %.2f nats | best prefix-matching head %d.%d %.2f, previous-token head %d.%d %.2f\n', ...
            ckSteps(ck), loss1(ck), loss2(ck), loss1(ck) - loss2(ck), lI - 1, hI - 1, pI, lP - 1, hP - 1, pP);
end
fprintf('chance (uniform attention): prefix matching %.3f, previous token %.3f\n', chanceI, chanceP);
prefixMax = squeeze(max(prefixAll, [], [1 2]))';  prevMax = squeeze(max(prevAll, [], [1 2]))';

%% Step 3 — Plot
close(findall(groot, 'Type', 'figure', 'Tag', 'run_induction'));   % this script's figure from an earlier run
if trained                                                         % no training axis: per-token loss and the per-head scores
    figure('Color', 'w', 'Position', [40 40 2100 640], 'Tag', 'run_induction');  tl = tiledlayout(1, 3, 'TileSpacing', 'loose');
    nexttile;  sm = @(v) movmean(mean(v, 1), 3);                   % a. loss on each token (3-token running mean)
    if isempty(edit)
        plot(1:n - 1, sm(lossTok), 'k-', 'LineWidth', 2.5);
    else                                                           % clean and ablated
        plot(1:n - 1, sm(lossTok0), 'k-', 'LineWidth', 2);  hold on
        plot(1:n - 1, sm(lossTok), '-', 'Color', [0 0.45 0.74], 'LineWidth', 2.5);  hold off
        legend({'clean', ablNote(3:end)}, 'Location', 'southwest');
    end
    xline(seqLen + 0.5, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');  grid on;  xlim([1 n - 1]);
    xlabel('Token position (first copy | second copy)');  ylabel('loss (nats per token)');
    title({'a. loss on each token', sprintf('%d random sequences, each repeated twice', nSeq)});
    lb = [linspace(1, 0.42, 64)', linspace(1, 0.62, 64)', linspace(1, 0.92, 64)'];
    sc = {prefixAll(:, :, end), prevAll(:, :, end)};               % [layer × head]
    tt = {{'b. prefix-matching score', 'attention to the token after the earlier copy'}, {'c. previous-token score', 'attention to the token just before'}};
    for k = 1:2                                                    % b, c. every head: layer across, head up
        nexttile;  imagesc(0:nL - 1, 0:nH - 1, sc{k}');  axis xy;  axis square;  colormap(gca, lb);  clim([0 1]);
        cb = colorbar;  cb.Label.String = 'mean attention';  xticks(0:nL - 1);  yticks(0:nH - 1);  xlabel('Layer');  ylabel('Head');
        [best, ib] = max(sc{k}, [], 'all');  [lb_, hb] = ind2sub([nL nH], ib);   % the head with the highest score
        hold on;  plot(lb_ - 1, hb - 1, 'k*', 'MarkerSize', 16, 'LineWidth', 1.5);  hold off
        title([tt{k}, {sprintf('* = best head, %d.%d (%.2f)', lb_ - 1, hb - 1, best)}]);
    end
    title(tl, sprintf('%s: in-context copying, trained model%s', model, ablNote), 'Interpreter', 'none');
    return
end
x = max(ckSteps, 0.5);  xl = [0.4 max(x) * 1.5];
figure('Color', 'w', 'Position', [100 100 1500 430], 'Tag', 'run_induction');  tl = tiledlayout(1, 3, 'TileSpacing', 'compact');
ax = nexttile;  semilogx(x, loss1, 'o-', x, loss2, 's-', x, loss1 - loss2, 'k^-', 'LineWidth', 1.6);  grid on;
legend('first copy', 'second copy', 'induction score (first - second)', 'Location', 'best');  ylabel('loss (nats per token)');
step_token_axis(ax, xl, tokPerStep);  title({'a. copying a repeated random sequence', '', ''});
ax = nexttile;  semilogx(x, prefixMax, 'o-', x, prevMax, 's-', 'LineWidth', 1.6);  hold on;
yline(chanceI, ':', 'Color', [0 0.45 0.74]);  yline(chanceP, ':', 'Color', [0.85 0.33 0.1]);  hold off;  grid on;  ylim([0 1]);
legend('prefix matching (best head)', 'previous token (best head)', 'Location', 'northwest');  ylabel('attention weight');
step_token_axis(ax, xl, tokPerStep);  title({'b. induction and previous-token heads', '', ''});
nexttile;  imagesc(0:nH - 1, 0:nL - 1, prefixAll(:, :, end), [0 1]);  axis xy;  colorbar;
xlabel('head');  ylabel('layer');  title({sprintf('c. prefix-matching score per head, step %d', ckSteps(end)), ''});
title(tl, sprintf('%s: in-context copying%s', model, ablNote), 'Interpreter', 'none');

% -------------------------------------------------------------------------
function nll = token_loss(logits, ids)
% loss (nats) of tokens 2..end, each predicted from the position before it
    Z = logits(:, 1:end - 1);  Z = Z - max(Z, [], 1);  Ls = Z - log(sum(exp(Z), 1));
    nll = -Ls(sub2ind(size(Ls), ids(2:end) + 1, 1:numel(ids) - 1));
end
