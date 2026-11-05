% run_cosine_tokenembed.m
% Cosine similarity between token embeddings (see run_cosine_resid for the residual stream).
% Do related words point in similar directions before any layer has run?
%   1. load the weights; look up each word's token embedding (a column of W_E)
%   2. cosine similarity between every pair of embeddings (cosine_sim)
%   3. average within and between categories
%   4. heatmap
%
% Note: the exported W_E is centered per token (each embedding has its mean over
% the 768 dimensions removed; see export_gpt2_weights), which removes one shared
% offset but not all of the shared structure between embeddings.
%
% Needs: cosine_sim, model_tokenize, model_weights_file.

%% Settings
model = 'gpt2-small';
% Words grouped by category; each must be a single GPT-2 token (with its leading space)
words = {' cat', ' dog', ' horse', ' cow', ...           % animals
         ' car', ' truck', ' bus', ' train', ...         % vehicles
         ' Paris', ' London', ' Rome', ' Berlin', ...    % cities
         ' red', ' blue', ' green', ' yellow'};          % colours
groups = repelem({'animals', 'vehicles', 'cities', 'colours'}, 4);

%% Step 1 — Load and look up the embeddings
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file(model));
ids = zeros(1, numel(words));
for k = 1:numel(words)
    t = model_tokenize(words{k});  t = t(2:end);           % drop <|endoftext|>
    if numel(t) ~= 1, error('"%s" is %d GPT-2 tokens; use a single-token word.', words{k}, numel(t)); end
    ids(k) = t;
end
E = W.W_E(:, ids + 1);                                   % [768 × nWords]  one embedding per word

%% Step 2 — Cosine similarity between every pair
C = cosine_sim(E);                                       % [nWords × nWords]

%% Step 3 — Within- vs between-category averages
[cats, ~, g] = unique(groups, 'stable');
same = g == g';  off = ~eye(numel(words));               % pairs in the same category; leave out the diagonal (= 1)
fprintf('Mean cosine within a category:   %.3f\n', mean(C(same & off)));
fprintf('Mean cosine between categories:  %.3f\n', mean(C(~same)));
for c = 1:numel(cats)
    in = g == c;
    w = C(in, in);  fprintf('  %-9s within %.3f\n', cats{c}, mean(w(~eye(sum(in)))));
end

%% Step 4 — Heatmap
close(findall(groot, 'Type', 'figure', 'Tag', 'run_cosine_tokenembed'));
figure('Color', 'w', 'Position', [100 200 640 560], 'Tag', 'run_cosine_tokenembed');
imagesc(C);  axis square;  clim([-1 1]);
cm = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
colormap(cm);  cb = colorbar;  cb.Label.String = 'cosine similarity';
labels = strtrim(words);
xticks(1:numel(words)); xticklabels(labels); xtickangle(45);
yticks(1:numel(words)); yticklabels(labels);
for c = 1:numel(cats) - 1                                % lines between categories
    b = find(g == c, 1, 'last') + 0.5;
    xline(b, 'Color', [0.4 0.4 0.4]);  yline(b, 'Color', [0.4 0.4 0.4]);
end
for i = 1:numel(words)                                   % values in the cells
    for j = 1:numel(words)
        text(j, i, sprintf('%.2f', C(i, j)), 'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end
title(sprintf('%s token embeddings (W\\_E): cosine similarity', model));
h = findall(gcf, '-property', 'FontSize');                % all text one point larger
for k = 1:numel(h), h(k).FontSize = h(k).FontSize + 1; end
