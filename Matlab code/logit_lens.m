function p_L = logit_lens(x, W, L, display_top_predictions, positions)
% logit_lens  Next-token probabilities from the output of layer ℓ (L).
%   p_L = logit_lens(x, W, L) applies the final layer norm and the
%   unembedding to x^(ℓ), the residual stream after layer ℓ, at every
%   position, i.e. "what would GPT-2 predict if it stopped after layer ℓ?".
%     x      a struct with x.resid_post = residual stream, 768 × n × 12
%            (d_model × position × layer), plus optional x.prompt, x.text
%            (text shown in the printout) and x.token_ids (to show which token
%            each table follows). E.g. x in run_model, or
%            a = load('...\activations.mat'). A plain 768 × n × 12 array
%            (e.g. out.resid_post) also works; then no text is shown.
%     W      weights struct (only W.W_U and W.b_U are used)
%     L    layer ℓ whose output is read, 0-based (0-11). Can be a list of layers, e.g. 0:11.
%     display_top_predictions   optional, default false. If true:
%            - prints the top 5 predictions (rank, id, token, prob), one table
%              per layer and position, and
%            - plots vs layer, one figure per position, always all layers 0-11
%              (whatever L is), for the top 5 tokens at the last layer:
%                top:    probability, plus a dashed line for the top token at
%                        each layer, labelled with its text
%                middle: rank among all 50257 tokens (1 = top, log scale)
%                bottom: cosine between the (centered) residual stream and
%                        each token's output direction E[t] (row of W_U):
%                        the angle part of the logit
%            - plus one figure for all positions: the top token at every
%              layer (rows) and position (columns), shaded by its probability
%              Figures from an earlier logit_lens call are closed first; other
%              figures are left open.
%     positions  optional, which positions to print and plot, 0-based like
%            layers (0 = <|endoftext|>, n-1 = last token). Default: the last
%            token. Only affects printout and plots; p_L always covers
%            every position.
%     p_L  p^(ℓ): 50257 × n              for one layer
%                   50257 × n × numel(L)   for several layers
%            p_L(t + 1, pos + 1) = p^(ℓ)(t), probability of token t after
%            position pos (both 0-based)
%   With L = 11 (the last layer) this is the model's real prediction.
%
% How it works (notation as in Lesson 7, Tab 2):
%   p^(ℓ)(t) = softmax( W_U · LayerNorm( x^(ℓ) ) + b_U )
%   — the predicted token distribution if computation stopped at layer ℓ.
%     x^(ℓ)      the output of layer ℓ                x_L = x.resid_post(:, :, L + 1)
%     LayerNorm  center and scale (its weights are folded into W_U, b_U)   Step 1
%     W_U, b_U   the model's own unembedding: no training, no new weights  Step 2
%     softmax    turns the logits into probabilities                       Step 3
%     p^(ℓ)(t)   probability of token t                p_L(t + 1, pos + 1)
% Unlike a probing classifier, nothing is learned: the "classifier" is the
% model's own W_U, so it can only answer in the model's vocabulary. The
% lesson's formula leaves out b_U; it is included here so that at ℓ = 11
% the result equals the model's actual prediction.
%
% Examples:
%   p = logit_lens(x, W, 6);           % layer 6, every position
%   p(:, end)                          % ... after the last token
%   P = logit_lens(x, W, 0:11);        % every layer
%   [~, top] = max(squeeze(P(:, end, :)));  model_token(top - 1)   % top token per layer, last position
%   logit_lens(x, W, 11, true);        % also print the top 5 after the last token, and plot vs layer
%   logit_lens(x, W, 9, true, [3 8]);  % print layer 9's top 5 after positions 3 and 8
%   logit_lens(x, W, 11, true, 0:size(x.resid_post, 2) - 1);   % print every position (0 = <|endoftext|>)
%
% See also model_forward, logit_lens_fromfile.

% Text of all positions, for the printout: prompt (+ generated text, if any)
txt = "";
tokenIds = [];
if isstruct(x)
    if isfield(x, 'prompt'),    txt = string(x.prompt); end
    if isfield(x, 'text'),      txt = txt + string(x.text); end
    if isfield(x, 'token_ids'), tokenIds = double(x.token_ids); end
    x = x.resid_post;                       % [768 × n × 12]
end

if any(L < 0 | L > size(x, 3) - 1)
    error('logit_lens:layer', 'L must be between 0 and %d (layers are 0-based).', size(x, 3) - 1);
end
% Shapes (for reference; n = positions, k = numel(L) layers asked for):
%   x       — [768 × n × 12]      (d_model × position × layer)
%   W.W_U   — [50257 × 768]       (n_vocab × d_model)
%   W.b_U   — [1 × 50257]
%   p_L   — [50257 × n × k]     (n_vocab × position × layer)

%% Pick the layer(s): x^(ℓ)
x_L = x(:, :, L + 1);                   % [768 × n × k]  (+1: MATLAB arrays start at 1)
[d, n, k] = size(x_L);
x_L = reshape(x_L, d, n * k);           % [768 × n*k]  one column per position (and layer)

%% Step 1 — LayerNorm  (manual, since LN is folded into W_U/b_U)
if ~(isfield(W, 'cfg') && isfield(W.cfg, 'arch') && strcmp(W.cfg.arch, 'llama'))
    x_L = x_L - mean(x_L, 1);         % center each column (not for Llama's RMSNorm)
end
x_L = x_L ./ sqrt(mean(x_L .^ 2, 1) + 1e-5);   % scale to unit variance (population std + eps, as PyTorch)

%% Step 2 — Logits: W_U · LayerNorm(x^(ℓ)) + b_U
logits = W.W_U * x_L + W.b_U';               % [50257 × n*k]  one score per vocabulary token, per column

%% Step 3 — Softmax: p^(ℓ)  (shift for numerical stability)
p_L = exp(logits - max(logits, [], 1));   % [50257 × n*k]
p_L = p_L ./ sum(p_L, 1);             % each column sums to 1

%% Step 4 — Back to position × layer
p_L = reshape(p_L, [], n, k);           % [50257 × n × k]

%% Optional: print the top 5, for each layer and chosen position
if nargin < 4
    display_top_predictions = false;
end
if nargin < 5 || isempty(positions)
    positions = n - 1;                      % default: the last token (0-based)
end
if any(positions < 0 | positions > n - 1)
    error('logit_lens:position', 'positions must be between 0 and %d (0-based).', n - 1);
end
show = @(id) '"' + replace(model_token(id, W), newline, "\n") + '"';   % token text, e.g. " Paris"

if display_top_predictions
    if strlength(txt) > 0
        fprintf('\n************************************\nNext-token prediction for prompt:\n\n%s', txt)
    end

    for j = 1:k
        for pos = positions(:)'
            [sorted_p, idx] = sort(p_L(:, pos + 1, j), 'descend');   % +1: MATLAB index
            where = sprintf('after position %d', pos);
            if pos == n - 1, where = [where ' (last token)']; end
            if numel(tokenIds) == n, where = [where ', ' char(show(tokenIds(pos + 1)))]; end
            fprintf('\n____________________________________\nLayer %d, %s\nRank  token_id  token        prob\n', L(j), where);
            for i = 1:5
                fprintf('%4d  %8d  %-12s %.4f%%\n', i, idx(i) - 1, show(idx(i) - 1), sorted_p(i) * 100);
            end
        end
    end
end

%% Optional: plot probability vs layer, one figure per chosen position
if display_top_predictions
    close(findall(groot, 'Type', 'figure', 'Tag', 'logit_lens'));   % close earlier logit_lens figures (only those)
    layers = 0:size(x, 3) - 1;                                  % always every layer ℓ: 0-11
    p_all = logit_lens(x(:, positions + 1, :), W, layers);     % [50257 × numel(positions) × 12]  p^(ℓ) for every ℓ
    for j = 1:numel(positions)
        pos = positions(j);
        p_pos = 100 * reshape(p_all(:, j, :), [], numel(layers));   % [50257 × 12]  p^(ℓ) (%) at this position, each layer
        [~, top5] = maxk(p_pos(:, end), 5);                        % top 5 at the last layer (the final prediction)
        [pTop, iTop] = max(p_pos, [], 1);                          % top token at each layer
        ranks = zeros(5, numel(layers));                       % [5 × 12]  rank of each top-5 token (1 = most likely)
        for i = 1:5
            ranks(i, :) = sum(p_pos > p_pos(top5(i), :), 1) + 1;       % how many tokens beat it, + 1
        end

        off = 40 * mod(j - 1, 8);                              % fan out: each extra position 40 px right and down
        top = get(groot, 'ScreenSize') * [0; 0; 0; 1] - 100;   % top edge, just below the screen's top
        % Cosine between the residual stream and each top-5 token's output direction E[t] (row of W_U).
        % The logit is sqrt(d) * ||E[t]|| * cos + b_U(t): this is its angle part.
        xp = double(reshape(x(:, pos + 1, :), size(x, 1), []));   % [768 × 12]  residual stream after each layer
        if ~(isfield(W, 'cfg') && isfield(W.cfg, 'arch') && strcmp(W.cfg.arch, 'llama'))
            xp = xp - mean(xp, 1);                             % centered, as LayerNorm does (not for RMSNorm)
        end
        E = double(W.W_U(top5, :));                            % [5 × 768]  output directions of the top 5
        cosE = (E * xp) ./ (vecnorm(E, 2, 2) * vecnorm(xp));   % [5 × 12]

        figure('Color', 'w', 'Position', [20 + off, top - 900 - off, 780, 900], 'Tag', 'logit_lens');   % tagged, so the next call can close it
        tl = tiledlayout(3, 1, 'TileSpacing', 'compact');

        % Subplot 1 — probability vs layer
        ax1 = nexttile;
        h1 = plot(layers, p_pos(top5, :)', '-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on
        h1(1).LineWidth = 3;  uistack(h1(1), 'top');           % the final top token: 2 × thicker, drawn on top
        hTop = plot(layers, pTop, 'k--', 'LineWidth', 1);
        text(layers, pTop, cellstr(show(iTop - 1)), 'Interpreter', 'none', 'FontSize', 8, ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Rotation', 35);
        hold off
        ylabel('Probability (%)'); ylim([0 115]); grid on

        % Subplot 2 — rank vs layer, same tokens and colours (log scale, rank 1 at the top)
        ax2 = nexttile;
        h2 = semilogy(layers, ranks', '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
        h2(1).LineWidth = 3;  uistack(h2(1), 'top');
        set(ax2, 'YDir', 'reverse');
        ylim([0.8, max(ranks(:)) * 1.5]);
        ticks = 10 .^ (0:ceil(log10(max(ranks(:)) * 1.5)));   % 1, 10, 100, ...
        set(ax2, 'YTick', ticks, 'YTickLabel', string(ticks), 'YMinorGrid', 'off', 'YMinorTick', 'off');
        ylabel('Rank (1 = top)'); grid on

        % Subplot 3 — cosine with each token's output direction, same tokens and colours
        ax3 = nexttile;
        h3 = plot(layers, cosE', '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
        h3(1).LineWidth = 3;  uistack(h3(1), 'top');
        yline(0, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
        ylabel({'cos(stream, E[t])', '(angle part of the logit)'}); xlabel('Layer'); grid on

        step = 1 + (numel(layers) > 12);                       % label every layer (small), every 2nd (XL)
        set([ax1 ax2 ax3], 'XTick', layers(1:step:end), 'XLim', [layers(1) - 0.5, layers(end) + 0.5], ...
            'XTickLabelRotation', 0);                          % layer numbers upright
        lgd = legend(ax1, [h1; hTop], [cellstr(show(top5 - 1)); {'top token at each layer'}], 'Interpreter', 'none');   % handles: uistack changed the plot order
        lgd.Layout.Tile = 'east';                              % beside both plots, hides no data

        where = sprintf('after position %d', pos);
        if numel(tokenIds) == n, where = [where ', ' char(show(tokenIds(pos + 1)))]; end
        if strlength(txt) > 0
            title(tl, {char(txt), ['Logit lens ' where]}, 'Interpreter', 'none');
        else
            title(tl, ['Logit lens ' where], 'Interpreter', 'none');
        end
    end

    % One more figure: the top token at every layer and position (all positions, not only the chosen ones)
    P = logit_lens(x, W, layers);                              % [50257 × n × 12]  p^(ℓ) at every position and layer
    [pTop, iTop] = max(P, [], 1);                              % top token and its probability
    pTop = squeeze(pTop);  iTop = squeeze(iTop);               % [n × 12]  position × layer
    figure('Color', 'w', 'Position', [60, 80, max(780, 110 * n), 120 + 32 * numel(layers)], 'Tag', 'logit_lens');
    imagesc(0:n - 1, layers, 100 * pTop');  axis xy            % shade = probability of the top token (%)
    colormap([linspace(1, 0.42, 64)', linspace(1, 0.62, 64)', linspace(1, 0.92, 64)']);   % white -> blue
    clim([0 100]);  cb = colorbar;  cb.Label.String = 'p(top token) (%)';
    for pos = 0:n - 1                                          % label each cell with its top token
        text(pos * ones(1, numel(layers)), layers, cellstr(show(iTop(pos + 1, :) - 1)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');
    end
    set(gca, 'XTick', 0:n - 1, 'YTick', layers, 'TickLabelInterpreter', 'none');
    if numel(tokenIds) == n
        set(gca, 'XTickLabel', cellstr(show(tokenIds)));  xlabel('input token at each position');
    else
        xlabel('position');
    end
    ylabel('Layer');
    if strlength(txt) > 0
        title({char(txt), 'Logit lens: top token at every layer and position'}, 'Interpreter', 'none');
    else
        title('Logit lens: top token at every layer and position');
    end
end
end
