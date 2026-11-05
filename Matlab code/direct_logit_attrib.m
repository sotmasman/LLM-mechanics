function d = direct_logit_attrib(out, W, t, position, display_results, nTop_heads, clim_heads, extra_heads)
% direct_logit_attrib  Direct logit attribution: how much each component pushed a token's logit.
%   d = direct_logit_attrib(out, W) splits the final logit of the predicted token at the last
%   position into one contribution per model component. The contributions
%   add up exactly to the logit.
%     out    model_forward output, e.g. g.out from run_model (needs resid_pre,
%            attn_by_head, mlp_out, resid_post, logits, ids)
%     W      weights struct (uses W.W_U, W.b_U, W.b_O)
%     t      optional, target token: 0-based token ID. Default: the top
%            predicted token at that position. Two IDs [t1 t2] give the logit
%            DIFFERENCE t1 - t2 (why t1 beat t2), e.g. [6342 45996] = " Paris" - " Marse".
%     position  optional, 0-based. Default: the last position of out
%            (in run_model with nNew = 1: the prediction that chose the new token).
%     display_results  optional, default false. If true, plots the
%            contributions in four panels: by layer (attention and MLP bars,
%            plus a black dot for each layer's total, attention + MLP), the
%            running sum (where the total builds up), by head, and where the top
%            heads look (their attention from this position). Nothing is printed;
%            all the numbers are in d.
%     nTop_heads  optional, default 5. How many heads (largest |contribution|)
%            to report in d.top_heads / d.top_attn and the third panel.
%            'all' = every head, in layer.head order (0.0, 0.1, ..., 11.11): the third
%            panel then shows all of them without cell labels, with each head's DLA
%            as a coloured column beside it.
%     extra_heads  optional, [layer head] rows (0-based), e.g. [5 5; 4 11]: heads always
%            added to the top heads (third panel and d.top_heads), whatever their DLA.
%     clim_heads  optional, default [] (automatic). Colour range of the head
%            heatmap (second panel): colours run from -clim_heads (blue) to
%            +clim_heads (red). [] scales each figure to its own largest head,
%            which shows the structure within one prompt; a fixed value
%            (e.g. 3) makes figures from different prompts comparable by eye.
%            Heads beyond the range are shown in the strongest colour.
%
% Output struct (all in logit units; positive = pushed token t up):
%   d.emb        1 × 1     x^(emb) · E[t]: token + position embedding
%   d.heads      12 × 12   each attention head (head × layer, both +1 indexed)
%   d.attn_bias  1 × 12    attention output bias b_O of each layer
%   d.attn       1 × 12    Δx_attn,ℓ · E[t] for each layer ℓ: sum of its heads + b_O
%   d.mlp        1 × 12    Δx_mlp,ℓ · E[t] for each layer ℓ
%   d.b_U        1 × 1     unembedding bias of token t
%   d.total      1 × 1     sum of all of the above
%   d.running    1 × (2·nL+2)  running sum in the order the terms are added: emb, then 0 attn, 0 MLP,
%                          1 attn, ..., 11 MLP, then b_U (last value = d.total). After layer L's
%                          MLP it is the stream after layer L read like the final output (final
%                          layer norm scale); not the logit lens, which uses each layer's own scale
%   d.logit      1 × 1     the model's actual logit(t) (or difference), for checking
%   d.t, d.position        what was attributed (token ID(s), 0-based position)
%   d.t_str                the token text, from model_token (e.g. " Paris";
%                          a 1 × 2 string array for a logit difference)
%   d.top_heads  nTop × 2  the heads with the largest |contribution|, as
%                          [layer head] (0-based), largest first
%   d.top_attn   nTop × n  their attention from this position to every
%                          position (rows sum to 1). Shows which earlier token
%                          each head reads from. Empty if out has no attn_pattern
%                          (e.g. a Python-saved activations file).
%
% How it works (notation as in Lesson 7, Tab 3):
%   x^(L) = x^(0) + Σ_ℓ Δx_attn,ℓ + Σ_ℓ Δx_mlp,ℓ        final residual stream (x_final)
%   logit(t) = E[t] · LayerNorm(x^(L)) + b_U(t)          E[t] = row t of W_U
% LayerNorm centers (linear, so it applies to each term) and divides by one
% scale per position, which is held at its actual value. So each term's share
% is E[t] · (Δx - mean(Δx)) / scale, and the shares sum exactly to logit(t).
%
% Examples:
%   d = direct_logit_attrib(g.out, W);                          % top predicted token, last position
%   d = direct_logit_attrib(g.out, W, 6342);                    % t = " Paris"
%   d = direct_logit_attrib(g.out, W, [6342 45996], [], true);  % " Paris" - " Marse", with plot
%   d = direct_logit_attrib(g.out, W, [], [], true, 10);        % show the top 10 heads
%   d = direct_logit_attrib(g.out, W, [], [], true, [], 3);     % head heatmap on a fixed ±3 scale
%   d.heads(9 + 1, 10 + 1)                      % head 10.9's contribution (layer 10, head 9)
%
% Finding the strongest components (max/ind2sub return MATLAB positions,
% which start at 1, so subtract 1 to get layer and head numbers):
%   [val, idx] = max(d.mlp);
%   fprintf('Strongest MLP: layer %d (%+.2f)\n', idx - 1, val);         % e.g. layer 10 (+2.54)
%
%   [val, idx] = max(d.heads(:));               % search all 144 heads
%   [row, col] = ind2sub(size(d.heads), idx);   % row = head + 1, column = layer + 1
%   fprintf('Strongest head: %d.%d (%+.2f)\n', col - 1, row - 1, val);   % e.g. 9.8 (+4.14)
%
%   layer = 9; head = 8;
%   d.heads(head + 1, layer + 1)                % one head directly: head 9.8
%
%   Use min instead of max for the strongest push AGAINST the token, or
%   max(abs(...)) for the largest effect either way.
%
% See also model_forward, logit_lens.

% Shapes (for reference; n = positions):
%   out.resid_pre, out.resid_post, out.mlp_out — [768 × n × 12]
%   out.attn_by_head                            — [768 × 12 × n × 12]  (d_model × head × position × layer)
%   out.attn_pattern                            — [n × n × 12 × 12]    (query × key × head × layer)
%   W.W_U — [50257 × 768],  W.b_U — [1 × 50257],  W.b_O — [768 × 12]

n = size(out.resid_post, 2);
n_layers = size(out.resid_post, 3);
n_heads = size(out.attn_by_head, 2);
if nargin < 4 || isempty(position)
    position = n - 1;                                   % default: last position (0-based)
end
if position < 0 || position > n - 1
    error('direct_logit_attrib:position', 'position must be between 0 and %d (0-based).', n - 1);
end
p = position + 1;                                       % MATLAB index
if nargin < 3 || isempty(t)
    [~, top] = max(out.logits(:, p));
    t = top - 1;                                        % default: top predicted token
end
if nargin < 5
    display_results = false;
end
if nargin < 6 || isempty(nTop_heads)
    nTop_heads = 5;
end
if nargin < 7
    clim_heads = [];                                    % automatic colour range
end
if nargin < 8
    extra_heads = [];
end
allHeads = (ischar(nTop_heads) || isstring(nTop_heads)) && strcmpi(nTop_heads, 'all');

%% Step 1 — E[t]: the direction in residual space that logit(t) reads
E_t = double(W.W_U(t(1) + 1, :))';                      % [768 × 1]  row t of W_U (+1: MATLAB index)
b_U_t = double(W.b_U(t(1) + 1));
logit_t = double(out.logits(t(1) + 1, p));              % the model's actual logit(t)
if numel(t) == 2                                        % logit difference t1 - t2
    E_t = E_t - double(W.W_U(t(2) + 1, :))';            % E[t1] - E[t2]
    b_U_t = b_U_t - double(W.b_U(t(2) + 1));
    logit_t = logit_t - double(out.logits(t(2) + 1, p));
end

%% Step 2 — LayerNorm scale of x_final at this position (held fixed)
x_final = double(out.resid_post(:, p, n_layers));       % [768 × 1]  x^(L), final residual stream
x_final = x_final - mean(x_final);
scale = sqrt(mean(x_final .^ 2) + 1e-5);                % one number for this position
dot_E_t = @(dx) (E_t' * (dx - mean(dx, 1))) / scale;    % Δx [768 × k] -> Δx · E[t] [1 × k], through LayerNorm

%% Step 3 — Each term's share of logit(t)
xEmb          = double(out.resid_pre(:, p, 1));                                    % [768 × 1]  x^(emb), embedding
dx_attn_heads = reshape(double(out.attn_by_head(:, :, p, :)), [], n_heads * n_layers);   % [768 × 144]  each head's Δx
dx_mlp        = reshape(double(out.mlp_out(:, p, :)), [], n_layers);               % [768 × 12]  Δx_mlp,ℓ

d.emb       = dot_E_t(xEmb);                                                      % x^(emb) · E[t]
d.heads     = reshape(dot_E_t(dx_attn_heads), n_heads, n_layers);                 % [12 × 12]  head × layer
d.attn_bias = dot_E_t(double(W.b_O));                                             % [1 × 12]
d.attn      = sum(d.heads, 1) + d.attn_bias;                                      % [1 × 12]  Δx_attn,ℓ · E[t]
d.mlp       = dot_E_t(dx_mlp);                                                    % [1 × 12]  Δx_mlp,ℓ · E[t]
d.b_U       = b_U_t;
d.total     = d.emb + sum(d.attn) + sum(d.mlp) + d.b_U;                            % should equal logit(t)
d.running   = cumsum([d.emb, reshape([d.attn; d.mlp], 1, []), d.b_U]);            % [1 × 2·nL+2]  in the order they run
d.logit     = logit_t;
d.t         = t;
d.t_str     = model_token(t, W);                                                      % e.g. " Paris" (two strings for a difference)
d.position  = position;

%% Step 4 — Where the top heads look
% Rank all heads by |contribution|, then read each top head's attention
% pattern from this position: row p of attn_pattern(:, :, head, layer) =
% how much the token at this position attends to each earlier token.
if allHeads
    [hRow, LCol] = ndgrid(1:n_heads, 1:n_layers);                               % every head, layer.head order
    hRow = hRow(:);  LCol = LCol(:);
else
    [~, order] = sort(abs(d.heads(:)), 'descend');                               % all 144 heads, largest first
    sel = order(1:min(nTop_heads, numel(order)));
    if ~isempty(extra_heads)                                                     % heads always shown
        ex = sub2ind(size(d.heads), extra_heads(:, 2) + 1, extra_heads(:, 1) + 1);
        sel = [sel; ex(~ismember(ex, sel))];
    end
    [hRow, LCol] = ind2sub(size(d.heads), sel);                                  % MATLAB row / column
end
d.top_heads = [LCol - 1, hRow - 1];                                              % [nTop × 2]  (layer, head), 0-based
d.top_attn = [];
if isfield(out, 'attn_pattern')
    d.top_attn = zeros(size(d.top_heads, 1), n);                                 % [nTop × n]
    for k = 1:size(d.top_heads, 1)
        d.top_attn(k, :) = out.attn_pattern(p, :, hRow(k), LCol(k));              % query = this position
    end
end

%% Optional: plot (the numbers are all in d)
if display_results
    show = @(id) '"' + replace(model_token(id, W), newline, "\n") + '"';
    if numel(t) == 2
        what = sprintf('%s - %s (logit difference)', show(t(1)), show(t(2)));
    else
        what = sprintf('%s (logit)', show(t));
    end

    close(findall(groot, 'Type', 'figure', 'Tag', 'direct_logit_attrib'));   % close earlier direct_logit_attrib figures (only those)
    nPanels = 3 + (~isempty(d.top_attn)) * (1 + allHeads);   % all heads: the last panel is twice as tall
    top = get(groot, 'ScreenSize') * [0; 0; 0; 1] - 100;   % top edge, just below the screen's top
    h = min(300 * nPanels + 60, top - 40);              % figure height, at most the screen's
    figure('Color', 'w', 'Position', [820, max(40, top - h), 780, h], 'Tag', 'direct_logit_attrib');   % right of logit_lens's figure
    tl = tiledlayout(nPanels, 1, 'TileSpacing', 'loose');

    nexttile;                                           % per layer: attention, MLP and their total
    layers = 0:n_layers - 1;
    step = 1 + (n_layers > 12);                         % label every layer (small), every 2nd (XL)
    xE = -step;                                         % where the embedding (emb) is drawn: -1 (small), -2 (XL)
    plot(xE, d.emb, 's', 'Color', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerSize', 7); hold on   % embedding
    plot(layers, d.attn, '-o', 'Color', [0.85 0.33 0.10], 'MarkerFaceColor', [0.85 0.33 0.10], 'LineWidth', 1.5, 'MarkerSize', 5);
    plot(layers, d.mlp,  '-o', 'Color', [0.93 0.69 0.13], 'MarkerFaceColor', [0.93 0.69 0.13], 'LineWidth', 1.5, 'MarkerSize', 5);
    plot(layers, d.attn + d.mlp, '-o', 'Color', 'k', 'MarkerFaceColor', 'k', 'LineWidth', 1.5, 'MarkerSize', 5);   % each layer's net
    xB = n_layers - 1 + step;                           % where the bias b_U is drawn: after the last layer
    plot(xB, d.b_U, 'd', 'Color', [0.49 0.18 0.56], 'MarkerFaceColor', [0.49 0.18 0.56], 'MarkerSize', 8);   % bias: no layer, same for every prompt
    yline(0, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');                % zero line
    hold off
    v = [d.emb, d.attn, d.mlp, d.attn + d.mlp, d.b_U, 0];
    ylim([min(v), max(v)] + [-1 1] * 0.1 * (max(v) - min(v)));   % 10% margin so no dot sits on the edge
    xticks([xE, 0:step:n_layers - 1, xB]); xticklabels(['emb', string(0:step:n_layers - 1), 'b_U']); xtickangle(0);
    set(gca, 'TickLabelInterpreter', 'none');           % "b_U" as is, not a subscript
    xlim([xE - 0.6, xB + 0.6]); grid on
    ylabel('Contribution (logit)'); xlabel('Layer');
    legend({'emb (embedding)', 'attention', 'MLP', 'layer total (attention + MLP)', 'b_U (bias, same for every prompt)'}, ...
           'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 3, 'Interpreter', 'none');
    title('By layer   (all terms together add up to the total)', 'Interpreter', 'none');

    nexttile;                                           % running sum: the terms added in the order they run
    xs = [xE, reshape([layers - 0.25; layers + 0.25], 1, []), xB];   % emb, attention left of each layer, MLP right, b_U
    plot(xs, d.running, 'k-', 'LineWidth', 1.5); hold on
    plot(xE, d.running(1), 's', 'Color', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerSize', 7);
    plot(xs(2:2:end - 1), d.running(2:2:end - 1), 'o', 'Color', [0.85 0.33 0.10], 'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerSize', 5);
    plot(xs(3:2:end - 1), d.running(3:2:end - 1), 'o', 'Color', [0.93 0.69 0.13], 'MarkerFaceColor', [0.93 0.69 0.13], 'MarkerSize', 5);
    plot(xB, d.running(end), 'd', 'Color', [0.49 0.18 0.56], 'MarkerFaceColor', [0.49 0.18 0.56], 'MarkerSize', 8);
    yline(d.logit, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1);  yline(0, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    hold off
    v = [d.running, d.logit, 0];
    ylim([min(v), max(v)] + [-1 1] * 0.1 * (max(v) - min(v)));
    xticks([xE, 0:step:n_layers - 1, xB]); xticklabels(['emb', string(0:step:n_layers - 1), 'b_U']); xtickangle(0);
    set(gca, 'TickLabelInterpreter', 'none');
    xlim([xE - 0.6, xB + 0.6]); grid on
    ylabel('Running sum (logit)'); xlabel('Layer');
    legend({'running sum', 'emb', 'after attention', 'after MLP', '+ b_U = total', 'model''s actual value'}, ...
           'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 3, 'Interpreter', 'none');
    title({'', 'Running sum: direct contributions so far (final layer norm scale; not the logit lens)'}, 'Interpreter', 'none');

    ax2 = nexttile;                                     % per head
    imagesc(0:n_layers - 1, 0:n_heads - 1, d.heads);
    axis xy                                             % head 0 at the bottom, heads increasing upward
    if isempty(clim_heads)
        m = max(abs(d.heads(:)));                       % automatic: this figure's largest head
    else
        m = clim_heads;                                 % fixed, for comparing prompts
    end
    clim([-m m]);
    colormap(gca, [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ...
                   ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)']);   % blue - white - red
    cb = colorbar; cb.Label.String = 'Contribution (logit)';
    xlabel('Layer'); ylabel('Head'); xticks(0:step:n_layers - 1); xtickangle(0); yticks(0:n_heads - 1);
    if isempty(clim_heads), scaleNote = 'auto'; else, scaleNote = 'fixed'; end
    title({'', sprintf('Each attention head (red = pushes up, blue = pushes down; scale ±%.2g, %s)', m, scaleNote)});   % blank line: space below the panel above

    if ~isempty(d.top_attn) && allHeads                % where every head looks, DLA alongside
        t3 = tiledlayout(tl, 1, 24, 'TileSpacing', 'tight');  t3.Layout.Tile = 4;  t3.Layout.TileSpan = [2 1];
        yt = ((0:n_layers - 1) + 0.5) * n_heads + 0.5;  % the middle of each layer's 12 rows
        axD = nexttile(t3);                             % each head's DLA, same colours as the heatmap above
        imagesc(d.heads(:));  clim([-m m]);  colormap(axD, colormap(ax2));
        yticks(yt);  yticklabels(string(0:n_layers - 1));  xticks([]);  ylabel('Layer (heads 0 to 11 within each)');
        title('DLA');
        ax3 = nexttile(t3, [1 23]);
        imagesc(100 * d.top_attn);  colormap(ax3, flipud(gray));  clim([0 100]);  hold on
        for L = 1:n_layers - 1, yline(L * n_heads + 0.5, 'Color', [0.75 0.75 0.75]); end   % between layers
        hold off
        cb = colorbar;  cb.Label.String = 'Attention (%)';
        toks = strtrim(model_token(out.ids, W));
        xticks(1:n);  xticklabels(compose('%d: %s', (0:n - 1)', toks'));  xtickangle(30);
        yticks(yt);  yticklabels([]);
        ax3.XAxis.TickLabelInterpreter = 'none';
        title({'', sprintf('Where every head looks from position %d', position)});
    elseif ~isempty(d.top_attn)                        % where the top heads look
        ax3 = nexttile;
        A = 100 * d.top_attn;                           % [nTop × n]  attention (%)
        imagesc(A);  colormap(ax3, flipud(gray));  clim([0 100]);
        cb = colorbar;  cb.Label.String = 'Attention (%)';
        toks = strtrim(model_token(out.ids, W));
        xticks(1:n);  xticklabels(compose('%d: %s', (0:n - 1)', toks'));  xtickangle(30);
        yticks(1:size(A, 1));
        % y labels: "layer.head (DLA)", with the DLA value red if positive, blue if negative
        dla_top = d.heads(sub2ind(size(d.heads), d.top_heads(:, 2) + 1, d.top_heads(:, 1) + 1));
        colors = repmat("0.1,0.3,0.85", size(dla_top));                % blue (negative)
        colors(dla_top > 0) = "0.8,0.1,0.1";                           % red (positive)
        yticklabels(compose('%d.%d (\\color[rgb]{%s}%+.2f\\color{black})', ...
                            d.top_heads(:, 1), d.top_heads(:, 2), colors, dla_top));
        ax3.XAxis.TickLabelInterpreter = 'none';        % token text shown as is
        ax3.YAxis.TickLabelInterpreter = 'tex';         % lets the y labels use colour
        for k = 1:size(A, 1)
            for j = find(A(k, :) > 5)                   % label cells above 5%
                text(j, k, sprintf('%.0f', A(k, j)), 'HorizontalAlignment', 'center', ...
                     'Color', [1 1 1] * (A(k, j) > 50), 'FontSize', 8);   % white on dark cells
            end
        end
        ylabel('Head (layer.head, DLA)');
        nX = size(A, 1) - min(nTop_heads, numel(d.heads));   % heads added by extra_heads
        if nX > 0, extraNote = sprintf(' + %d chosen', nX); else, extraNote = ''; end
        title({'', sprintf('Where the top %d heads%s look from position %d', min(nTop_heads, numel(d.heads)), extraNote, position)});
    end

    title(tl, sprintf('DLA: %s, after position %d   (total %.3f = model %.3f)', what, position, d.total, d.logit), ...
          'Interpreter', 'none');
end
end
