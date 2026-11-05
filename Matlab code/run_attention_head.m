% run_attention_head.m
% One attention head at one token, step by step: recompute by hand what model_forward does inside the head,
% check each result against it, and plot where the head looks and what it brings back.
%   steps 1-4  where the head looks:   query, keys, scores, softmax  -> attention weights A (4b: every query)
%   steps 5-6  what it brings back:    values, weighted sum, W_O     -> the head's write z
%   step 7     its direct logit attribution (DLA) for the target token, split by source position
%   step 8     K-composition check: silence an earlier head and recompute A
%   step 9     figure: a. the head's full attention pattern (query × key), the examined query row outlined
%                      b. scores  c. attention weights (and with the earlier head silenced)
%                      d. what each position's value says about the target  e. attention × value = DLA by source
% Default: head 5.5 of GPT-2 small at the second " seven" of "Code: blue seven turtle elbow spoon granite."
% written twice: 0.893 of its attention on the first " turtle", DLA +0.295 for " turtle".
% Layers, heads and positions are 0-based (head 5.5 = layer 5, head 5; position 0 = <|endoftext|>);
% arrays are 1-based, hence the + 1.
%
% Needs: model_weights_file, model_tokenize, model_forward, model_token.

%% Settings
model       = 'gpt2-small';
prompt      = 'Code: blue seven turtle elbow spoon granite. Code: blue seven turtle elbow spoon granite.';
L = 5;  hd = 5;          % the head: layer L, head hd
queryPos    = 13;        % the token to examine (0-based position; printed below). 13 = the second " seven"
target      = [];        % token for the DLA in step 7, e.g. ' turtle'; [] = the token that comes next in the prompt
silenceHead = [4 11];    % silenced in step 8: [layer head] = one head; [layer] = all heads of that layer; [] = skip

%****************************************************

%% Step 0 — Run the model once; steps 1-7 recompute pieces of this run by hand and check them against it
W = load(model_weights_file(model));                        % all weights (W_Q, W_K, W_V, W_O, W_U, ...)
ids = model_tokenize(prompt, W);                            % [1 × n]  token IDs; ids(1) = <|endoftext|>, added at the start
out = model_forward(ids, W);  n = numel(ids);               % the full forward pass: every layer's residual stream,
                                                            % attention patterns and head outputs, kept in out
lab = arrayfun(@(i) string(model_token(i, W)), ids);  lab(1) = "<eot>";   % each token's text, for the tables
fprintf('positions: %s\n', strjoin(compose('%d%s', (0:n - 1)', lab'), ' |'));   % e.g. "0<eot> |1Code |2: |3 blue ..."
i = queryPos + 1;                                           % array index of the query (MATLAB counts from 1)

%% Step 1 — The stream the head reads: the residual stream entering layer L, LayerNormed
% (LayerNorm's gain and bias are folded into W_Q, W_K, W_V, so only centre and scale here)
% Each token (column) is centred by its mean and divided by its standard deviation, both taken across its 768
% numbers ("layer" norm: per position, across features). The s.d. divides by N = 768, not N - 1 (MATLAB's
% std(x) would use N - 1; std(x, 1, 1) matches). 1e-5 is LayerNorm's epsilon (GPT-2's value): it keeps the
% division safe if a vector were constant and is otherwise negligible, so the denominator is sqrt(variance + eps).
x = out.resid_pre(:, :, L + 1);                             % [768 × n]  one column per token
mu = mean(x, 1);  h = (x - mu) ./ sqrt(mean((x - mu).^2, 1) + 1e-5);

%% Step 2 — Queries and keys: what each token looks for, and what it offers
q = W.W_Q(:, :, hd + 1, L + 1) * h + W.b_Q(:, hd + 1, L + 1);   % [64 × n]
k = W.W_K(:, :, hd + 1, L + 1) * h + W.b_K(:, hd + 1, L + 1);   % [64 × n]

%% Step 3 — Scores: the query at position i against every earlier key, divided by sqrt(64) = 8
% (later keys are left out: the causal mask). Why sqrt(64): a dot product of two 64-number vectors grows with
% their length (typical size ~sqrt(64) = 8 for random entries), and large scores saturate the softmax in step 4
% (nearly all weight on one position, tiny gradients in training); dividing by sqrt(d_head) keeps scores ~1.
s = (q(:, i)' * k(:, 1:i)) / 8;                             % [1 × i]

%% Step 4 — Softmax: scores -> attention weights (all positive, sum to 1)
% The weights are mixing proportions, not probabilities of an event: the head's output (step 6) is a weighted
% average of the values, so it does not grow with the prompt's length. exp() makes every weight positive and
% stretches the differences between scores ("soft maximum": the top position gets most, not all, of the weight);
% positions compete, since raising one weight lowers the rest. Because the weights must sum to 1, a head with
% nothing to do still has to look somewhere: many park on <|endoftext|> (an "attention sink").
a = exp(s - max(s));                                        % subtracting the max changes nothing but avoids overflow
A = a / sum(a);                                             % [1 × i]  row i of the attention pattern
fprintf('\nquery: position %d "%s", head %d.%d\n', i - 1, lab(i), L, hd);
fprintf('%4s %-12s %8s %8s %10s %7s\n', 'pos', 'token', 'q.k', 'score', 'exp(s-max)', 'A');
for j = 1:i, fprintf('%4d %-12s %8.2f %8.3f %10.4f %7.3f\n', j - 1, lab(j), q(:, i)' * k(:, j), s(j), a(j), A(j)); end
[~, jMax] = max(A);                                         % the position the head attends to most
fprintf('sum exp = %.4f; A(%d "%s") = %.4f, stored attn_pattern = %.4f\n', sum(a), jMax - 1, lab(jMax), A(jMax), out.attn_pattern(i, jMax, hd + 1, L + 1));

% 4b. The full attention pattern: steps 3-4 for every query at once (figure panel a; row i is A above)
S = (q' * k) / 8;                                           % [n × n]  every query (row) against every key (column)
S(triu(true(n), 1)) = -Inf;                                 % causal mask: a query sees only itself and earlier keys
Aall = exp(S - max(S, [], 2));  Aall = Aall ./ sum(Aall, 2);   % softmax of each row; exp(-Inf) = 0 above the diagonal
fprintf('full pattern: stored attn_pattern matches to %.1e\n', max(abs(Aall - out.attn_pattern(:, :, hd + 1, L + 1)), [], 'all'));

%% Step 5 — Values: what each position passes on if attended to
v = W.W_V(:, :, hd + 1, L + 1) * h + W.b_V(:, hd + 1, L + 1);   % [64 × n]

%% Step 6 — The head's write: values mixed by the attention weights, projected back to the stream by W_O
% z is one head's part of the attention layer's output to the residual stream. All heads of layer L write at
% once; the full attention output of the layer is the sum over its 12 heads plus the bias b_O, e.g. for layer 5:
%     attention output of layer 5 = z_5.0 + z_5.1 + ... + z_5.11 + b_O
% and that sum is added to the residual stream (model_forward: resid_mid = resid_pre + that sum).
r = v(:, 1:i) * A';                                         % [64 × 1]   weighted sum of values
z = W.W_O(:, :, hd + 1, L + 1) * r;                         % [768 × 1]  added to the residual stream (with the other heads and b_O)
fprintf('\nhead write z: |z| = %.2f; stored attn_by_head matches to %.1e\n', norm(z), max(abs(z - out.attn_by_head(:, hd + 1, i, L + 1))));

%% Step 7 — DLA: how much the head's write z raises the target token's logit directly
% The model's logit for token t is  e · LayerNorm(x_final) + b_U(t),  where x_final is the residual stream
% after the last layer and e is token t's row of the unembedding W_U. x_final is a sum of every component's
% write, z among them. LayerNorm subtracts the mean (linear) and divides by one scale, the s.d. of x_final
% (sdFinal, sigma in the lesson); holding it at its actual value makes the whole thing linear, so z's own share is
%     DLA = e · (z - mean(z)) / sdFinal          (as in direct_logit_attrib)
% And since z = sum over j of A(j) × (position j's value through W_O), the DLA splits by source position.

% 7a. The target token t: the next token in the prompt, unless set above
if isempty(target) && i < n
    t = ids(i + 1);                                         % default: the token that actually comes next
elseif isempty(target)
    [~, t] = max(out.logits(:, i));  t = t - 1;             % last position: the model's top prediction
else
    t = model_tokenize(target, W);  t = t(2);               % model_tokenize puts <|endoftext|> first
end
tStr = string(model_token(t, W));
fprintf('target "%s"\n', tStr);

% 7b. e: the direction in the stream that the target's logit reads (token t's unembedding vector)
% GPT-2 ties its unembedding to the token embedding E_t, after the final LayerNorm's learned scale gamma and shift
% beta (768 numbers each): logit(t) = (gamma .* xhat + beta) · E_t, with xhat the centred, scaled stream. Our
% weights file folds them in: W_U row t = gamma .* E_t (centred over tokens), b_U(t) = beta · E_t. So e already
% includes gamma, and 7c only centres and divides by sdFinal. (gamma is ~1.25 in most dimensions, 10-15 in a few,
% near 0 in a few.)
e = double(W.W_U(t + 1, :))';                               % [768 × 1]

% 7c. sdFinal: the final LayerNorm's scale at this position (sigma), held fixed
xf = double(out.resid_post(:, i, end));                     % [768 × 1]  x_final (stream leaving the last layer)
sdFinal = sqrt(mean((xf - mean(xf)).^2) + 1e-5);            % its standard deviation

% 7d. Each source position's value, written into the stream (W_O × v_j), and what it says about the target
% A and says are independent factors from the head's two circuits: A(j) (QK: W_Q, W_K) is where the head looks and
% does not depend on the target; says(j) (OV: W_V, W_O, then e) is what position j's content would add to the
% target's logit and does not depend on the query. DLA = sum over j of A(j) × says(j): a position contributes only
% if the head both looks there and finds something there that pushes the target. In an ideal copying head, says
% would be large only where the target token sits; for 5.5 it is ~0.3 at " turtle", " elbow" and " spoon" alike,
% so the head's specificity comes almost entirely from A.
u = double(W.W_O(:, :, hd + 1, L + 1) * v(:, 1:i));         % [768 × i]  column j: what attending fully to j would write
says = e' * (u - mean(u, 1)) / sdFinal;                     % [1 × i]    its push on the target's logit

% 7e. Weight each by the attention, A(j) × says(j); the sum is the head's DLA
% The sum (+0.295 for the default) is the head's direct push on the target's logit at this position, in logits:
% its share in an exact split of the logit into the embedding, every head, every b_O, every MLP and b_U (with
% sdFinal held fixed), the same number direct_logit_attrib reports for this head. Raising one logit by 0.295
% multiplies the target's odds against an unaffected competitor by exp(0.295) = 1.34. It is direct only: effects
% passed through later heads or MLPs are credited to them, so it is not what removing the head would do
% (that is ablation).
fprintf('%4s %-12s %7s %12s %10s\n', 'pos', 'token', 'A', 'says', 'A.*says');
for j = 1:i, fprintf('%4d %-12s %7.3f %12.2f %10.3f\n', j - 1, lab(j), A(j), says(j), A(j) * says(j)); end
fprintf('sum = %.3f = head %d.%d''s DLA for "%s" (A on "%s" = %.3f is an attention weight, not DLA)\n\n', ...
        A * says', L, hd, tStr, lab(jMax), A(jMax));

%% Step 8 — K-composition check: silence an earlier head and recompute the weights
% Default: 5.5's key at " turtle" matches partly because the previous-token head 4.11 wrote "my previous
% token was seven" there. Only heads in earlier layers (silenceHead(1) < L) can change this head's pattern.
% Silencing (zero-ablation): model_forward reruns with the edit function silence (bottom of this script), which at
% the point 'attn_by_head' of that layer sets the head's write to zero at every position: v(:, head, :) = 0, its 768
% numbers per position after W_O (the z of step 6). The head still runs; nothing it writes reaches the stream. The
% layer's output becomes the other heads + b_O, and everything later (including this head's queries, keys and A) is
% recomputed. Zeroing changes the stream at every position, the query's too, so the drop is not purely through the
% keys; tab 11's ablate also offers mean-ablation (the head's average write instead of zero), a gentler test.
% With a single number, silenceHead = [layer], every head of that layer is silenced: the layer's attention output
% is then b_O alone (the MLPs are untouched).
if ~isempty(silenceHead)
    if isscalar(silenceHead), sName = sprintf('all heads of layer %d', silenceHead);  else, sName = sprintf('head %d.%d', silenceHead); end
    o2 = model_forward(ids, W, @(name, Lx, v) silence(name, Lx, v, silenceHead));
    A2 = double(o2.attn_pattern(i, 1:i, hd + 1, L + 1));   % [1 × i]  the same row, with the earlier head(s) silenced
    fprintf('without %s: A(%d "%s") = %.3f (was %.3f), A(<eot>) = %.3f (was %.3f)\n', sName, jMax - 1, lab(jMax), ...
            A2(jMax), A(jMax), A2(1), A(1));
end

%% Step 9 — Figure: a. the full attention pattern; b-e. one bar per earlier position (the key) for query i
pos = 0:i - 1;  qName = sprintf('position %d "%s"', i - 1, strtrim(lab(i)));
close(findall(groot, 'Type', 'figure', 'Tag', 'run_attention_head'));   % this script's figures from earlier runs
figure('Color', 'w', 'Position', [40 40 2200 900], 'Tag', 'run_attention_head');  tiledlayout(2, 3, 'TileSpacing', 'compact');
nexttile(1, [2 1]);                                         % a. the full pattern, query × key (left column, both rows)
imagesc(0:n - 1, 0:n - 1, Aall);  axis square;  clim([0 1]);
colormap(gca, [linspace(1, 0.05, 256)', linspace(1, 0.2, 256)', linspace(1, 0.45, 256)']);  cb = colorbar;  cb.Label.String = 'attention weight';
rectangle('Position', [-0.5, i - 1.5, n, 1], 'EdgeColor', [0.85 0.33 0.10], 'LineWidth', 2.5);   % the query row shown in b-e
set(gca, 'XTick', 0:n - 1, 'XTickLabel', lab, 'YTick', 0:n - 1, 'YTickLabel', lab, 'XTickLabelRotation', 60, 'TickLabelInterpreter', 'none');
xlabel('key (token attended to)');  ylabel('query (current token)');
title({sprintf('a. attention pattern of head %d.%d (step 4b)', L, hd), sprintf('every query; outlined: query %s, shown in b-e', qName)});
axB = gobjects(1, 4);
axB(1) = nexttile(2);  bar(pos, s, 'FaceColor', [0.5 0.5 0.5]);   % b. scores
title({sprintf('b. scores q_i \\cdot k_j / 8 (steps 1-3), head %d.%d', L, hd), sprintf('query: %s', qName)});  ylabel('score');
axB(2) = nexttile(3);  hold on                              % c. attention weights
bar(pos, A, 'FaceColor', [0 0.45 0.74]);
if ~isempty(silenceHead)
    bar(pos, A2, 'FaceColor', 'none', 'EdgeColor', [0.85 0.33 0.10], 'LineWidth', 1.5);
    legend({'attention', [sName ' silenced']}, 'Location', 'northwest');
end
text(jMax - 1, A(jMax), sprintf('%.2f', A(jMax)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
hold off;  ylim([0 1.1]);  ylabel('attention weight');  title({'c. attention weights: softmax of b (step 4) = row of a', 'sum to 1; outlined: earlier head(s) silenced (step 8)'});
axB(3) = nexttile(5);  bar(pos, says, 'FaceColor', [0.47 0.67 0.19]);   % d. what each value says about the target
ylabel('logit');  title({sprintf('d. what each value says about "%s" (steps 5, 7a-7d)', strtrim(tStr)), '(W_O v_j \cdot e_t / \sigma: the push if the head attended only there)'});
axB(4) = nexttile(6);  bar(pos, A .* says, 'FaceColor', [0.49 0.18 0.56]);   % e. attention × value
ylabel('logit');  title({sprintf('e. attention \\times value: DLA by source (steps 6, 7e), total %+.3f', A * says'), sprintf('(the head''s direct push on "%s")', strtrim(tStr))});
for ax = axB                                                % the bar panels share one x-axis: the keys up to query i
    set(ax, 'XTick', pos, 'XTickLabel', lab(1:i), 'XTickLabelRotation', 60, 'TickLabelInterpreter', 'none');  grid(ax, 'on');
    xlim(ax, [-0.7 i - 0.3]);  xlabel(ax, 'key position (token attended to)');
end

function v = silence(name, L, v, lh)
% model_forward edit: zero the write of head lh = [layer head] (0-based), or of every head of layer lh = [layer]
if strcmp(name, 'attn_by_head') && L == lh(1)
    if isscalar(lh), v(:) = 0;  else, v(:, lh(2) + 1, :) = 0; end
end
end
