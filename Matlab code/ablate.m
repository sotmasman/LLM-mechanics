function edit = ablate(what, which, how, positions)
% ablate  Make an edit function that removes (ablates) model components.
%   edit = ablate(what, which, how, positions) returns an edit for
%   model_forward / model_generate that knocks out the chosen components:
%     out = model_forward(ids, W, edit);
%
%     what       'head'  attention heads; which = [layer head] rows, e.g. [10 0; 9 8]
%                'mlp'   MLP layers;      which = layers, e.g. 0:7
%                'attn'  whole attention layers (all 12 heads); which = layers
%                'layer' whole layers, attention and MLP together; which = layers. With 'zero'
%                        this nearly skips the layer: it adds only its attention bias b_O
%                        (kept, as for 'attn')
%     which      see above. Layers and heads are 0-based, as everywhere.
%     how        optional: 'zero' (default), 'mean' or 'noise'
%                  'zero'  replace the component's output with 0
%                  'mean'  replace it with its average output over this prompt's
%                          positions (excluding position 0, <|endoftext|>, whose
%                          activations are unusually large in GPT-2). Gentler than
%                          zero: removes what is specific to each token but keeps
%                          the component's typical output.
%                  'noise' replace it with random values (Gaussian) with the same mean and
%                          standard deviation, unit by unit, as its output over this prompt's
%                          positions (again leaving out position 0): keeps the typical output
%                          and its spread, removes what is specific to each token. Random:
%                          set rng(seed) before the run to repeat it
%     positions  optional, 0-based positions to ablate at; [] (default) = all.
%                E.g. 4 = only at " Greece" in "The capital of Greece is the city of".
%
% Notes
%   - 'attn' and 'layer' ablate the heads' outputs; the layer's attention output bias b_O is kept.
%   - Ablation removes a component and asks what breaks. Models are often
%     redundant, so a small effect does not prove a component is unimportant:
%     another component may compensate.
%
% Examples
%   out = model_forward(ids, W, ablate('head', [10 0]));               % zero head 10.0
%   out = model_forward(ids, W, ablate('head', [10 0; 9 8], 'mean'));  % mean-ablate 10.0 and 9.8
%   out = model_forward(ids, W, ablate('mlp', 0:7, 'mean', 4));        % MLPs L0-L7 at position 4 only
%   out = model_forward(ids, W, ablate('layer', 6));                    % remove layer 6 (attention + MLP; b_O kept)
%   rng(1);  out = model_forward(ids, W, ablate('head', [5 5], 'noise'));   % head 5.5 replaced by matched noise
%   g   = model_generate(ids, W, 10, Edit=ablate('head', [9 8]));      % generate without head 9.8
%
% See also model_forward, run_ablation.

if nargin < 3 || isempty(how)
    how = 'zero';
end
if nargin < 4
    positions = [];
end
what = lower(char(what));
how = lower(char(how));
if ~any(strcmp(what, {'head', 'mlp', 'attn', 'layer'}))
    error('ablate:what', 'what must be ''head'', ''mlp'', ''attn'' or ''layer''.');
end
if ~any(strcmp(how, {'zero', 'mean', 'noise'}))
    error('ablate:how', 'how must be ''zero'', ''mean'' or ''noise''.');
end
if strcmp(what, 'head') && size(which, 2) ~= 2
    error('ablate:which', 'For ''head'', which must be [layer head] rows, e.g. [10 0; 9 8].');
end

edit = @(name, L, v) do_ablate(name, L, v, what, which, how, positions);
end

% =========================================================================
function v = do_ablate(name, L, v, what, which, how, positions)
% Called by model_forward at every edit point; changes v only where needed.
switch what
    case 'head'                                    % v = attn_by_head [768 × 12 × n]
        if ~strcmp(name, 'attn_by_head'), return, end
        heads = which(which(:, 1) == L, 2);        % heads to ablate in this layer
        for h = heads(:)'
            v(:, h + 1, :) = replacement(v(:, h + 1, :), how, positions, 3);
        end
    case 'attn'                                    % all heads of the listed layers
        if ~strcmp(name, 'attn_by_head') || ~any(which == L), return, end
        v = replacement(v, how, positions, 3);
    case 'mlp'                                     % v = mlp_out [768 × n]
        if ~strcmp(name, 'mlp_out') || ~any(which == L), return, end
        v = replacement(v, how, positions, 2);
    case 'layer'                                   % attention (all heads) and MLP of the listed layers
        if ~any(which == L), return, end
        if strcmp(name, 'attn_by_head'), v = replacement(v, how, positions, 3); end
        if strcmp(name, 'mlp_out'),      v = replacement(v, how, positions, 2); end
end
end

function v = replacement(v, how, positions, posDim)
% Replace v at the chosen positions (along dimension posDim) with 0, the mean, or matched noise.
n = size(v, posDim);
if isempty(positions)
    pos = 1:n;                                     % all positions (MATLAB index)
else
    pos = positions + 1;
end
idx = repmat({':'}, 1, ndims(v));
idx{posDim} = pos;
switch how
    case 'zero'
        v(idx{:}) = 0;
    case 'mean'
        keep = idx;  keep{posDim} = 2:n;           % average over positions 1..n-1
        if n == 1, keep{posDim} = 1; end
        m = mean(v(keep{:}), posDim);              % the component's typical output
        rep = ones(1, ndims(v));  rep(posDim) = numel(pos);
        v(idx{:}) = repmat(m, rep);
    case 'noise'
        keep = idx;  keep{posDim} = 2:n;           % statistics over positions 1..n-1, as for 'mean'
        if n == 1, keep{posDim} = 1; end
        m = mean(v(keep{:}), posDim);  sd = std(v(keep{:}), 0, posDim);   % each unit's mean and spread
        sz = size(v);  sz(posDim) = numel(pos);
        v(idx{:}) = m + sd .* randn(sz, 'like', v);
end
end
