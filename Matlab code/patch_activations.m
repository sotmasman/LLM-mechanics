function edit = patch_activations(src, type, which, positions)
% patch_activations  Make an edit that copies activations from another run.
%   edit = patch_activations(src, type, which, positions) returns an edit for
%   model_forward that replaces chosen activations of the current run with
%   the same activations from a source run (another prompt):
%     src = model_forward(ids_source, W);
%     out = model_forward(ids_target, W, patch_activations(src, 'mlp', 3, 4));
%   The two prompts must have the same number of tokens (positions line up).
%
%     src        model_forward output of the source prompt
%     type       'resid'  residual stream ENTERING layer L (everything so far);
%                         which = layers
%                'mlp'    MLP output of layer L;              which = layers
%                'attn'   attention output of layer L (all heads); which = layers
%                'head'   one head's output; which = [layer head] rows
%     which      see above; layers and heads are 0-based
%     positions  0-based positions to patch; [] (default) = all positions
%
% Example: does Greece's layer-3 MLP output at " Greece" (position 4) make
% the Italy prompt say Athens?
%   g = model_forward(model_tokenize('The capital of Greece is the city of'), W);
%   i = model_tokenize('The capital of Italy is the city of');
%   out = model_forward(i, W, patch_activations(g, 'mlp', 3, 4));
%
% See also run_patching, ablate, model_forward.

if nargin < 4
    positions = [];
end
type = lower(char(type));
if ~any(strcmp(type, {'resid', 'mlp', 'attn', 'head'}))
    error('patch_activations:type', 'type must be ''resid'', ''mlp'', ''attn'' or ''head''.');
end
if strcmp(type, 'head') && size(which, 2) ~= 2
    error('patch_activations:which', 'For ''head'', which must be [layer head] rows, e.g. [10 0].');
end
edit = @(name, L, v) do_patch(name, L, v, src, type, which, positions);
end

% =========================================================================
function v = do_patch(name, L, v, src, type, which, positions)
% Called by model_forward at every edit point; copies from src where needed.
n = size(src.resid_pre, 2);
if isempty(positions), pos = 1:n; else, pos = positions + 1; end      % MATLAB index
switch type
    case 'resid'                                   % v = resid_pre [768 × n]
        if strcmp(name, 'resid_pre') && any(which == L)
            v(:, pos) = src.resid_pre(:, pos, L + 1);
        end
    case 'mlp'                                     % v = mlp_out [768 × n]
        if strcmp(name, 'mlp_out') && any(which == L)
            v(:, pos) = src.mlp_out(:, pos, L + 1);
        end
    case 'attn'                                    % v = attn_by_head [768 × 12 × n]
        if strcmp(name, 'attn_by_head') && any(which == L)
            v(:, :, pos) = src.attn_by_head(:, :, pos, L + 1);
        end
    case 'head'
        if strcmp(name, 'attn_by_head')
            for h = which(which(:, 1) == L, 2)'
                v(:, h + 1, pos) = src.attn_by_head(:, h + 1, pos, L + 1);
            end
        end
end
end
