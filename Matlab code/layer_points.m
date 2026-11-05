function [names, i] = layer_points(nL, mode, which)
% layer_points  Names of the residual-stream points, and the array index of one of them.
%   Convention (all scripts): a point is named by the layer whose OUTPUT it is.
%     'emb'     the embedding, before layer 0
%     '0' ...   the output of layer 0, 1, ..., nL-1 (the last one is the model's output)
%     '6.attn'  inside layer 6: after its attention, before its MLP (mode 'attn' only)
%   names = layer_points(nL)            {'emb', '0', ..., 'nL-1'}                 [1 × nL+1]
%   names = layer_points(nL, 'attn')    {'emb', '0.attn', '0', ..., 'nL-1'}       [1 × 2nL+1]
%   [names, i] = layer_points(nL, mode, which)   also the index of one point in names
%     which   'emb', a layer number (6 or '6'), or '6.attn' (mode 'attn')
%   The order matches the state arrays:
%     mode 'out'  (default)  cat(3, out.resid_pre(:, :, 1), out.resid_post)   (out = model_forward output)
%     mode 'attn'            token_trajectories: emb, then after attention and after the MLP of each layer
%
% Examples:
%   names = layer_points(12);                  % emb, 0, ..., 11  (tick labels)
%   [~, i] = layer_points(12, 'out', 6);       % i = 8: the output of layer 6
%   [~, i] = layer_points(12, 'out', 'emb');   % i = 1
%   [~, i] = layer_points(12, 'attn', '6.attn');   % i = 14
%
% See also token_trajectories, model_forward.

if nargin < 2 || isempty(mode), mode = 'out'; end
outs = compose('%d', 0:nL - 1);                          % {'0', ..., 'nL-1'}
if strcmp(mode, 'attn')
    names = [outs + ".attn"; string(outs)];              % [2 × nL]  row 1: L.attn, row 2: L
    names = [{'emb'}, cellstr(names(:)')];               % emb, 0.attn, 0, 1.attn, 1, ...
else
    names = [{'emb'}, outs];                             % emb, 0, ..., nL-1
end
if nargin < 3, i = []; return; end
if isnumeric(which), which = sprintf('%d', which); end   % 6 -> '6'
i = find(strcmp(names, char(which)));
if isempty(i)
    error('layer_points:which', '"%s" is not a point; use ''emb'', a layer 0-%d%s.', char(which), nL - 1, ...
          repmat(' or ''L.attn''', 1, strcmp(mode, 'attn')));
end
end
