% run_ablation.m
% Remove (ablate) model components and see how the prediction changes:
%   1. tokenize the prompt and load the weights
%   2. clean run: the model's prediction, and the target and competitor tokens
%   3. ablated run: the same prompt with the chosen components removed
%   4. compare: p(target), p(competitor) and their logit difference, clean vs
%      ablated; and, if several components are listed, each one on its own
%   5. (optional) DLA of the ablated run
%
% Needs: model_tokenize, model_forward, ablate, model_token, direct_logit_attrib.
% Set the prompt and the ablation below, then run.

%% Settings
model       = 'gpt2-small';     % 'gpt2-small' or 'gpt2-xl'
prompt      = 'Code: blue seven turtle elbow spoon granite. Code: blue seven turtle';

ablate_type      = 'layer';      % 'head', 'mlp', 'attn' or 'layer' (attention + MLP)
ablate_which     = [4:5];;       % 'head': [layer head] rows; 'mlp'/'attn'/'layer': layers, e.g. 0:7
ablate_how       = 'zero';      % 'zero', 'noise' (random, same mean and sd), 'mean' (average over the prompt's positions) 
ablate_positions = [];          % 0-based positions to ablate at; [] = all

target      = '';              % token to track, e.g. ' Athens'; '' = the clean run's top token
competitor  = 'runner-up';     % token to compare with: 'runner-up' (clean run's 2nd) or e.g. ' Pir'

showDLA     = true;            % true = DLA plot of the ablated run (target vs competitor)

%% Step 1 — Tokenize and load
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
ids = model_tokenize(prompt);                            % [1 × n], 0-based IDs
W   = load(model_weights_file(model));
q   = numel(ids) - 1;                                   % last position: where the next token is predicted
tok = @(s) local_token(s);                              % text (or ID) -> single token ID (see bottom)

%% Step 2 — Clean run
clean = model_forward(ids, W);
[~, order] = sort(clean.probs, 'descend');
if isempty(target), tId = order(1) - 1; else, tId = tok(target); end
if strcmpi(competitor, 'runner-up')
    order = order(order - 1 ~= tId);  cId = order(1) - 1;
else
    cId = tok(competitor);
end

%% Step 3 — Ablated run
edit = ablate(ablate_type, ablate_which, ablate_how, ablate_positions);
abl  = model_forward(ids, W, edit);

%% Step 4 — Compare
if isempty(ablate_positions), where = 'all positions'; else, where = ['positions ' mat2str(ablate_positions)]; end
if strcmp(ablate_type, 'head')
    names = compose('%d.%d', ablate_which(:, 1), ablate_which(:, 2));
else
    names = compose('L%d', ablate_which(:));
end
fprintf('\n"%s"   target %s, competitor %s\n', prompt, show_tok(tId), show_tok(cId));
fprintf('Ablation: %s %s, %s, %s\n\n', ablate_type, strjoin(names, ', '), ablate_how, where);
fprintf('%-26s %10s %12s %12s   %s\n', 'run', 'p(target)', 'p(competitor)', 'logit diff', 'top 3 tokens (p)');
show_row('clean', clean, tId, cId);
show_row('ablated (all listed)', abl, tId, cId);
nItems = size(ablate_which, 1);  if ~strcmp(ablate_type, 'head'), nItems = numel(ablate_which); end
if nItems > 1                                           % each listed component on its own
    for k = 1:nItems
        if strcmp(ablate_type, 'head'), one = ablate_which(k, :); else, one = ablate_which(k); end
        show_row(['  only ' names{k}], model_forward(ids, W, ablate(ablate_type, one, ablate_how, ablate_positions)), tId, cId);
    end
end

%% Step 5 — DLA of the ablated run (optional)
if showDLA
    d_abl = direct_logit_attrib(abl, W, [tId cId], q, true);
end

% -------------------------------------------------------------------------
function show_row(label, o, tId, cId)
% One line of the comparison table: p(target), p(competitor), logit difference, the 3 most likely tokens.
    [pTop, top] = maxk(o.probs, 3);
    top3 = strjoin(arrayfun(@(k) sprintf('%s %.1f%%', show_tok(top(k) - 1), 100 * pTop(k)), 1:3, 'UniformOutput', false), ', ');
    fprintf('%-26s %9.2f%% %11.2f%% %+12.2f   %s\n', label, 100 * o.probs(tId + 1), 100 * o.probs(cId + 1), ...
            o.logits(tId + 1, end) - o.logits(cId + 1, end), top3);
end

function s = show_tok(id)
% A token's text in quotes; a line break or tab is shown as \n or \t (GPT-2 has tokens for them)
    s = '"' + strrep(strrep(string(model_token(id)), newline, '\n'), char(9), '\t') + '"';
end

function id = local_token(s)
    if isnumeric(s), id = s; return, end
    p = model_tokenize(s);  p = p(2:end);
    if numel(p) ~= 1
        error('"%s" is %d GPT-2 tokens (%s); use a single-token word.', s, numel(p), strjoin(model_token(p), ' + '));
    end
    id = p;
end
