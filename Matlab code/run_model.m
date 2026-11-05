% run_model.m  (was run_gpt2.m)
% Run GPT-2 (any size) or Pythia: generate nNew tokens from a new prompt, running
% every step in order:
%   0. (once ever) export all weights            model_weights_file -> export_gpt2_weights_lowmem.py
%                                                (Pythia: export_pythia_weights.py)
%   1. tokenize the prompt                       model_tokenize -> tokenize_prompt.py
%   2. load the weights
%   3. generate nNew tokens                      model_generate
%   4. show each generated token                 model_token, model_decode
%   5. (optional) logit lens: tables + plots     logit_lens
%   6. (optional) direct logit attribution       direct_logit_attrib
%   7. (optional) save the activations           save
% For other layers or positions afterwards: logit_lens(x, W, L, true, positions).
%
% Python is only used to tokenize (tokenizer only, ~1 s) and, the very first
% time, to export the weights. Everything else runs in MATLAB.
%
% Set the prompt and options below, then run.

%% Settings
model           = 'gpt2-small';   % 'gpt2-small' (12 layers, fast) ... 'gpt2-xl' (48 layers, ~15 s to load, ~5 s per token),
                               % or 'pythia-160m' (12 layers; its own tokenizer)
checkpoint      = [];          % [] = the trained model; or ONE training step to run that checkpoint instead, e.g. 128
                               % 
                               % (on disk: gpt2-small, gpt2-small-battlestar (Stanford CRFM seeds alias x21, battlestar x49),
                               % pythia-160m, pythia-160m-seed1, pythia-160m-seed2 (PolyPythias); see model_checkpoints)

prompt          = 'The Eiffel Tower is in the city of'
nNew            = 1;        % tokens to generate (1 = one next token only)

temperature     = 0;        % 0 = greedy decoding: always the most likely token (deterministic);
                            %     > 0 = sampling, e.g. 0.8 (1 = the model's own probabilities)
                            % "Temperature 0" is shorthand for greedy: dividing the logits by 0 isn't
                            % defined, but as temperature -> 0 all probability goes to the top token.
topK            = [];       % when sampling, only the K most likely tokens ([] = all)
seed            = 1;        % when sampling, fixes the random draws

analysisPositions = 'first';  % positions analysed in steps 5 and 6, 0-based:
                            %   'first' = last prompt token (the prediction that chose the first generated token) — default
                            %   'last'  = last position of x (the prediction that chose the last generated token)
                            %   numbers = those positions, e.g. [4 8] (one figure per position in each step)

showLogitLens   = true;     % true = step 5: top-5 table after the last layer + probability/rank vs layer plot

showDLA         = true;     % true = step 6: direct logit attribution table + plots
DLA_compare     = ' John';  % what step 6 compares the chosen token with:
                            %   'runner-up' = the next most likely token (why it won) — default
                            %   'none'      = no comparison: the chosen token's logit alone
                            %   ' Rome'     = a token you choose: text with its leading space, or a
                            %                 0-based ID (e.g. 10598). Must be a single GPT-2 token.
DLA_nTop_heads  = 6;        % step 6's last panel: where the 5 heads with the largest DLA look;
                            %   'all' = every head (layer.head order, no cell labels, DLA alongside)
DLA_extra_heads = [7 3; 7 9; 8 6; 8 10];       % heads always added to that panel, [layer head] rows, e.g. [5 5; 4 11]
DLA_clim_heads  = [];       % colour range of step 6's head heatmap: [] = automatic (each figure
                            %   scaled to its own largest head); a number, e.g. 3, = fixed ±3,
                            %   so figures from different prompts can be compared by eye

saveActivations = false;    % true = save prompt + generated tokens and all activations (step 7)
folderName      = '';       % folder under Activations\ for the saved file;
                            % '' = made from the prompt ("the_capital_of_greece_is_the_city_of")
                            %      (+ "_gpt2-xl" etc. for models other than gpt2-small)

addpath(fileparts(mfilename('fullpath')));        % this script's folder (Matlab code), on any computer
projectDir  = project_root();

if numel(checkpoint) > 1                                 % one model state per run
    error('checkpoint: give one training step (e.g. 128), not %d. To compare many checkpoints use run_DIM or run_unit_correlations.', numel(checkpoint));
end

% Where step 7 would save; checked now so a clash stops the run before any work
if isempty(folderName)
    folderName = regexprep(lower(regexprep(strtrim(prompt), '[^A-Za-z0-9]+', '_')), '^_+|_+$', '');   % "The cat" -> "the_cat"
    if ~strcmpi(model, 'gpt2-small'), folderName = [folderName '_' lower(model)]; end   % e.g. "the_cat_gpt2-xl"
    if ~isempty(checkpoint), folderName = sprintf('%s_step%d', folderName, checkpoint); end   % e.g. "the_cat_step128"
end
saveFile = fullfile(projectDir, 'Activations', folderName, 'activations.mat');
if saveActivations && isfile(saveFile)
    error('%s already exists. Choose another folderName, or set saveActivations = false.', saveFile);
end

%% Step 0 — Full weights (only if they were never exported)
% Finds the model's weights file (GPT-2 <size> model\gpt2_full_weights.mat). If it
% doesn't exist, model_weights_file exports it once from the downloaded model.
% The weights never change, so after the first time this step does nothing.
if isempty(checkpoint)
    weightsFile = model_weights_file(model);
else                                                     % a training checkpoint (error if not on disk)
    weightsFile = model_checkpoints(model, checkpoint);  weightsFile = weightsFile{1};
    fprintf('%s, training checkpoint at step %d\n', model, checkpoint);
end

%% Step 1 — Tokenize the prompt (Python tokenizer only, ~1 s)
% Splits the prompt into GPT-2's tokens and returns their IDs (e.g. " Paris" -> 6342).
% <|endoftext|> (50256) is put first, as in every saved prompt. The model only
% ever sees these numbers, never the text itself.
ids = model_tokenize(prompt, model);                          % [1 × n], 0-based, starts with <|endoftext|>

%% Step 2 — Load the weights
% Reads all of GPT-2's weights (embeddings, every layer's attention and MLP,
% unembedding) into the struct W. They stay fixed for the whole run; every
% forward pass below just reads them.
W = load(weightsFile);                                  % all weights (small: ~650 MB, ~1 s; XL: ~6.6 GB, ~15 s)
show = @(id) '"' + replace(model_token(id, W), newline, "\n") + '"';   % token text in quotes, newline shown as \n

%% Step 3 — Generate
% Adds nNew tokens one at a time: run the model, pick a token (greedy or
% sampled), append it, repeat. g holds the new tokens, the probability of each
% pick, the top 5 alternatives at every step and the activations of the last
% pass (g.out).
g = model_generate(ids, W, nNew, Temperature=temperature, TopK=topK, Seed=seed);

% Residual stream of the last generation pass, the output of each layer, with its text.
% It covers the prompt + all generated tokens except the last one (never fed
% back in), so its last position is the prediction that chose the last token.
% With nNew = 1 that's just the prompt: its last position chose the one new token.
% (built in one statement, so it replaces any earlier x left in the workspace)
x = struct('resid_post', g.out.resid_post, ...                    % [768 × n × 12]  d_model × position × layer
           'prompt',     prompt, ...                              % the prompt
           'text',       char(model_decode(g.new_ids(1:end-1), W)), ...  % generated text covered by x ('' if nNew = 1)
           'token_ids',  g.out.ids, ...                           % [1 × n] token read at each position (input), 0-based
           'next_ids',   g.ids(2:end));                           % [1 × n] token that followed each position (output):
                                                                  %   a prompt token, or the generated token chosen there
% Probability the model (last layer) gave to each next token:
lastL = size(x.resid_post, 3) - 1;                                % last layer: 11 (small), 47 (XL)
p_last = logit_lens(x, W, lastL);                                   % [50257 × n]  final prediction
x.p_next_ids = p_last(sub2ind(size(p_last), x.next_ids + 1, 1:numel(x.next_ids)));   % [1 × n]
% e.g. p = logit_lens(x, W, L); p(:, end) = what the output of layer L predicted when choosing the last token

%% Step 4 — Show the result
% Prints the prompt with the generated text in [brackets], then a table with
% one row per generated token: its ID, its text, how likely the model thought
% it was, and the runner-up it beat.
fprintf('\n%s[%s]\n', prompt, g.text);
fprintf('\nStep  token_id  token            p(chosen)  runner-up\n');
for t = 1:numel(g.new_ids)
    runnerUp = g.top5_ids(1, t);
    if runnerUp == g.new_ids(t), runnerUp = g.top5_ids(2, t); end
    fprintf('%4d  %8d  %-16s %8.2f%%  %s\n', t, g.new_ids(t), show(g.new_ids(t)), ...
            100 * g.chosen_prob(t), show(runnerUp));
end

% Positions for steps 5 and 6, as 0-based numbers
if strcmpi(analysisPositions, 'first')
    positions = numel(ids) - 1;                         % last prompt token: chose the first generated token
elseif strcmpi(analysisPositions, 'last')
    positions = size(x.resid_post, 2) - 1;              % last position of x: chose the last generated token
else
    positions = analysisPositions;                      % e.g. [4 8]
end

%% Step 5 — Logit lens (optional)
% Prints the top 5 predictions after the last layer (11) and plots, after every
% layer, the probability and rank of the final top-5 tokens: shows at which
% layer the generated token emerged. One figure per position.
if showLogitLens
    logit_lens(x, W, lastL, true, positions);
end

%% Step 6 — Direct logit attribution (optional)
% Splits the logit of the token chosen at each position into the share from
% each component (embedding, every attention head, every MLP), and plots
% them by layer and by head. DLA_compare sets what it is compared with: the
% runner-up (which components made it win), a token you choose, or nothing.
% One table and figure per position; d(k) holds the result for positions(k).
if showDLA
    % The token to compare with: 'runner-up', 'none', or a 0-based ID
    if isnumeric(DLA_compare)
        compareWith = DLA_compare;                      % given as a token ID
    elseif any(strcmpi(DLA_compare, {'runner-up', 'none'}))
        compareWith = lower(char(DLA_compare));
    else
        pieces = model_tokenize(DLA_compare, model);     % [<|endoftext|>, token(s)]
        pieces = pieces(2:end);                         % drop <|endoftext|>
        if numel(pieces) ~= 1
            error(['DLA_compare "%s" is %d tokens, not one: %s (IDs %s). ' ...
                   'Only a single token can be compared; choose another word, or give one ' ...
                   'piece''s ID if that piece alone is meaningful.'], ...
                  DLA_compare, numel(pieces), strjoin('"' + model_token(pieces, W) + '"', ' + '), mat2str(pieces));
        end
        compareWith = pieces;
    end

    close(findall(groot, 'Type', 'figure', 'Tag', 'run_model_dla'));   % figures from the previous run
    clear d
    for k = 1:numel(positions)
        pos = positions(k);
        t = x.next_ids(pos + 1);                        % token that followed that position (the one chosen)
        if strcmp(compareWith, 'runner-up')
            [~, order] = sort(g.out.logits(:, pos + 1), 'descend');
            order = order(order - 1 ~= t);              % the most likely token other than the chosen one
            t = [t, order(1) - 1];                      % [chosen, runner-up]
        elseif isnumeric(compareWith) && compareWith ~= t
            t = [t, compareWith];                       % [chosen, your token]
        elseif isnumeric(compareWith)
            fprintf('\nPosition %d: the chosen token is "%s" itself, so it is attributed on its own.\n', pos, model_token(t, W));
        end                                             % 'none': t stays the chosen token alone
        d(k) = direct_logit_attrib(g.out, W, t, pos, true, DLA_nTop_heads, DLA_clim_heads, DLA_extra_heads);   %#ok<SAGROW>
        % direct_logit_attrib closes its own earlier figure on each call, so
        % re-tag this one to keep one figure per position
        f = findall(groot, 'Type', 'figure', 'Tag', 'direct_logit_attrib');
        off = 40 * mod(k - 1, 8);                       % fan out: each extra position 40 px right and down
        set(f, 'Tag', 'run_model_dla', 'Position', f.Position + [off, -off, 0, 0]);
    end
end

%% Step 7 — Save the activations (optional)
% One file, Activations\<folderName>\activations.mat, with the activations of
% the last generation pass (same positions as x). Same variable names and
% layouts as the Python-saved files, plus the new MATLAB fields and the
% generation record (all generated tokens). Computed in MATLAB (g.out).
if saveActivations
    S = struct();                                       % start empty: no leftovers from an earlier S
    S.model          = model;
    S.checkpoint     = checkpoint;                      % training step ([] = the trained model)
    S.prompt         = prompt;
    S.text           = x.text;                          % generated text covered by the activations
    S.generated_text = char(g.text);                    % all generated text, including the last token
    S.token_ids      = int32(g.out.ids);                % token at each position, 0-based
    S.token_strs     = cellstr(model_token(g.out.ids, W));  % 1 × n cell
    S.next_ids       = int32(x.next_ids);               % token that followed each position
    S.p_next_ids     = x.p_next_ids;                    % its probability at the last layer
    S.seq_len        = int32(numel(g.out.ids));
    S.n_prompt     = int32(numel(ids));                 % positions 1..n_prompt are the prompt
    S.resid_pre    = g.out.resid_pre;                   % 768 × n × 12
    S.resid_mid    = g.out.resid_mid;
    S.resid_post   = x.resid_post;                      % the residual stream from step 3
    S.mlp_out      = g.out.mlp_out;
    S.attn_by_head = g.out.attn_by_head;                % 768 × 12 × n × 12
    S.attn_pattern = g.out.attn_pattern;                % n × n × 12 × 12
    S.mlp_hidden   = g.out.mlp_hidden;                  % 3072 × n × 12
    S.logits       = g.out.logits;                      % 50257 × n
    S.new_ids      = g.new_ids;                         % generation record: all generated tokens
    S.chosen_prob  = g.chosen_prob;
    S.top5_ids     = g.top5_ids;
    S.top5_probs   = g.top5_probs;
    S.settings     = struct('nNew', nNew, 'temperature', temperature, 'topK', topK, 'seed', seed);
    if ~isfolder(fileparts(saveFile)), mkdir(fileparts(saveFile)); end
    save(saveFile, '-struct', 'S');
    fprintf('\nSaved %s\n', saveFile);
end
% x holds the residual stream of the last generation pass (x.resid_post,
% 768 × n × 12) and the text of its positions (x.prompt + x.text); g.out holds
% all other activations of that pass (model_forward output). For the last
% generated token's own activations: model_forward(g.ids, W).
