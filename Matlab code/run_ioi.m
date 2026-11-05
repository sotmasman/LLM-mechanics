% run_ioi.m
% Indirect object identification (IOI): find the circuit GPT-2 small uses to complete
% "When Mary and John went to the store, John gave a drink to" with " Mary" (Lesson 9, tab 13;
% Wang et al. 2022). The answer is the name that appears once (the indirect object, IO), not the
% name that appears twice (the subject, S).
%   1. load the weights; tokenize the prompts; find the positions of IO, S1, S2 (the second S) and END
%   2. the clean and corrupted runs, and the logit difference LD = logit(IO) - logit(S)
%   3. direct logit attribution of LD to every head: which heads write the answer (name movers)
%   4. patching single heads at one position: which heads carry the information that S is repeated,
%      to END (S-inhibition heads) and at S2 (induction and duplicate-token heads)
%   5. do the S-inhibition heads steer the name movers? The name movers' attention to each name, in the
%      clean and corrupted runs, with the S-inhibition heads' clean output copied in at END, and with
%      the S-inhibition heads mean-ablated at END
%   6. print the head classes; plot DLA and the two patching maps (figure 1), and step 5 (figure 2)
% Where the heads look from END: run_model with showDLA (its last panel; DLA_extra_heads adds chosen heads).
%
% Corrupted prompt: S2 replaced by the IO name ("..., Mary gave a drink to"), so the repeated name is
% now Mary and the model answers " John". Patching (denoise, as in run_patching): run the corrupted
% prompt, copy ONE head's output at ONE position in from the clean run, and score
%   (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt):  1 = that head alone restores the clean answer.
% Layers, heads and positions are 0-based (head 9.9 = layer 9, head 9; position 0 = <|endoftext|>);
% arrays are 1-based, hence the + 1.
%
% Needs: model_weights_file, model_tokenize, model_forward, model_token, direct_logit_attrib.

%% Settings
model          = 'gpt2-small';
clean_prompt   = 'When Mary and John went to the store, John gave a drink to';
corrupt_prompt = 'When Mary and John went to the store, Mary gave a drink to';   % S2 -> the IO name
io_name        = ' Mary';     % the indirect object: the answer (single token, with its leading space)
s_name         = ' John';     % the subject: named twice
nShow          = 3;           % heads per class to print
sInh           = [7 3; 7 9; 8 6; 8 10];   % step 5: the S-inhibition heads, [layer head] rows
movers         = [9 6; 9 9; 10 0];        % step 5: the name movers whose attention is measured

%****************************************************

%% Step 1 — Weights, tokens and the four positions
addpath(fileparts(mfilename('fullpath')));   % this script's folder (Matlab code), on any computer
W = load(model_weights_file(model));
nL = size(W.W_in, 3);  nH = size(W.W_Q, 3);              % 12 layers, 12 heads (small)
idC = model_tokenize(clean_prompt);  idX = model_tokenize(corrupt_prompt);   % [1 × n]  token IDs, <|endoftext|> first
if numel(idC) ~= numel(idX), error('The two prompts must have the same number of tokens.'); end
n = numel(idC);
io = model_tokenize(io_name);  io = io(2);              % token IDs of the two names (without <|endoftext|>)
s  = model_tokenize(s_name);   s = s(2);
pS = find(idC == s);                                    % the subject's positions (array indices): S1 and S2
pos = struct('IO', find(idC == io, 1), 'S1', pS(1), 'S2', pS(2), 'END', n);   % END = the last token (" to")
fprintf('IO at %d, S1 at %d, S2 at %d, END at %d (0-based positions)\n', pos.IO - 1, pos.S1 - 1, pos.S2 - 1, pos.END - 1);

%% Step 2 — Clean and corrupted runs; the logit difference
oC = model_forward(idC, W);                             % clean: should prefer " Mary"
oX = model_forward(idX, W);                             % corrupted: S2 is now Mary, so it should prefer " John"
LD = @(o) o.logits(io + 1, end) - o.logits(s + 1, end); % logit(IO) - logit(S) at END (+1: IDs are 0-based)
LDc = LD(oC);  LDx = LD(oX);
fprintf('LD = logit(%s) - logit(%s): clean %+.2f, corrupted %+.2f\n', io_name, s_name, LDc, LDx);

%% Step 3 — Direct logit attribution of LD to every head
% Each head's output at END, read through the final layer norm and W_U as in tab 4: how much it pushes
% " Mary" over " John" directly. The largest positive heads write the answer: the name movers.
d = direct_logit_attrib(oC, W, [io s]);                 % d.heads [head × layer]: each head's share of LDc
dla = d.heads;

%% Step 4 — Patch one head at one position (denoise)
% Run the corrupted prompt; at one head's output (model_forward's edit point 'attn_by_head', [768 × heads × n])
% replace one position by the clean run's value. Done for every head, at END and at S2.
score = @(o) (LD(o) - LDx) / (LDc - LDx);
patchEND = zeros(nH, nL);  patchS2 = zeros(nH, nL);     % [head × layer] scores
for L = 0:nL - 1
    for h = 0:nH - 1
        for at = [pos.END, pos.S2]
            clean_out = oC.attn_by_head(:, h + 1, at, L + 1);              % [768 × 1]  the clean head output there
            swap = {@(v) v, @(v) setcol(v, h + 1, at, clean_out)};         % {unchanged, clean output copied in}
            edit = @(name, l, v) swap{1 + (strcmp(name, 'attn_by_head') && l == L)}(v);
            sc = score(model_forward(idX, W, edit));
            if at == pos.END, patchEND(h + 1, L + 1) = sc; else, patchS2(h + 1, L + 1) = sc; end
        end
    end
end

%% Step 5 — Do the S-inhibition heads steer the name movers?
% The S-inhibition heads attend to S2 (the repeated name) and write little to the answer themselves (run_model,
% Fig. 19). If their job is to tell the name movers which name to avoid, then their output at END alone should
% decide where the name movers look. Four runs, and in each the name movers' attention from END to each name
% (summed over the name's positions) and LD:
%   clean; corrupted (now " Mary" is repeated); corrupted with the S-inhibition heads' clean output copied in at
%   END; clean with the S-inhibition heads mean-ablated at END (their output there replaced by its mean over the
%   prompt's positions, without <|endoftext|>, as in run_ablation).
copyIn = @(name, l, v) sinh_edit(name, l, v, sInh, n, 'copy', oC);
meanAb = @(name, l, v) sinh_edit(name, l, v, sInh, n, 'mean', []);
runs = {'clean', oC, idC;  'corrupted', oX, idX;
        'corrupted + S-inh. clean at END', model_forward(idX, W, copyIn), idX;
        'clean, S-inh. mean-ablated at END', model_forward(idC, W, meanAb), idC};
attIO = zeros(size(movers, 1), size(runs, 1));  attS = attIO;   % [name mover × run] attention to " Mary" / " John"
ldRun = zeros(1, size(runs, 1));
for r = 1:size(runs, 1)
    o = runs{r, 2};  ids = runs{r, 3};  ldRun(r) = LD(o);
    for k = 1:size(movers, 1)
        a = o.attn_pattern(n, :, movers(k, 2) + 1, movers(k, 1) + 1);   % [1 × n]  attention from END
        attIO(k, r) = sum(a(ids == io));  attS(k, r) = sum(a(ids == s));
    end
end

%% Step 6 — Results: the head classes, then the figure
top = @(M, dir) sortrows([M(:), kron((0:nL - 1)', ones(nH, 1)), repmat((0:nH - 1)', nL, 1)], dir);   % [value layer head]
show = @(T) strjoin(compose('%d.%d (%+.2f)', T(1:nShow, 2), T(1:nShow, 3), T(1:nShow, 1)), ', ');
fprintf('Name movers (largest DLA):            %s\n', show(top(dla, -1)));
fprintf('Negative name movers (most negative): %s\n', show(top(dla, 1)));
fprintf('Patched at END (S-inhibition, ...):   %s\n', show(top(patchEND, -1)));
fprintf('Patched at S2 (induction, duplicate): %s\n', show(top(patchS2, -1)));

cm = [linspace(0.15, 1, 64)', linspace(0.3, 1, 64)', ones(64, 1); ones(64, 1), linspace(1, 0.3, 64)', linspace(1, 0.15, 64)'];
close(findall(groot, 'Type', 'figure', 'Tag', 'run_ioi'));   % this script's figure from an earlier run
figure('Color', 'w', 'Position', [60 200 1500 420], 'Tag', 'run_ioi');
tiledlayout(1, 3, 'TileSpacing', 'compact');
maps = {dla, patchEND, patchS2};
titles = {sprintf('a. DLA of logit(%s) - logit(%s), per head', io_name, s_name), ...
          'b. denoise: one head patched at END', 'c. denoise: one head patched at S2'};
cbLabels = {'direct contribution to LD (logits)', 'fraction of LD restored', 'fraction of LD restored'};
for j = 1:3
    ax = nexttile;
    m = max(abs(maps{j}(:)));
    imagesc(0:nL - 1, 0:nH - 1, maps{j});  axis xy;  clim([-m m]);  colormap(ax, cm);
    cb = colorbar;  cb.Label.String = cbLabels{j};
    xlabel('Layer');  ylabel('Head');  title(titles{j});  xticks(0:nL - 1);  yticks(0:nH - 1);
end

figure('Color', 'w', 'Position', [60 100 1500 380], 'Tag', 'run_ioi');   % step 5
tiledlayout(1, size(movers, 1) + 1, 'TileSpacing', 'compact');
runLabels = {'clean', 'corrupted', 'corrupted + S-inh.', 'clean - S-inh.'};
for k = 1:size(movers, 1)
    nexttile;  bar([attIO(k, :); attS(k, :)]');  ylim([0 1]);  grid on;
    xticklabels(runLabels);  ylabel('attention from END');
    title(sprintf('%c. name mover %d.%d', 'a' + k - 1, movers(k, :)));
    if k == 1, legend({io_name, s_name}, 'Location', 'northeast'); end
end
nexttile;  bar(ldRun, 'FaceColor', [0.4 0.4 0.4]);  grid on;  xticklabels(runLabels);
ylabel(sprintf('LD = logit(%s) - logit(%s)', io_name, s_name));  title(sprintf('%c. the answer', 'a' + size(movers, 1)));
fprintf('\nName movers'' attention from END to%s /%s, and LD:\n', io_name, s_name);
for r = 1:size(runs, 1)
    fprintf('%-36s LD %+5.2f |%s\n', runs{r, 1}, ldRun(r), sprintf('  %d.%d %.2f / %.2f', [movers, attIO(:, r), attS(:, r)]'));
end

function v = setcol(v, h, p, x)
    % v [768 × heads × n]: replace head h's output at position p by x
    v(:, h, p) = x;
end

function v = sinh_edit(name, l, v, sInh, p, how, oC)
    % model_forward edit for step 5: at position p, replace each S-inhibition head's output by the clean run's
    % ('copy') or by its mean over the positions after <|endoftext|> ('mean')
    if ~strcmp(name, 'attn_by_head'), return, end
    for j = find(sInh(:, 1) == l)'
        h = sInh(j, 2) + 1;
        if strcmp(how, 'copy'), v(:, h, p) = oC.attn_by_head(:, h, p, l + 1);
        else, v(:, h, p) = mean(v(:, h, 2:end), 3); end
    end
end
