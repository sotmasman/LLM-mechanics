function [files, steps, tokPerStep, names] = model_checkpoints(model, checkpointRange)
% model_checkpoints  Weights files of a model's training checkpoints on disk.
%   [files, steps, tokPerStep, names] = model_checkpoints(model, checkpointRange)
%     model            'gpt2-small' (Stanford CRFM, seed alias x21), 'gpt2-small-battlestar' (seed x49),
%                      'gpt2-small-run1' (our replication of Stanford's settings, 10,000 steps),
%                      'gpt2-small-eta1', 'gpt2-small-eta2', 'gpt2-small-eta3', 'gpt2-small-eta4', 'gpt2-small-arch1' (our variants of run 1), 'gpt2-6l-512' (size grid), 'gpt2-2l-768', 'gpt2-1l-768',
%                      'pythia-160m', or the
%                      PolyPythias seeds 'pythia-160m-seed1', 'pythia-160m-seed2' (other seeds of Pythia-160M),
%                      or 'pythia-410m' (24 layers x 1024)
%     checkpointRange  the training steps wanted, e.g. 0:2000 (every checkpoint on disk from
%                      0 to 2,000) or 0:100:2000 (multiples of 100); [] = the final model only
%   files       {1 × n} weights files (gpt2_full_weights.mat)
%   steps       [1 × n] their training steps ([] for the final model)
%   tokPerStep  training tokens per step (sequences per batch × tokens per sequence):
%               Stanford GPT-2 512 × 1024, Pythia 1024 × 2048
%   names       {1 × n} labels, e.g. 'pythia-160m step 128'
%
% See also run_unit_correlations, run_DIM, model_weights_file.

ck = struct('gpt2_small',        struct('folder', 'GPT-2 small checkpoints (alias x21)', 'tokPerStep', 512 * 1024), ...
            'gpt2_small_battlestar', struct('folder', 'GPT-2 small checkpoints (battlestar x49)', 'tokPerStep', 512 * 1024), ...   % 2nd Stanford seed
            'gpt2_small_run1',   struct('folder', 'GPT-2 small run1 checkpoints',       'tokPerStep', 512 * 1024), ...   % our replication (train_gpt2.py, lab computer)
            'gpt2_small_eta1',   struct('folder', 'GPT-2 small eta1 checkpoints',       'tokPerStep', 512 * 1024), ...   % Eta1 · run 1, linear warmup
            'gpt2_small_eta2',   struct('folder', 'GPT-2 small eta2 checkpoints',       'tokPerStep', 512 * 1024), ...   % Eta2 · run 1, constant learning rate 3e-4
            'gpt2_small_eta3',   struct('folder', 'GPT-2 small eta3 checkpoints',       'tokPerStep', 512 * 1024), ...   % Eta3 · run 1, constant learning rate 5e-4
            'gpt2_small_eta4',   struct('folder', 'GPT-2 small eta4 checkpoints',       'tokPerStep', 512 * 1024), ...   % Eta4 · run 1, constant learning rate 1e-4
            'gpt2_small_arch1',  struct('folder', 'GPT-2 small arch1 checkpoints',      'tokPerStep', 512 * 1024), ...   % Arch1 · GPT-2 small, Llama 2 block
            'gpt2_6l_512',       struct('folder', 'GPT2-6L-512 checkpoints',            'tokPerStep', 512 * 1024), ...   % size grid: 6 layers x 512, constant 3e-4
            'gpt2_2l_768',       struct('folder', 'GPT2-2L-768 checkpoints',            'tokPerStep', 512 * 1024), ...   % depth: 2 layers x 768, constant 3e-4
            'gpt2_1l_768',       struct('folder', 'GPT2-1L-768 checkpoints',            'tokPerStep', 512 * 1024), ...   % depth: 1 layer x 768 (no induction circuit possible)
            'gpt2_4l_768',       struct('folder', 'GPT2-4L-768 checkpoints',            'tokPerStep', 512 * 1024), ...   % depth: 4 layers x 768, 7,500 steps (control of:)
            'gpt2_4l_768_syncp1',   struct('folder', 'GPT2-4L-768 sync+1 checkpoints',  'tokPerStep', 512 * 1024), ...   % synchrony penalty lambda +1, steps 0-500
            'gpt2_4l_768_syncm005', struct('folder', 'GPT2-4L-768 sync-0.05 checkpoints', 'tokPerStep', 512 * 1024), ... % synchrony reward lambda -0.05, steps 0-500
            'gpt2_4l_768_cosp1',    struct('folder', 'GPT2-4L-768 cos+1 checkpoints',   'tokPerStep', 512 * 1024), ...   % whole-layer cosine penalty lambda +1, steps 0-500
            'gpt2_4l_768_syncp1all', struct('folder', 'GPT2-4L-768 sync+1 all checkpoints', 'tokPerStep', 512 * 1024), ... % synchrony penalty lambda +1, all 7,500 steps
            'gpt2_4l_768_cosp1all', struct('folder', 'GPT2-4L-768 cos+1 all checkpoints', 'tokPerStep', 512 * 1024), ...  % whole-layer cosine penalty lambda +1, all 7,500 steps
            'pythia_160m',       struct('folder', 'Pythia 160M checkpoints',             'tokPerStep', 1024 * 2048), ...
            'pythia_160m_seed1', struct('folder', 'Pythia 160M seed1 checkpoints',       'tokPerStep', 1024 * 2048), ...   % PolyPythias
            'pythia_160m_seed2', struct('folder', 'Pythia 160M seed2 checkpoints',       'tokPerStep', 1024 * 2048), ...   % (EleutherAI 2025)
            'pythia_410m',       struct('folder', 'Pythia 410M checkpoints',             'tokPerStep', 1024 * 2048));      % 24 layers x 1024
key = strrep(lower(char(model)), '-', '_');
if startsWith(char(model), 'sync-pilot-')        % synchrony-penalty pilot arms (sync_penalty_pilot.py): 'sync-pilot-lam-1'
    key = 'sync_pilot';                            % (cosine penalty), 'sync-pilot-ucorr-1' (unit-correlation penalty);
    arm = regexprep(extractAfter(char(model), 'sync-pilot-'), '(ucorr|coslayer)', '$1_lam');   % lam may contain '.' or '-'
                                                   % ('sync-pilot-4l768_coslayer1': whole-layer cosine penalty)
    ck.(key) = struct('folder', ['Sync pilot ' strrep(arm, 'lam', 'lam_') ' checkpoints'], 'tokPerStep', 32 * 1024);
end
if isempty(checkpointRange)
    files = {model_weights_file(model)};  steps = [];  names = {char(model)};
    tokPerStep = NaN;  if isfield(ck, key), tokPerStep = ck.(key).tokPerStep; end
    return
end
if ~isfield(ck, key)
    error('model_checkpoints:none', 'No checkpoints on disk for %s (only %s).', model, strjoin(strrep(fieldnames(ck), '_', '-'), ', '));
end
dirC = fullfile(project_root(), ck.(key).folder);
f = dir(fullfile(dirC, 'checkpoint-*'));
onDisk = sort(cellfun(@(n) sscanf(n, 'checkpoint-%d'), {f.name}));
steps = onDisk(ismember(onDisk, checkpointRange));
if isempty(steps)
    error('model_checkpoints:range', 'No %s checkpoints on disk at those steps (on disk: %s).', model, mat2str(onDisk));
end
files = arrayfun(@(k) fullfile(dirC, sprintf('checkpoint-%d', k), 'gpt2_full_weights.mat'), steps, 'UniformOutput', false);
tokPerStep = ck.(key).tokPerStep;
names = compose('%s step %d', char(model), steps);
end
