function weightsFile = model_weights_file(model, export)
% model_weights_file  Path of a model's MATLAB weights; exports them once if missing.
%   weightsFile = model_weights_file(model)
%     model  'gpt2-small' (124 M, 12 layers) or 'gpt2-xl' (1.5 B, 48 layers);
%            also 'gpt2-medium' / 'gpt2-large' (downloaded), and Pythia:
%            'pythia-160m' (folder Pythia 160M model)
%     export false = only return the path, never export (default true)
%   Returns  <project_root>\<model folder>\gpt2_full_weights.mat
%
% If the .mat file doesn't exist yet but the folder holds the Hugging Face files
% (model.safetensors, config.json), runs export_gpt2_weights_lowmem.py (GPT-2) or
% export_pythia_weights.py (Pythia) once (~1 min for small, ~10 min for XL). The
% weights never change, so after the first time this only returns the path.
%
% Example:
%   W = load(model_weights_file('gpt2-xl'));
%
% See also model_forward, run_model, run_patching, run_ablation.

projectDir = project_root();
if nargin < 2, export = true; end
sizes   = struct('gpt2_small', 'small', 'gpt2_medium', 'medium', 'gpt2_large', 'large', 'gpt2_xl', 'XL');
pythia  = struct('pythia_160m', 'Pythia 160M model');    % Pythia models: key -> folder
key = strrep(lower(char(model)), '-', '_');
if isfield(sizes, key)
    modelDir = fullfile(projectDir, sprintf('GPT-2 %s model', sizes.(key)));
    exporter = 'export_gpt2_weights_lowmem.py';
    hfName   = ['openai-community/' regexprep(lower(char(model)), '^gpt2-small$', 'gpt2')];
elseif isfield(pythia, key)
    modelDir = fullfile(projectDir, pythia.(key));
    exporter = 'export_pythia_weights.py';
    hfName   = ['EleutherAI/' lower(char(model))];
else
    error('Unknown model "%s". Use ''gpt2-small'', ''gpt2-medium'', ''gpt2-large'', ''gpt2-xl'' or ''pythia-160m''.', model);
end
weightsFile = fullfile(modelDir, 'gpt2_full_weights.mat');

if export && ~isfile(weightsFile)
    if ~isfile(fullfile(modelDir, 'model.safetensors'))
        error('%s has no model.safetensors. Download the model from Hugging Face (%s) first.', modelDir, hfName);
    end
    fprintf('Exporting %s weights (once) ...\n', model);
    status = system([python_command(fullfile(projectDir, 'Python code', exporter)) ' "' modelDir '"'], '-echo');
    if status ~= 0, error('%s failed (see above).', exporter); end
end
end
