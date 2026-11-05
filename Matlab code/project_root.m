function r = project_root()
% project_root  The project folder (code, models, datasets): by default the folder that holds
%   'Matlab code' and 'Python code', i.e. the root of this repository. The model folders
%   ('GPT-2 small model', ...) and 'Datasets' go there (see README.md).
%   Set the environment variable LLM_PROJECT_ROOT to use another folder.
%
% See also python_command, model_weights_file, model_checkpoints.

r = getenv('LLM_PROJECT_ROOT');
if isempty(r)
    r = fileparts(fileparts(mfilename('fullpath')));    % the folder above 'Matlab code'
end
end
