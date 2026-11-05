function cmd = python_command(script)
% python_command  The system command that runs a Python script on this computer, for system().
%   cmd = python_command(script) returns e.g. 'py "C:\...\script.py"' on Windows (the py launcher, with
%   UTF-8 output) and 'python3 "/home/.../script.py"' elsewhere. Set the environment variable LLM_PYTHON
%   to use another Python (e.g. a conda environment's python). Arguments are appended by the caller.
%
% See also project_root, model_tokenize, model_weights_file.

py = getenv('LLM_PYTHON');
if ispc
    if isempty(py), py = 'py'; else, py = ['"' py '"']; end
    cmd = ['set PYTHONIOENCODING=utf-8&& ' py ' "' script '"'];
else
    if isempty(py), py = 'python3'; else, py = ['"' py '"']; end
    cmd = ['PYTHONIOENCODING=utf-8 ' py ' "' script '"'];
end
end
