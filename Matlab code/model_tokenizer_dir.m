function d = model_tokenizer_dir(src)
% model_tokenizer_dir  Folder with a model's tokenizer files (vocab.json, ...).
%   d = model_tokenizer_dir(src)
%     src  a model name ('gpt2-small' ... 'gpt2-xl', 'pythia-160m'), or the
%          weights struct W (GPT-2 or Pythia, from W.cfg.arch). Default 'gpt2-small'.
% All GPT-2 sizes share one tokenizer and all Pythia sizes another, so the
% weights alone are enough (also for training checkpoints).
%
% See also model_tokenize, model_token, model_decode.

if nargin < 1 || isempty(src), src = 'gpt2-small'; end
if isstruct(src)
    if isfield(src, 'cfg') && isfield(src.cfg, 'arch') && strcmp(src.cfg.arch, 'neox')
        src = 'pythia-160m';
    else
        src = 'gpt2-small';
    end
elseif startsWith(lower(char(src)), 'pythia')           % any Pythia name, incl. seeds (pythia-160m-seed1)
    src = 'pythia-160m';
elseif startsWith(lower(char(src)), 'gpt2')             % any GPT-2 name, incl. sizes and seeds
    src = 'gpt2-small';
end
d = fileparts(model_weights_file(src, false));
end
