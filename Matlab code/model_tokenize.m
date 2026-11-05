function ids = model_tokenize(text, src)
% model_tokenize  Turn text into GPT-2 or Pythia token IDs.
%   ids = model_tokenize(text) returns a 1 × n row of 0-based GPT-2 token IDs,
%   starting with <|endoftext|> (50256) like every saved prompt, ready for
%   model_forward / model_generate.
%   ids = model_tokenize(text, src) uses another model's tokenizer: src = a
%   model name (e.g. 'pythia-160m', whose <|endoftext|> is 0) or the weights
%   struct W. All GPT-2 sizes share one tokenizer, all Pythia sizes another.
%
%   ids = model_tokenize(texts) with a string or cell array of several texts
%   returns a cell array of ID rows, one per text, from a single Python call.
%
%   It calls Python code\tokenize_prompt.py, which loads only the
%   tokenizer, not the model: about 1 s.
%   The IDs are identical to the Hugging Face / TransformerLens tokenizer.
%
%   Example:
%     ids = model_tokenize('The capital of France is the city of');
%     % [50256 464 3139 286 4881 318 262 1748 286]
%     model_decode(ids(2:end))          % back to the text
%
% See also model_decode, model_token.

script = fullfile(project_root(), 'Python code', 'tokenize_prompt.py');
if nargin < 2, src = 'gpt2-small'; end
modelDir = model_tokenizer_dir(src);              % folder with the tokenizer files

% Pass the text through a UTF-8 file, so any characters (Greek, accents)
% survive the Windows command line. Several texts go as one JSON list.
many = numel(string(text)) > 1;
tmp = tempname;  if many, tmp = [tmp '.json']; else, tmp = [tmp '.txt']; end
fid = fopen(tmp, 'w', 'n', 'UTF-8');
if many, fprintf(fid, '%s', jsonencode(cellstr(text))); else, fprintf(fid, '%s', char(text)); end
fclose(fid);
[status, output] = system([python_command(script) ' "' tmp '" "' modelDir '"']);
delete(tmp);

lines = splitlines(strtrim(string(output)));
if many                                          % one line of IDs per text, at the end
    lines = lines(end - numel(string(text)) + 1:end);
    ids = arrayfun(@(l) str2double(split(l))', lines, 'UniformOutput', false)';
    bad = any(cellfun(@(v) any(isnan(v)) || ~ismember(v(1), [0 50256]), ids));
else
    ids = str2double(split(lines(end)))';        % last line = the IDs
    bad = any(isnan(ids)) || ~ismember(ids(1), [0 50256]);
end
if status ~= 0 || bad
    error('model_tokenize:failed', 'tokenize_prompt.py failed:\n%s', output);
end
end
