function [text, raw] = model_token(id, src)
%MODEL_TOKEN Look up the text of a GPT-2 or Pythia token from its token ID.
%   TEXT = MODEL_TOKEN(ID) returns the readable text of token ID, using the
%   0-based IDs that GPT-2 and TransformerLens use. For example,
%   model_token(6342) returns " Paris" (with its leading space).
%   ID can be an array; TEXT is then a string array of the same size.
%
%   [TEXT, RAW] = MODEL_TOKEN(ID) also returns the token exactly as stored in
%   vocab.json, where a leading space appears as "Ġ" (e.g. "ĠParis").
%
%   MODEL_TOKEN(ID, SRC) uses another model's vocabulary: SRC = a model name
%   (e.g. 'pythia-160m'), the weights struct W, or the path of a vocab.json.
%   Default: GPT-2 (all sizes share it). IDs beyond the vocabulary (Pythia's
%   padding rows 50277-50303) give "<unused>".
%
%   Notes:
%   - Row r of W_U (and element r of b_U or logits) is token ID r-1.
%   - A few tokens are only part of a multi-byte character (e.g. half of an
%     emoji). Their TEXT contains the replacement character U+FFFD.
%   - vocab.json is read once and cached; later calls are fast.
%
%   Example:
%       model_token(45997)            % " extravagant"
%       model_token([6342 3576])      % [" Paris"  " London"]
%       [~, idx] = max(logits);  model_token(idx - 1)

arguments
    id {mustBeNumeric, mustBeInteger, mustBeNonnegative}
    src = 'gpt2-small'
end
if (ischar(src) || isstring(src)) && endsWith(src, '.json')
    vocabPath = char(src);
else
    vocabPath = fullfile(model_tokenizer_dir(src), 'vocab.json');
end

persistent cachePath cacheRaw
if isempty(cachePath) || ~strcmp(cachePath, vocabPath)
    cacheRaw = loadVocab(vocabPath);
    cachePath = vocabPath;
end

n = numel(cacheRaw);
raw = repmat("<unused>", size(id));
ok = id < n;                                     % IDs beyond the vocabulary (padding rows)
raw(ok) = cacheRaw(id(ok) + 1);
text = raw;
for k = find(ok(:))'
    text(k) = decodeToken(raw(k));
end
end

function raw = loadVocab(vocabPath)
% Returns a string array where raw(id+1) is the token stored for that ID.
% jsondecode can't read the whole file (it turns keys into struct field
% names and changes them), so pull out the "key": id pairs with a regexp and
% use jsondecode only to unescape the key strings.
src = fileread(vocabPath);
pairs = regexp(src, '"((?:[^"\\]|\\.)*)"\s*:\s*(\d+)', 'tokens');
keys = cellfun(@(p) p{1}, pairs, 'UniformOutput', false);
ids = str2double(cellfun(@(p) p{2}, pairs, 'UniformOutput', false));
quoted = cellfun(@(k) ['"' k '"'], keys, 'UniformOutput', false);
decoded = jsondecode(['[' strjoin(quoted, ',') ']']);
raw = strings(max(ids) + 1, 1);
raw(ids + 1) = string(decoded);
end

function txt = decodeToken(rawToken)
% GPT-2 stores each byte as a printable character (space -> "Ġ", newline ->
% "Ċ", ...). Map the characters back to bytes, then decode the bytes as UTF-8.
persistent byteOf
if isempty(byteOf)
    printable = [33:126, 161:172, 174:255];   % bytes that keep their own character
    others = setdiff(0:255, printable);       % bytes moved to U+0100 onwards
    bytes = [printable, others];
    chars = [printable, 256 + (0:numel(others) - 1)];
    byteOf = zeros(1, max(chars) + 1);
    byteOf(chars + 1) = bytes;
end
codes = double(char(rawToken));
txt = string(native2unicode(uint8(byteOf(codes + 1)), 'UTF-8'));
end
