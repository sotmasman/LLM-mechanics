function text = model_decode(ids, src)
% model_decode  Turn a sequence of GPT-2 or Pythia token IDs into text.
%   text = model_decode(ids) joins the tokens and decodes them together.
%   Unlike joining model_token results one by one, this also gets characters
%   right that GPT-2 splits across several tokens (Greek letters, accented
%   letters, emoji). IDs are 0-based.
%
%   model_decode(ids, src) uses another model's vocabulary: src = a model name
%   (e.g. 'pythia-160m') or the weights struct W. Default: GPT-2.
%
%   Example:  model_decode([464 3139 286 4881])   % "The capital of France"
%
% See also model_token.

if nargin < 2, src = 'gpt2-small'; end
[~, raw] = model_token(ids, src);                  % tokens as stored, e.g. "ĠParis"
raw = raw(raw ~= "<unused>");                      % drop padding IDs (Pythia)
codes = double(char(strjoin(raw(:)', "")));         % one character per byte

% GPT-2 stores each byte as a printable character (space -> "Ġ", ...).
% Map them back to bytes, then decode all the bytes together as UTF-8.
printable = [33:126, 161:172, 174:255];             % bytes that keep their own character
others = setdiff(0:255, printable);                 % bytes moved to U+0100 onwards
byteOf = zeros(1, 256 + numel(others));
byteOf([printable, 256 + (0:numel(others) - 1)] + 1) = [printable, others];
text = string(native2unicode(uint8(byteOf(codes + 1)), 'UTF-8'));
end
