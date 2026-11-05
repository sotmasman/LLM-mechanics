function p = token_positions(ids, W, position)
% token_positions  Array indices (1-based) of the token(s) a script reads.
%   p = token_positions(ids, W, position)
%     ids       the token IDs (start token first), as from model_tokenize
%     W         the weights (to look up the token text, GPT-2 or Pythia)
%     position  'last'      the last token (in most sentence sets the final ".")
%               'last word' the last token that is not punctuation (the word before
%                           a final "."): the same as deleting the "." and reading the
%                           last token, since no token sees later ones
%               'mean'      all tokens except the start token (the script averages them)
%               a number    that 0-based position (0 = the start token)
% Pythia's final "." is an attention sink (massive activations, almost no sentence
% information), so for Pythia use 'last word' when sentences end in "." (Sun et al. 2024).
%
% See also model_tokenize, model_token.

if isnumeric(position)
    p = position + 1;
    return
end
switch position
    case 'last'
        p = numel(ids);
    case 'last word'
        isWord = ~cellfun(@isempty, regexp(cellstr(model_token(ids, W)), '[A-Za-z0-9]'));
        isWord(1) = false;                              % never the start token
        p = find(isWord, 1, 'last');
    case 'mean'
        p = 2:numel(ids);
    otherwise
        error('token_positions:unknown', 'position must be ''last'', ''last word'', ''mean'' or a number.');
end
end
