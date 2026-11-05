function [sentences, labels, groupID] = prompts_negation(groupBy)
% prompts_negation  Negation as a 2 x 2 design: 10 sets of 4 sentences, one frame and
% one adjective pair per set, e.g.
%     The movie was good.        (positive adjective, not negated  -> positive)
%     The movie was not bad.     (negative adjective, negated      -> positive)
%     The movie was bad.         (negative adjective, not negated  -> negative)
%     The movie was not good.    (positive adjective, negated      -> negative)
% Every adjective and the word "not" appear equally often in both sentiment classes,
% so no single word predicts the sentiment: it has to come from combining "not" with
% the adjective. Used by run_DIM (contrast 'negation-sentiment' / 'negation-adjective').
%   groupBy    which label to return (default 'sentiment'):
%              'sentiment'  positive / negative meaning (compositional: the hard one)
%              'adjective'  positive / negative adjective (lexical control)
%              (A 'negated' option, with / without "not", was removed: those sentences
%              differ in length, so position alone separates them; not a fair comparison.)
%   sentences  {1 × 40}
%   labels     {1 × 40}  the chosen label
%   groupID    [1 × 40]  set number 1-10: the 4 sentences of a set share frame and
%                        adjectives, so they are left out together

if nargin < 1, groupBy = 'sentiment'; end
frames = {'The movie was', 'The meal was', 'The weather was', 'The host was', 'The trip was', ...
          'The room was', 'The answer was', 'The view was', 'The neighbourhood was', 'The child was'};
posAdj = {'good', 'great', 'nice', 'friendly', 'fun', 'clean', 'helpful', 'beautiful', 'safe', 'happy'};
negAdj = {'bad', 'terrible', 'awful', 'rude', 'boring', 'dirty', 'useless', 'ugly', 'dangerous', 'sad'};

sentences = {};  adjective = {};  sentiment = {};  groupID = [];
for k = 1:numel(frames)
    f = frames{k};
    sentences(end + 1:end + 4) = {[f ' ' posAdj{k} '.'], [f ' not ' negAdj{k} '.'], ...
                                  [f ' ' negAdj{k} '.'], [f ' not ' posAdj{k} '.']};
    adjective(end + 1:end + 4) = {'positive adjective', 'negative adjective', 'negative adjective', 'positive adjective'};
    sentiment(end + 1:end + 4) = {'positive', 'positive', 'negative', 'negative'};
    groupID(end + 1:end + 4)   = k;
end
switch groupBy
    case 'sentiment', labels = sentiment;
    case 'adjective', labels = adjective;
    otherwise, error('groupBy must be ''sentiment'' or ''adjective''.');
end
end
