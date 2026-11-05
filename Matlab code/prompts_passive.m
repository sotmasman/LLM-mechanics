function [sentences, labels, groupID] = prompts_passive(groupBy)
% prompts_passive  Who acts, as a 2 x 2 (who acts x active/passive): 12 sets of 4
% sentences, one predator, one prey animal and one verb per set, e.g.
%     The cat watched the mouse.          (predator acts, active,  ends with prey)
%     The mouse watched the cat.          (prey acts,     active,  ends with predator)
%     The mouse was watched by the cat.   (predator acts, passive, ends with predator)
%     The cat was watched by the mouse.   (prey acts,     passive, ends with prey)
% In active sentences the actor always comes first, so "who acts" = "what kind of word
% came last"; mixing in passives breaks that link: in each role class one sentence ends
% with the predator and one with the prey, and one is active, one passive (passives are
% longer, but that is balanced across the role classes). Verbs make sense both ways
% round, so the prey acting is not an odd sentence. Used by run_DIM.
%   groupBy    which label to return (default 'role'):
%              'role'      predator acts / prey acts (who did it: the real test)
%              'lastword'  ends with prey / ends with predator (lexical control)
%   sentences  {1 × 48}
%   labels     {1 × 48}
%   groupID    [1 × 48]  set number 1-12: the 4 sentences of a set share the same
%                        words, so they are left out together

if nargin < 1, groupBy = 'role'; end
pred  = {'cat', 'wolf', 'hawk', 'fox', 'lion', 'snake', 'tiger', 'eagle', 'owl', 'bear', 'shark', 'leopard'};
prey  = {'mouse', 'deer', 'rabbit', 'hen', 'zebra', 'frog', 'goat', 'lamb', 'squirrel', 'salmon', 'seal', 'monkey'};
verbs = {'watched', 'followed', 'noticed', 'heard', 'smelled', 'approached', ...
         'ignored', 'spotted', 'startled', 'sniffed', 'avoided', 'studied'};

sentences = {};  role = {};  lastword = {};  groupID = [];
for k = 1:numel(pred)
    P = pred{k};  Q = prey{k};  v = verbs{k};
    sentences(end + 1:end + 4) = {sprintf('The %s %s the %s.', P, v, Q), ...          % predator acts, active
                                  sprintf('The %s %s the %s.', Q, v, P), ...          % prey acts, active
                                  sprintf('The %s was %s by the %s.', Q, v, P), ...   % predator acts, passive
                                  sprintf('The %s was %s by the %s.', P, v, Q)};      % prey acts, passive
    role(end + 1:end + 4)      = {'predator acts', 'prey acts', 'predator acts', 'prey acts'};
    lastword(end + 1:end + 4)  = {'ends with prey', 'ends with predator', 'ends with predator', 'ends with prey'};
    groupID(end + 1:end + 4)   = k;
end
switch groupBy
    case 'role',     labels = role;
    case 'lastword', labels = lastword;
    otherwise, error('groupBy must be ''role'' or ''lastword''.');
end
end
