function [sentences, labels, groupID] = prompts_topic()
% prompts_topic  16 sentences in 4 topics (food, sport, weather, politics), 4 each.
% All 7 tokens, ending in "."; sentence-initial "The" spread across topics so no
% topic shares an extra token. Used by run_RSA, run_DIM (and run_cosine_resid example 2).
%   sentences  {1 × 16}  the sentences
%   labels     {1 × 16}  topic of each sentence, same order
%   groupID    [1 × 16]  leave-out groups: no matched sentences, so each is its own group

sentences = {'I ate pasta for dinner tonight.', 'She baked fresh bread this morning.', ...
             'The soup needs more salt now.', 'We ordered pizza with extra cheese.', ...                   % food
             'The team won the match easily.', 'He scored two goals last night.', ...
             'The runner broke the world record.', 'Our coach made us practice daily.', ...                % sport
             'Heavy rain fell all afternoon today.', 'A cold wind blew all night.', ...
             'The forecast says snow tomorrow morning.', 'The sky cleared after the storm.', ...           % weather
             'A court ruled against the company.', 'Our mayor signed the new law.', ...
             'The judge delayed the trial again.', 'The voters chose a new president.'};                   % politics / law
labels  = repelem({'food', 'sport', 'weather', 'politics'}, 4);
groupID = 1:numel(sentences);
end
