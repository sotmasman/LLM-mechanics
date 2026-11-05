function [sentences, labels, groupID] = prompts_sentiment()
% prompts_sentiment  10 positive / 10 negative sentences in matched pairs: the same
% sentence with opposite sentiment ("I love this movie." / "I hate this movie.").
% Used by run_DIM and run_steering.
%   sentences  {1 × 20}  positive 1-10, then their negative twins 11-20
%   labels     {1 × 20}  'positive' / 'negative'
%   groupID    [1 × 20]  sentence k and its twin k + 10 share a group: leave them out
%                        together (a twin left in the other class is nearly the same vector)

sentences = {'I love this movie.', 'This is a wonderful day.', 'The food was delicious.', ...
             'She is a kind and generous person.', 'What a fantastic performance!', 'I am so happy with the result.', ...
             'The service was excellent.', 'This book is brilliant.', 'We had a great time together.', 'The view is beautiful.', ...   % positive
             'I hate this movie.', 'This is a terrible day.', 'The food was disgusting.', ...
             'She is a cruel and selfish person.', 'What an awful performance!', 'I am so sad with the result.', ...
             'The service was horrible.', 'This book is boring.', 'We had a miserable time together.', 'The view is ugly.'};            % negative
labels  = repelem({'positive', 'negative'}, 10);
groupID = [1:10, 1:10];
end
