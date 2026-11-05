function [sentences, labels, groupID] = prompts_strong_weak_praise()
% prompts_strong_weak_praise  10 strong-praise / 10 weak-praise sentences in matched pairs: the same
% sentence and the same final adjective, with an intensifier (strong praise) or a hedge (weak praise) before it
% ("The food was extremely good." / "The food was fairly good."). The last word is identical in the
% two classes, so at the last word (position 'last word') a readout can only succeed if the model has
% combined the preceding word with the adjective: selectivity at layer 0 is at chance by construction.
% How positive, carried by context. (An earlier set with different final adjectives, "delicious" vs "okay",
% was at ceiling from layer 0, i.e. solved by word identity, and was removed.)
% Intensifier / hedge pairs, each used in 2-3 sentence pairs: extremely / fairly, truly / somewhat,
% incredibly / mildly, really / slightly. All words are single tokens in GPT-2 and Pythia (checked).
% Used by run_DIM, run_probe and run_PCA (contrast 'strong-weak-praise'; first named 'intensifier-hedge').
%   sentences  {1 × 20}  strong praise 1-10, then their weak-praise twins 11-20
%   labels     {1 × 20}  'strong' / 'weak'
%   groupID    [1 × 20]  sentence k and its twin k + 10 share a group: leave them out together

sentences = {'The food was extremely good.', 'The day was truly pleasant.', 'The service was incredibly helpful.', ...
             'This book is really interesting.', 'The performance was extremely impressive.', 'The view is truly beautiful.', ...
             'The movie was incredibly enjoyable.', 'The hotel was really comfortable.', 'The weather was extremely nice.', ...
             'The coffee was truly tasty.', ...                                                                         % strong praise
             'The food was fairly good.', 'The day was somewhat pleasant.', 'The service was mildly helpful.', ...
             'This book is slightly interesting.', 'The performance was fairly impressive.', 'The view is somewhat beautiful.', ...
             'The movie was mildly enjoyable.', 'The hotel was slightly comfortable.', 'The weather was fairly nice.', ...
             'The coffee was somewhat tasty.'};                                                                         % weak praise
labels  = repelem({'strong', 'weak'}, 10);
groupID = [1:10, 1:10];
end
