function [sentences, labels, groupID, reference] = prompts_capital()
% prompts_capital  Two prompts that differ only in the country word, plus 8 more of the
% same template as reference sentences (all 8 tokens). Used by run_cosine_resid (example 1).
%   sentences  {1 × 2}   'The capital of Greece is the city of', '... Italy ...'
%   labels     {1 × 2}   the country of each prompt
%   groupID    [1 × 2]   each prompt is its own group
%   reference  {1 × 8}   same template, other countries: used only to estimate the mean
%                        for centering (not compared)

sentences = {'The capital of Greece is the city of', 'The capital of Italy is the city of'};
labels    = {'Greece', 'Italy'};
groupID   = 1:numel(sentences);
reference = {'The capital of France is the city of', 'The capital of Spain is the city of', ...
             'The capital of Germany is the city of', 'The capital of Japan is the city of', ...
             'The capital of Egypt is the city of', 'The capital of Canada is the city of', ...
             'The capital of Russia is the city of', 'The capital of India is the city of'};
end
