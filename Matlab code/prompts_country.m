function [sentences, labels, groupID] = prompts_country()
% prompts_country  10 Italy / 10 France sentences in matched pairs: the same sentence with
% the other country ("I spent the summer in Italy." / "I spent the summer in France.").
% No capital city is named. Used by run_steering (concept 'country').
%   sentences  {1 × 20}  Italy 1-10, then their France twins 11-20
%   labels     {1 × 20}  'Italy' / 'France'
%   groupID    [1 × 20]  sentence k and its twin k + 10 share a group

italy = {'I spent the summer in Italy.', 'We are flying to Italy next week.', 'Italy has many beautiful old towns.', ...
         'My grandmother was born in Italy.', 'The wine from Italy is famous.', 'She studied art in Italy.', ...
         'They drove across Italy by car.', 'Italy won the football match.', 'He bought a small house in Italy.', ...
         'The trains in Italy were slow.'};
sentences = [italy, replace(italy, 'Italy', 'France')];
labels  = repelem({'Italy', 'France'}, 10);
groupID = [1:10, 1:10];
end
