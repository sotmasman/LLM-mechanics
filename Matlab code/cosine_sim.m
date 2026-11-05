function C = cosine_sim(A, B)
% cosine_sim  Cosine similarity between every column of A and every column of B.
%   C = cosine_sim(A, B)   A [d × m], B [d × n]  ->  C [m × n],  C(i,j) = cos(angle between A(:,i) and B(:,j))
%   C = cosine_sim(A)      same as cosine_sim(A, A): all pairs within A [m × m]
%
%   cos θ = (u · v) / (‖u‖ ‖v‖), from -1 (opposite) through 0 (orthogonal) to 1 (same direction).
%   Only the directions matter; the lengths of the vectors are divided out.
%
% Example (embeddings of two tokens):
%   W = load(model_weights_file('gpt2-small'));
%   cosine_sim(W.W_E(:, 3797 + 1), W.W_E(:, 3290 + 1))     % " cat" vs " dog" (IDs are 0-based, +1 to index)
%
% See also run_cosine_tokenembed, run_cosine_resid.

if nargin < 2
    B = A;
end

% Step 1: work in double precision (activations are stored as single)
A = double(A);
B = double(B);

% Step 2: length (Euclidean norm) of every column: ‖u‖ = sqrt(sum of squares)
normA = vecnorm(A);                    % [1 × m]
normB = vecnorm(B);                    % [1 × n]

% Step 3: divide each column by its length -> unit vectors (length 1)
A = A ./ normA;                        % [d × m]
B = B ./ normB;                        % [d × n]

% Step 4: dot product of every unit column of A with every unit column of B.
%         For unit vectors the dot product is the cosine.
C = A' * B;                            % [m × n]
end
