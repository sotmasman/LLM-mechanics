function C = rome_covariance(W, layers, corpus_file, nTokens, chunk)
% rome_covariance  Second moment of the MLP keys, C = mean(k k'), for ROME (run_rome step 3).
%   C = rome_covariance(W, layers, corpus_file, nTokens, chunk)
%     W            weights (as for model_forward)
%     layers       0-based layers, e.g. 17 or 0:4:44 (one forward pass serves them all)
%     corpus_file  GPT-2 token IDs as uint16 (nanoGPT format), e.g. fullfile(project_root(), 'Datasets', 'wt2', 'train.bin')
%     nTokens      how many corpus tokens to use (e.g. 100000)
%     chunk        tokens per forward pass (default 128); <|endoftext|> is put first and not counted
%   C  [d_mlp × d_mlp × numel(layers)] double. A key k is the MLP's hidden activation (after GELU) at one
%   position, the input of W_out; C says which key directions ordinary text uses, so the edit can avoid them.
%
% Needs: model_forward.

if nargin < 5 || isempty(chunk), chunk = 128; end
fid = fopen(corpus_file, 'r');  ids = fread(fid, nTokens, 'uint16=>double')';  fclose(fid);   % [1 × nTokens]
d_mlp = size(W.W_in, 1);
C = zeros(d_mlp, d_mlp, numel(layers));  n = 0;  t0 = tic;
for s = 1:chunk:numel(ids) - chunk + 1
    o = model_forward([50256, ids(s:s + chunk - 1)], W);
    for j = 1:numel(layers)
        K = double(o.mlp_hidden(:, 2:end, layers(j) + 1));   % [d_mlp × chunk]  the keys of this chunk
        C(:, :, j) = C(:, :, j) + K * K';
    end
    n = n + chunk;
    if mod(n, 20 * chunk) == 0, fprintf('  covariance: %d of %d tokens (%.0f s)\n', n, nTokens, toc(t0)); end
end
C = C / n;
end
