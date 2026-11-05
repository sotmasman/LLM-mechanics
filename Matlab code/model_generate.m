function g = model_generate(ids, W, nNew, opts)
% model_generate  Generate text with GPT-2 small, one token at a time.
%   g = model_generate(ids, W, nNew)  adds nNew tokens to ids, each time
%   picking the most likely next token (greedy).
%   model_generate(...) with no output prints the prompt and the new text.
%
% How multi-token generation works
%   GPT-2 only ever predicts ONE next token. Longer text comes from running
%   it in a loop and feeding each choice back in ("autoregressive"
%   generation). Each pass through the loop:
%     1. Run    model_forward on the whole sequence so far (prompt + tokens
%               generated so far). Only the last position's prediction,
%               out.probs (50257 × 1), is used.
%     2. Pick   one token from out.probs: the most likely (greedy,
%               Temperature = 0), or a random draw (Temperature > 0).
%     3. Append it to ids. On the next pass the model sees its own choice
%               as part of the input, and predicts the token after it.
%     4. Stop   after nNew tokens, or earlier if <|endoftext|> was chosen.
%   Example with a 9-token prompt and nNew = 3:
%     pass 1: run on 9 tokens  -> pick token 10, append
%     pass 2: run on 10 tokens -> pick token 11, append
%     pass 3: run on 11 tokens -> pick token 12, append  -> 12 tokens
%   Every pass reruns the full sequence from scratch (no caching), so a
%   pass costs about 0.25 s and the cost grows slowly with length.
%   Weights never change between passes; only the sequence grows.
%   g.out keeps the activations of the last pass (pass 3 above: 11 tokens).
%   No extra pass is run, so with nNew = 1 there is exactly one pass.
%
% Inputs
%   ids    Starting token IDs, 0-based (e.g. a.token_ids from activations.mat)
%   W      All weights, as for model_forward
%   nNew   Number of tokens to generate
%
% Options (name=value)
%   Temperature  0 (default) = greedy decoding: always the most likely
%                token (temperature 0 is shorthand for greedy). Above 0 =
%                sample at random: lower is safer, higher more varied,
%                1 = the model's own probabilities.
%   TopK         When sampling, only choose among the K most likely tokens.
%   Seed         Fix the random draws so a sampled run can be repeated.
%   Edit         An intervention, as in model_forward. Applied at every step.
%   StopAtEnd    Stop early if <|endoftext|> is generated. Default true.
%
% Output struct
%   g.ids          1 × (n + new)   prompt + generated tokens
%   g.new_ids      1 × new         generated tokens
%   g.text         the generated text
%   g.chosen_prob  1 × new         probability of each chosen token when it was picked
%   g.top5_ids     5 × new         the 5 most likely tokens at each step,
%   g.top5_probs   5 × new           and their probabilities
%   g.out          model_forward output of the LAST pass of the loop: all
%                  activations of prompt + generated tokens except the last
%                  generated token (it was never fed back in), so
%                  n + new - 1 positions. Its last position is the prediction
%                  that chose the last generated token, and each earlier
%                  generated step is at the position before its token (GPT-2
%                  only looks backwards). To also get the last token's own
%                  activations, run model_forward(g.ids, W).
%
% Note on edits: an edit on fixed positions (e.g. position 4, or all
% positions) means the same at every step. An edit on "the last position"
% (v(:, end)) hits the newest token at each step; in g.out it was applied
% only at g.out's last position, so for earlier steps use g.chosen_prob /
% g.top5_* to see its effect.
%
% Example
%   W = load(model_weights_file('gpt2-small'));
%   a = load(fullfile(project_root(), 'Activations', 'capital_of_france', 'activations.mat'));
%   model_generate(a.token_ids, W, 20)                              % greedy
%   g = model_generate(a.token_ids, W, 20, Temperature=0.8, TopK=40, Seed=1);
%
% See also model_forward, model_decode.

arguments
    ids
    W struct
    nNew (1,1) double {mustBeInteger, mustBePositive}
    opts.Temperature (1,1) double {mustBeNonnegative} = 0
    opts.TopK double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
    opts.Seed double {mustBeScalarOrEmpty} = []
    opts.Edit = []
    opts.StopAtEnd (1,1) logical = true
end

ids = double(ids(:)');
nPrompt = numel(ids);
bos = 50256;                                        % <|endoftext|>: 50256 for GPT-2, W.cfg.bos_id (0) for Pythia
if isfield(W, 'cfg') && isfield(W.cfg, 'bos_id'), bos = double(W.cfg.bos_id); end
if nPrompt + nNew > size(W.W_pos, 2)
    error('model_generate:tooLong', 'Prompt + new tokens must be at most %d.', size(W.W_pos, 2));
end
if isempty(opts.Seed)
    rs = RandStream.getGlobalStream;
else
    rs = RandStream('twister', 'Seed', opts.Seed);   % own stream: repeatable, leaves rng alone
end

% Records of each step (one column per generated token)
chosen_prob = zeros(1, 0);
top5_ids = zeros(5, 0);
top5_probs = zeros(5, 0);

%% The generation loop: one pass = one new token
for step = 1:nNew

    %% 1. Run the model on everything so far
    % ids grows by one each pass: prompt, then prompt + 1 new token, ...
    out = model_forward(ids, W, opts.Edit);          % the edit (if any) applies on every pass
    p = double(out.probs);                          % [50257 × 1] p(next token), from the last position
    [tp, ti] = maxk(p, 5);                          % top 5, kept for the record

    %% 2. Pick the next token
    if opts.Temperature == 0
        % Greedy decoding: take the most likely token. Deterministic.
        next = ti(1) - 1;                           % -1: MATLAB index -> 0-based token ID
    else
        % Sampling. Dividing the logits by the temperature reshapes the
        % distribution: T < 1 sharpens it (safer), T > 1 flattens it (more varied).
        z = double(out.logits(:, end)) / opts.Temperature;
        q = exp(z - max(z));                        % softmax numerator (shifted for stability)
        if ~isempty(opts.TopK)
            [~, order] = sort(q, 'descend');
            q(order(opts.TopK + 1:end)) = 0;        % top-k: rule out all but the K most likely
        end
        q = q / sum(q);                             % sampling probabilities
        next = find(cumsum(q) >= rand(rs), 1) - 1;  % draw one token: the first whose cumulative p passes a random number
    end

    %% 3. Append it, so the next pass sees it as input
    ids(end + 1) = next;                            %#ok<AGROW>
    chosen_prob(end + 1) = p(next + 1);             %#ok<AGROW>  model's probability for the chosen token (before temperature)
    top5_ids(:, end + 1) = ti - 1;                  %#ok<AGROW>  the alternatives at this step
    top5_probs(:, end + 1) = tp;                    %#ok<AGROW>

    %% 4. Stop early if the model ended the text
    if opts.StopAtEnd && next == bos                % <|endoftext|>
        break
    end
end

%% Collect the results
g.ids = ids;
g.new_ids = ids(nPrompt + 1:end);
g.text = model_decode(g.new_ids, W);
g.chosen_prob = chosen_prob;
g.top5_ids = top5_ids;
g.top5_probs = top5_probs;
% Activations of the last pass the loop ran (no extra run). It covers
% prompt + all generated tokens except the last one, which was never fed
% back in. Its last position is the prediction that chose the last token;
% earlier generated steps are earlier positions (GPT-2 only looks backwards).
g.out = out;

if nargout == 0
    promptIds = ids(1:nPrompt);
    promptIds = promptIds(promptIds ~= bos | (1:nPrompt) > 1);   % hide the leading <|endoftext|>
    fprintf('%s[%s]\n', model_decode(promptIds, W), g.text);
    clear g
end
end
