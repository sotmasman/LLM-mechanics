function step_token_axis(ax, xl, tokPerStep)
% step_token_axis  Log x axis labelled in training steps (below) and tokens (above).
%   step_token_axis(ax, xl, tokPerStep): ax = the axes, xl = x limits in steps (step 0 is
%   drawn at 0.5), tokPerStep = training tokens per step (from model_checkpoints).
%   The token labels are text along the top edge, so both labellings share one axis;
%   leave two empty lines at the end of the panel title to make room for them.
%
% See also run_DIM, run_probe, model_checkpoints.

set(ax, 'XScale', 'log', 'XLim', xl, 'Layer', 'top');
xticks(ax, [0.5 1 10 100 1e3 1e4 1e5 1e6]);  xticklabels(ax, {'0', '1', '10', '100', '1k', '10k', '100k', '1M'});
xlabel(ax, 'Training step (step 0 at the left)');
tok = [0.5 * tokPerStep, 1e6 1e7 1e8 1e9 1e10 1e11 1e12];  lab = {'0', '1M', '10M', '100M', '1B', '10B', '100B', '1T'};
yl = ylim(ax);  yTop = yl(2);  if strcmp(get(ax, 'YDir'), 'reverse'), yTop = yl(1); end   % the top edge, also for reversed y axes
for k = 1:numel(tok)
    xs = tok(k) / tokPerStep;                            % the same point in steps
    tooClose = k > 1 && xs < 0.5 * 3;                    % would overlap the "0" label (step 0 is drawn at 0.5)
    if xs >= xl(1) && xs <= xl(2) && ~tooClose
        text(ax, xs, yTop, lab{k}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Clipping', 'off');
    end
end
text(ax, 0.5, 1.06, 'Training tokens', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
end
