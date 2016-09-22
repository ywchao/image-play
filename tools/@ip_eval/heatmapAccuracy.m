function [ acc ] = heatmapAccuracy( obj, output, label, thr, idxs, outputRes )
% Calculate accuracy according to PCK. First value to be returned is 
% average accuracy across 'idxs', followed by individual accuracies.
%
% The implementation is slightly different from eval.lua---can take either
% heatmap or joint locations
%
% output:
%   acc     (1+numel(idxs)) x 1

% hard-code normalize
norm = outputRes/10;

if ndims(output) == 4
    preds = getPreds(obj, output);
else
    assert(ndims(output) == 3 && size(output,3) == 2);
    preds = output;
end
if ndims(label) == 4
    gt = getPreds(obj, label);
else
    assert(ndims(label) == 3 && size(label,3) == 2);
    gt = label;
end

dists = calcDists(obj, preds, gt, ones(size(preds,1))*norm);

avgAcc = 0;
badIdxCount = 0;

if isempty(idxs)
    acc = zeros(size(dists,1),1);
    for i = 1:size(dists,1)
        acc(i+1) = distAccuracy(obj, dists(i,:));
        if acc(i+1) >= 0
            avgAcc = avgAcc + acc(i+1);
        else
            badIdxCount = badIdxCount + 1;
        end
    end
    acc(1) = avgAcc / (size(dists,1) - badIdxCount);
else
    acc = zeros(numel(idxs),1);
    for i = 1:numel(idxs)
        acc(i+1) = distAccuracy(obj, dists(idxs(i),:));
        if acc(i+1) >= 0
            avgAcc = avgAcc + acc(i+1);
        else
            badIdxCount = badIdxCount + 1;
        end
    end
    acc(1) = avgAcc / (numel(idxs) - badIdxCount);
end

end

