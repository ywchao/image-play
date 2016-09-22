classdef ip_eval
    methods
        preds = getPreds(obj, hm);
        
        dists = calcDists(obj, preds, label, normalize);
        
        acc = distAccuracy(obj, dists, thr)
        
        % The implementation is slightly different from eval.lua---input 
        % label is the joint location instead of heatmap.
        acc = heatmapAccuracy(obj, output, label, thr, idxs, outputRes)
    end
end