function [ dists ] = calcDists( obj, preds, label, normalize )
% input:
%   preds       BatchSize x OutputRes x 2
%   label       BatchSize x OutputRes x 2
%   normalize
% output:
%   dists       OutputRes x BatchSize

dists = zeros(size(preds,2), size(preds,1));
for i = 1:size(preds,1)
    for j = 1:size(preds,2)
        if label(i,j,1) > 1 && label(i,j,2) > 1
            diff = permute(label(i,j,:)-preds(i,j,:),[3 2 1]);
            dists(j,i) = norm(diff)/normalize(i);
        else
            dists(j,i) = -1;
        end
    end
end

end

