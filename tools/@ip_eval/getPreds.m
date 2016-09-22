function [ preds ] = getPreds( obj, hm )
    assert(ndims(hm) == 4, 'Input must be 4-D tensor')
    [~, idx] = max(reshape(hm, [size(hm,1) size(hm,2) size(hm,3)*size(hm,4)]), [], 3);
    preds = repmat(idx,[1 1 2]);
    preds(:,:,1) = floor((preds(:,:,1)-1)./size(hm,3))+1;
    preds(:,:,2) = mod((preds(:,:,2)-1),size(hm,3))+1;
end