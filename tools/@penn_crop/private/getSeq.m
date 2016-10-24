function [ ind ] = getSeq( obj, i )

id = obj.ind2sub(i, 1);

% get frame index
if strcmp(obj.seqType,'phase') == 1
    ii = obj.seqId == id;
    nFrame = obj.nFrame(ii, 1);
    ind = linspace(i, i+nFrame-1, obj.nPhase);
    ind = round(ind);
    assert(numel(ind) == obj.nPhase);
    ind = ind(1:obj.seqLength);
end
if strcmp(obj.seqType,'raw') == 1
    ind = i:i+obj.seqLength-1;
end

% replace overlength indices with the last index

% ind2sub
rep_ind = ind > size(obj.ind2sub,1);
rep_val = max(ind(rep_ind == 0));
ind(rep_ind) = rep_val;

% sequence id
rep_ind = obj.ind2sub(ind,1) ~= id;
rep_val = max(ind(rep_ind == 0));
ind(rep_ind) = rep_val;

end