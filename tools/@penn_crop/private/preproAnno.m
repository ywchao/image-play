function [ seqId, nFrame ] = preproAnno(obj)

seqId = unique(obj.ind2sub(:,1));
nFrame = zeros(numel(seqId),1);
for i = 1:numel(seqId)
    nFrame(i) = sum(obj.ind2sub(:,1) == seqId(i));
end

end