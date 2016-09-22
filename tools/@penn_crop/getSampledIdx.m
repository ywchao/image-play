function [ sidx ] = getSampledIdx( obj )

sidx = [];
scnt = 0;
% sidx should not depend on the input seqLength
seqType_ = obj.seqType;
seqLength_ = obj.seqLength;
obj.seqType = 'phase';
obj.seqLength = 16;
for i = 1:size(obj.ind2sub, 1)
    if obj.ind2sub(i, 2) == 1
        scnt = scnt + 1;
        % subsample videos (1/10) for training set only
        if strcmp(obj.split,'train') == 1 && mod(scnt, 10) ~= 1
            continue
        end
        seqIdx = getSeq(obj, i);
        if isempty(sidx)
            sidx = seqIdx;
        else
            sidx = [sidx seqIdx];
        end
    end
end
obj.seqType = seqType_;
obj.seqLength = seqLength_;

end

