function [ sidx ] = getSampleIdx( obj )

sidx = [];
scnt = 0;
for i = 1:size(obj.ind2sub, 1)
    if obj.ind2sub(i, 2) == 1
        scnt = scnt + 1;
        % subsample videos (1/10) for training set only
        if strcmp(obj.split,'train') == 1 && mod(scnt, 10) ~= 1
            continue
        end
        phaseSeq = getPhaseSeq(obj, i);
        if isempty(sidx)
            sidx = phaseSeq;
        else
            sidx = [sidx phaseSeq];
        end
    end
end

end

