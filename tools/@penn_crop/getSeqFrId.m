function [ sid, fid ] = getSeqFrId( obj, idx )

sid = obj.ind2sub(idx, 1);
fid = obj.ind2sub(idx, 2);

end