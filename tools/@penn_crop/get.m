function [ input, seqIdx, center, scale ] = get( obj, idx )
% This does not produce the exact output as the get() in penn-crop.lua
%   1. input is of type uint8 with values in range [0 255]

seqIdx = getSeq(obj, idx);

input = uint8(zeros(numel(seqIdx), 3, obj.inputRes, obj.inputRes));

for i = 1:numel(seqIdx)
    sid = seqIdx(i);
    % Load image
    img = loadImage(obj, sid);
    % Get center and scale (same for all frames)
    if i == 1
        [center, scale] = getCenterScale(obj, img);
    end
    % Transform image
    input(i,:,:,:) = obj.img.crop(img, center, scale, 0, obj.inputRes);
end

end

