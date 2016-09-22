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
    % TODO: following the implement of penn-crop.lua
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if size(img, 1) > size(img, 2)
        outputSize = [obj.inputRes NaN];
    else
        outputSize = [NaN obj.inputRes];
    end
    img = imresize(img, outputSize);
    % pad zeros
    inp = uint8(zeros(obj.inputRes,obj.inputRes,3));
    if size(img,1) > size(img,2)
        ul = [1 round((obj.inputRes-size(img,2))/2)+1];
    else
        ul = [round((obj.inputRes-size(img,1))/2)+1 1];
    end
    inp(ul(1):ul(1)+size(img,1)-1, ul(2):ul(2)+size(img,2)-1, :) = img;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    input(i,:,:,:) = permute(inp, [4 3 1 2]);
end

end

