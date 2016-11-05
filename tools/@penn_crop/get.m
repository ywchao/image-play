function [ input, hmap, proj, gtpts, center, scale ] = get( obj, idx )
% This does not produce the exact output as the get() in penn-crop.lua
%   1. input is of type uint8 with values in range [0 255]

seqIdx = getSeq(obj, idx);

input = uint8(zeros(numel(seqIdx), 3, obj.inputRes, obj.inputRes));
hmap = zeros(numel(seqIdx), size(obj.part,2), obj.outputRes, obj.outputRes);
proj = zeros(numel(seqIdx), size(obj.part,2), 2);
gtpts = zeros(numel(seqIdx), size(obj.part,2), 2);

for i = 1:numel(seqIdx)
    sid = seqIdx(i);
    % Load image
    img = loadImage(obj, sid);
    % Get center and scale (same for all frames)
    if i == 1
        [center, scale] = getCenterScale(obj, img);
    end
    % Transform image
    inp = obj.img.crop(img, center, scale, 0, obj.inputRes);
    % Get projection
    pts = permute(obj.part(sid,:,:),[2 3 1]);
    pj = zeros(size(pts));
    for j = 1:size(pts,1)
        if pts(j,1) ~= 0 && pts(j,2) ~= 0
            pj(j,:) = obj.img.transform(pts(j,:), center, scale, 0, obj.outputRes, false, false);
        end
    end
    % Generate heatmap
    hm = zeros(obj.outputRes,obj.outputRes,size(pts,1));
    for j = 1:size(pts,1)
        if pts(j,1) ~= 0 && pts(j,2) ~= 0
            hm(:,:,j) = obj.img.drawGaussian(hm(:,:,j),round(pj(j,:)),2);
        end
    end
    hm = permute(hm,[3 1 2]);
    input(i,:,:,:) = inp;
    hmap(i,:,:,:) = hm;
    proj(i,:,:,:) = pj;
    gtpts(i,:,:,:) = pts;
end

end

