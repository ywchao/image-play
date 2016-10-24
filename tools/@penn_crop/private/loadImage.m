function [ img ] = loadImage( obj, idx )

img = imread(fullfile(obj.dir, imgpath(obj, idx)));
img = permute(img, [3 1 2]);

end

