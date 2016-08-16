function [ img ] = loadImage( obj, idx )

img = imread(fullfile(obj.dir, imgpath(obj, idx)));

end

