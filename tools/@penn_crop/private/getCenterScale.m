function [ center, scale ] = getCenterScale( obj, img )

assert(ndims(img) == 3);
w = size(img,3);
h = size(img,2);
x = (w+1)/2;
y = (h+1)/2;
scale = max(w,h)/200;
center = [x, y];
% Small adjustment so cropping is less likely to take feet out
y = y + scale * 15;
scale = scale * 1.25;
center = [x, y]; 

end

