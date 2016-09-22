function [ center, scale ] = getCenterScale( obj, img )

assert(ndims(img) == 3);
w = size(img,2);
h = size(img,1);
x = (w+1)/2;
y = (h+1)/2;
scale = max(w,h)/200;
center = [x, y]; 

end

