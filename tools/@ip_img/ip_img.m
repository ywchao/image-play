classdef ip_img
    methods
        T = getTransform(obj, center, scale, rot, res);
        
        pt_new = transform(obj, pt, center, scale, rot, res, invert);
    end
end