function [ acc ] = distAccuracy( obj, dists, thr )
% input:
%   dists   1 x BatchSize
%   thr
% output:
%   acc

if ~exist('thr','var')
    thr = 0.5;
end

% return percentage below threshold while ignoring values with a -1
if sum(dists ~= -1) > 0
    acc = sum(dists <= thr & dists ~= -1) / sum(dists ~= -1);
else
    acc = -1;
end

end

