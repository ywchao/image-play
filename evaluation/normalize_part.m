function [ p, w, mu, sc ] = normalize_part( part, vis, mode )
% normalize using only visible joints

if nargin < 3
    mode = 'max';
end

% substract the x, y coordinates by mean
w = sum(vis, 2);
p = part .* repmat(vis, [1 1 2]);
mu = sum(p, 2) ./ repmat(w, [1 1 2]);
p = p - repmat(mu, [1 size(part,2) 1]);
p = p .* repmat(vis, [1 1 2]);

switch mode
    case 'max'
        % divid x, y coordinates by max length
        sc = max(sum(p.^2,3), [], 2);
        sc = sqrt(sc);
        p = p ./ repmat(sc, [1 size(part,2) 2]);
    case 'var'
        % divid x, y coordinates by their variances separately
        sc = sum(p.^2, 2) ./ repmat(w, [1 1 2]);
        p = p ./ repmat(sc, [1 size(part,2) 1]);
end

end

