
% split = 'val';
split = 'test';

% set visibility threshold for training data
% selected by maximizing validation accuracy
thres_vis = 9;

% nn all
fprintf('exp: nn-all\n');
mode = 'all';  %#ok
nn_vis_one;

% nn oracle
fprintf('exp: nn-oracle\n');
mode = 'oracle';
nn_vis_one;
