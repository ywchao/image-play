
config;

exp_name = 'hg-256';

split = 'val';

% set directories
curr_dir = sprintf('exp/penn-crop/pose-est-%s/eval_%s/',exp_name,split);

% set body joint config
param.pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
param.p_no = numel(param.pa);

% set vis params
param.msize = 4;
param.partcolor = {'b','b','b','g','r','g','r','b','b','g','r','g','r'};

i = 1;
sid = 147;
fid = 1;

% load current frame estimation
curr_file = sprintf('%s%05d.mat',curr_dir,i);
curr = load(curr_file);
curr = squeeze(curr.eval);

% clear figure
clf; clear h;

% % load input im
% im_file = [frdata_root sprintf('%04d/%06d.jpg',sid,fid)];
% im = imread(im_file);
% plot predicted pose
% subplot('Position',[0 0 1 1]); imshow(im); hold on;
subplot('Position',[0 0 1 1]); hold on;
for child = 2:param.p_no
    x1 = curr(param.pa(child),1);
    y1 = curr(param.pa(child),2);
    x2 = curr(child,1);
    y2 = curr(child,2);
    
    plot(x2, y2, 'o', ...
        'color', param.partcolor{child}, ...
        'MarkerSize', param.msize, ...
        'MarkerFaceColor', param.partcolor{child});
    plot(x1, y1, 'o', ...
        'color', param.partcolor{child}, ...
        'MarkerSize', param.msize, ...
        'MarkerFaceColor', param.partcolor{child});
    line([x1 x2], [y1 y2], ...
        'color', param.partcolor{child}, ...
        'linewidth',round(param.msize/2));
end
axis equal;
axis ij;
axis off;

save_dir = 'outputs/figures/';
save_file = [save_dir 'nn-skel.pdf'];
makedir(save_dir);
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf','-r0');
end

close;