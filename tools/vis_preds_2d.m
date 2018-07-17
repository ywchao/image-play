
exp_name = 'hg-256-res-clstm';

% split = 'train';
% split = 'val';
split = 'test';

% set vis root
png_root = ['./outputs/vis_' exp_name '/' split '_png/'];
gif_root = ['./outputs/vis_' exp_name '/' split '_gif/'];

% set parameters
pa = [8,1,2,3,1,5,6,9,10,11,0,9,12,13,9,15,16];
co = {'b','b','g','g','b','r','r','b','b','b','b','b','r','r','b','g','g'};
line_wd = 3;
thres = 0.10;

% set opt and init dataset
opt.data = './data/penn-crop';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% get video ids for visualization
interval = 2;
sidx = dataset.getSampledIdx();
sidx = sidx(1:interval:numel(sidx));

% limit to first min(K,len) videos for each action
K = 3;
list_seq = dir('./data/penn-crop/labels/*.mat');
list_seq = {list_seq.name}';
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/penn-crop/labels/' list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
sid = dataset.getSeqFrId(sidx);
seq = unique(sid);
keep = false(numel(seq),1);
for i = 1:numel(list_act)
    ii = find(ismember(seq,find(ia == i)));
    keep(ii(1:min(numel(ii),K))) = true;
end
seq = seq(keep);
run = sidx(ismember(sid,seq));

% init figure
figure(1);
set(gcf,'Position',[2 26 915 330]);
clear hi hh hp

% load libraries
libimg = ip_img();

fprintf('visualizing 2d predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    [sid, fid] = dataset.getSeqFrId(i);
    % make directories
    png_dir = [png_root num2str(sid,'%04d') '/'];
    gif_dir = [gif_root num2str(sid,'%04d') '/'];
    makedir(png_dir);
    makedir(gif_dir);
    
    % read image
    im_file = sprintf('./data/penn-crop/frames/%04d/%06d.jpg',sid,fid);
    im = imread(im_file);
    
    % load predictions
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',exp_name,split,i);
    preds = load(pred_file);
    hmap = preds.hmap;
    eval_file = sprintf('./exp/penn-crop/%s/eval_%s/%05d.mat',exp_name,split,i);
    evals = load(eval_file);
    pred = zeros(opt.seqLength,17,2);
    pred(:,joints,:) = evals.eval;
    pred(:,1,:) = (evals.eval(:,8,:) + evals.eval(:,9,:))/2;
    pred(:,8,:) = (evals.eval(:,2,:) + evals.eval(:,3,:) + evals.eval(:,8,:) + evals.eval(:,9,:))/4;
    pred(:,9,:) = (evals.eval(:,1,:) + evals.eval(:,2,:) + evals.eval(:,3,:))/3;
    pred(:,11,:) = evals.eval(:,1,:);
    conf = zeros(opt.seqLength,17);
    conf(:,joints) = evals.conf;
    conf(:,1) = min([evals.conf(:,8),evals.conf(:,9)],[],2);
    conf(:,8) = min([evals.conf(:,2),evals.conf(:,3),evals.conf(:,8),evals.conf(:,9)],[],2);
    conf(:,9) = min([evals.conf(:,1),evals.conf(:,2),evals.conf(:,3)],[],2);
    conf(:,11) = evals.conf(:,1);
    
    % load input
    [input, ~, ~, ~, ~, ~] = dataset.get(i);
    input = permute(input,[1,3,4,2]);
    
    for j = 1:opt.seqLength
        % skip if png exists
        png_file = [png_dir sprintf('%03d-%02d.png',fid,j)];
        if exist(png_file,'file')
            continue
        end
        
        % show image
        if exist('hi','var')
            delete(hi);
        end
        hi = subplot('Position',[0.00+0/3 0.00 1/3-0.00 1.00]);
        imshow(im); hold on;
        
        % draw heatmap
        if exist('hh','var')
            delete(hh);
        end
        hh = subplot('Position',[0.00+1/3 0.00 1/3-0.00 1.00]);
        hm = squeeze(hmap(j,:,:,:));
        ip = squeeze(input(1,:,:,:));
        inp64 = imresize(double(ip),[opt.outputRes opt.outputRes]) * 0.3;
        colorHms = cell(size(hm,1),1);
        for k = 1:size(hm,1)
            colorHms{k} = libimg.colorHM(squeeze(hm(k,:,:)));
            colorHms{k} = colorHms{k} * 255 * 0.7 + permute(inp64,[3 1 2]);
        end
        totalHm = libimg.compileImages(colorHms, 4, 4, opt.outputRes);
        totalHm = permute(totalHm,[2 3 1]);
        totalHm = uint8(totalHm);
        imshow(totalHm);
        
        % show projected 2D skeleton
        if exist('hp','var')
            delete(hp);
        end
        hp = subplot('Position',[0.00+2/3 0.00 1/3-0.00 1.00]);
        imshow(im); hold on;
        hd = nan(numel(pa),1);
        for ch = 1:numel(pa)
            if pa(ch) == 0
                continue
            end
            x1 = pred(j,pa(ch),1);
            y1 = pred(j,pa(ch),2);
            x2 = pred(j,ch,1);
            y2 = pred(j,ch,2);
            if conf(j,pa(ch)) > thres && conf(j,ch) > thres
                hd(ch) = line([x1 x2], [y1 y2], ...
                    'color', co{ch}, ...
                    'linewidth',line_wd);
            end
        end
        
        % save figure
        set(gcf,'PaperPositionMode','auto');
        print(gcf,png_file,'-dpng','-r0');
    end
    
    % generate gif
    gif_file = fullfile(gif_dir,sprintf('%03d.gif',fid));
    if exist(gif_file,'file')
        continue
    end
    for j = 1:opt.seqLength
        png_file = fullfile(png_dir,sprintf('%03d-%02d.png',fid,j));
        im = imread(png_file);
        [imind,cm] = rgb2ind(im,256);
        if j == 1
            imwrite(imind,cm,gif_file,'gif','Loopcount',inf,'DelayTime',0.25);
        else
            imwrite(imind,cm,gif_file,'gif','WriteMode','append','DelayTime',0.25);
        end
    end
end
fprintf('done.\n');

close;