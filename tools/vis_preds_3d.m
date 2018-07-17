
% add paths
addpath('skeleton2d3d/h36m_utils/Release-v1.1/external_utils/xml_io_tools');
addpath('skeleton2d3d/h36m_utils/Release-v1.1/H36M');
addpath('skeleton2d3d/h36m_utils/Release-v1.1/utils');
addpath('skeleton2d3d/h36m_utils/lawrennd-mocap');

exp_name = 'hg-256-res-clstm-res-64';

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

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

% init camera
CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

% set opt and init dataset
opt.data = './data/penn-crop';
opt.nPhase = 16;
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
set(gcf,'Position',[2 26 2135 330]);
clear hi hh hp hs1 hs2 hr1 hr2

% load libraries
libimg = ip_img();

fprintf('visualizing 3d predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    [sid, fid] = dataset.getSeqFrId(i);
    % make directories
    png_dir = [png_root num2str(sid,'%04d') '/'];
    gif_dir = [gif_root num2str(sid,'%04d') '/'];
    makedir(png_dir);
    makedir(gif_dir);
    
    % read image
    im_file = sprintf('data/penn-crop/frames/%04d/%06d.jpg',sid,fid);
    im = imread(im_file);
    
    % load predictions
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',exp_name,split,i);
    preds = load(pred_file);
    hmap = preds.hmap;
    repos = zeros(opt.seqLength,17,3);
    repos(:,joints,:) = preds.repos;
    repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
    repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
    repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
    repos(:,11,:) = preds.repos(:,1,:);
    repos = permute(repos,[1,3,2]);
    trans = preds.trans;
    focal = preds.focal;
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
        hi = subplot('Position',[0.00+0/7 0.00 1/7-0.00 1.00]);
        imshow(im); hold on;
        
        % draw heatmap
        if exist('hh','var')
            delete(hh);
        end
        hh = subplot('Position',[0.00+1/7 0.00 1/7-0.00 1.00]);
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
        hp = subplot('Position',[0.00+2/7 0.00 1/7-0.00 1.00]);
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
        
        % show 3D skeleton in camera coordinates
        for k = 1:2
            if k == 1
                if exist('hs1','var')
                    delete(hs1);
                end
                hs1 = subplot('Position',[0.03+3/7 0.07 1/7-0.04 0.93]);
            end
            if k == 2
                if exist('hs2','var')
                    delete(hs2);
                end
                hs2 = subplot('Position',[0.02+4/7 0.07 1/7-0.03 0.93]);
            end
            set(gca,'fontsize',6);
            pred3d = permute(repos(j,:,:),[2 3 1]);
            pred3d = pred3d + repmat(permute(trans(j,:),[2 1]),[1 size(pred3d,2)]);
            pred3d([2 3],:) = pred3d([3 2],:);
            hpos = showPose(pred3d,posSkel);
            for l = 1:numel(hpos)-1
                set(hpos(l+1),'linewidth',2);
            end
            minx = -1500; maxx = 1500;
            miny =     0; maxy = 6500;
            minz = -1500; maxz = 1500;
            axis([minx maxx miny maxy minz maxz]);
            set(gca,'ZTick',-2000:400:2000);
            set(gca,'ZDir','reverse');
            if k == 1
                view([6,10]);
            end
            if k == 2
                view([85,10]);
            end
            CVWorld = CameraVertex;
            CVWorld(:,[2 3]) = CVWorld(:,[3 2]);
            hc = zeros(size(CameraVertex,1),1);
            for ind = 1:length(IndSetCamera)
                hc(ind) = patch( ...
                    CVWorld(IndSetCamera{ind},1), ...
                    CVWorld(IndSetCamera{ind},2), ...
                    CVWorld(IndSetCamera{ind},3), ...
                    [0.5 0.5 0.5]);
            end
            if k == 2
                ht = title({ ...
                    sprintf('focal:  %3.0f',focal(j)), ...
                    sprintf('trans:  %4.0f  %4.0f  %4.0f',trans(j,:)) ...
                    });
                set(ht,'fontsize',10);
            end
        end
        
        % show 3D skeleton relative to center
        for k = 1:2
            if k == 1
                if exist('hr1','var')
                    delete(hr1);
                end
                hr1 = subplot('Position',[0.02+5/7 0.07 1/7-0.035 0.93]);
            end
            if k == 2
                if exist('hr2','var')
                    delete(hr2);
                end
                hr2 = subplot('Position',[0.02+6/7 0.07 1/7-0.035 0.93]);
            end
            set(gca,'fontsize',6);
            pred3d = permute(repos(j,:,:),[2 3 1]);
            pred3d([2 3],:) = pred3d([3 2],:);
            showPose(pred3d,posSkel);
            minx = -1000; maxx = 1000;
            miny = -1000; maxy = 1000;
            minz = -1000; maxz = 1000;
            axis([minx maxx miny maxy minz maxz]);
            set(gca,'ZTick',-1000:200:1000);
            set(gca,'ZDir','reverse');
            if k == 1
                view([6,10]);
            end
            if k == 2
                view([85,10]);
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