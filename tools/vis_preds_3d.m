
% add paths
addpath('skeleton2d3d/H36M_utils/H36M');
addpath('skeleton2d3d/H36M_utils/utils');
addpath('skeleton2d3d/H36M_utils/external_utils');
addpath('skeleton2d3d/H36M_utils/external_utils/lawrennd-mocap');
addpath('skeleton2d3d/H36M_utils/external_utils/xml_io_tools');

% expID = 'seq16-hg-256-res-clstm';  mode = 0;
% expID = 'seq16-hg-256-res-clstm-res-64-w1e-6';  mode = 1;

% split = 'train';
% split = 'val';
% split = 'test';

% set parameters
interval = 2;

% set vis root
vis_root = ['./outputs/vis_preds_penn/' expID '/' split '/'];

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

% init camera
CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

% set opt and init dataset
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% get video ids for visualization
sidx = dataset.getSampledIdx();
sidx = sidx(1:interval:numel(sidx));

% visualize first min(K,len) videos for each action
K = 3;
list_seq = dir('./data/Penn_Action_cropped/labels/*.mat');
list_seq = {list_seq.name}';
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/Penn_Action_cropped/labels/' list_seq{i}];
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
if mode == 0
    set(gcf,'Position',[2 26 915 330]);
end
if mode == 1
    set(gcf,'Position',[2 26 2135 330]);
end
clear hi hh hp hs1 hs2 hr1 hr2

% load libraries
libimg = ip_img();

fprintf('visualizing penn predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    [sid, fid] = dataset.getSeqFrId(i);
    % make directories
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    makedir(vis_dir);
    % load predictions
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',expID,split,i);
    preds = load(pred_file);
    hmap = preds.hmap;
    if mode == 1
        repos = zeros(opt.seqLength,17,3);
        repos(:,joints,:) = preds.repos;
        repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
        repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
        repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
        repos(:,11,:) = preds.repos(:,1,:);
        repos = permute(repos,[1,3,2]);
        trans = preds.trans;
        focal = preds.focal;
    end
    if exist(sprintf('./exp/penn-crop/%s/eval_%s/',expID,split),'dir')
        eval_file = sprintf('./exp/penn-crop/%s/eval_%s/%05d.mat',expID,split,i);
        preds = load(eval_file);
        pred2 = zeros(opt.seqLength,17,2);
        pred2(:,joints,:) = preds.eval;
        pred2(:,1,:) = (preds.eval(:,8,:) + preds.eval(:,9,:))/2;
        pred2(:,8,:) = (preds.eval(:,2,:) + preds.eval(:,3,:) + preds.eval(:,8,:) + preds.eval(:,9,:))/4;
        pred2(:,9,:) = (preds.eval(:,1,:) + preds.eval(:,2,:) + preds.eval(:,3,:))/3;
        pred2(:,11,:) = preds.eval(:,1,:);
    end
    % load input
    [input, ~, ~, ~, ~, ~] = dataset.get(i);
    input = permute(input,[1,3,4,2]);
    for j = 1:opt.seqLength
        % skip if figure exists
        vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];
        if exist(vis_file,'file')
            continue
        end
        % show image
        im_file = sprintf('data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid);
        im = imread(im_file);
        if exist('hi','var')
            delete(hi);
        end
        if mode == 0
            hi = subplot('Position',[0.00+0/3 0.00 1/3-0.00 1.00]);
        end
        if mode == 1
            hi = subplot('Position',[0.00+0/7 0.00 1/7-0.00 1.00]);
        end
        imshow(im); hold on;
        % draw heatmap
        if exist('hh','var')
            delete(hh);
        end
        if mode == 0
            hh = subplot('Position',[0.00+1/3 0.00 1/3-0.00 1.00]);
        end
        if mode == 1
            hh = subplot('Position',[0.00+1/7 0.00 1/7-0.00 1.00]);
        end
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
        if exist(sprintf('./exp/penn-crop/%s/eval_%s/',expID,split),'dir')
            if exist('hp','var')
                delete(hp);
            end
            if mode == 0
                hp = subplot('Position',[0.00+2/3 0.00 1/3-0.00 1.00]);
            end
            if mode == 1
                hp = subplot('Position',[0.00+2/7 0.00 1/7-0.00 1.00]);
            end
            imshow(im); hold on;
            show2DPose(permute(pred2(j,:,:),[3 2 1]),pos2dSkel);
            axis off;
        end
        if mode == 0
            % save figure
            set(gcf,'PaperPositionMode','auto');
            print(gcf,vis_file,'-dpng','-r0');
            continue
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
            pred = permute(repos(j,:,:),[2 3 1]);
            pred = pred + repmat(permute(trans(j,:),[2 1]),[1 size(pred,2)]);
            V = pred;
            V([2 3],:) = V([3 2],:);
            hpos = showPose(V,posSkel);
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
            pred = permute(repos(j,:,:),[2 3 1]);
            V = pred;
            V([2 3],:) = V([3 2],:);
            showPose(V,posSkel);
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
        print(gcf,vis_file,'-dpng','-r0');
    end
end
fprintf('done.\n');

close;