
% add paths
addpath('skeleton2d3d/H36M_utils/H36M');
addpath('skeleton2d3d/H36M_utils/utils');
addpath('skeleton2d3d/H36M_utils/external_utils');
addpath('skeleton2d3d/H36M_utils/external_utils/lawrennd-mocap');
addpath('skeleton2d3d/H36M_utils/external_utils/xml_io_tools');

expID = 'seq16-hg-256-res-clstm-res-64-w1e-6';  mode = 1;

split = 'test';

% set save root
save_root = './outputs/figures/pull_fig_2/';
makedir(save_root);

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

% left leg/right arm
pos2dSkel_lr = pos2dSkel;
% right leg/right arm
pos2dSkel_rr = pos2dSkel;
tmp = pos2dSkel_rr.tree(2:4);
pos2dSkel_rr.tree(2:4) = pos2dSkel_rr.tree(5:7);
pos2dSkel_rr.tree(5:7) = tmp;
pos2dSkel_rr.tree(3).parent = 2;
pos2dSkel_rr.tree(4).parent = 3;
pos2dSkel_rr.tree(6).parent = 5;
pos2dSkel_rr.tree(7).parent = 6;
pos2dSkel_rr.tree(2).children = 3;
pos2dSkel_rr.tree(3).children = 4;
pos2dSkel_rr.tree(5).children = 6;
pos2dSkel_rr.tree(6).children = 7;
% left leg/left arm
pos2dSkel_ll = pos2dSkel;
tmp = pos2dSkel_ll.tree(12:14);
pos2dSkel_ll.tree(12:14) = pos2dSkel_ll.tree(15:17);
pos2dSkel_ll.tree(15:17) = tmp;
pos2dSkel_ll.tree(13).parent = 12;
pos2dSkel_ll.tree(14).parent = 13;
pos2dSkel_ll.tree(16).parent = 15;
pos2dSkel_ll.tree(17).parent = 16;
pos2dSkel_ll.tree(12).children = 13;
pos2dSkel_ll.tree(13).children = 14;
pos2dSkel_ll.tree(15).children = 16;
pos2dSkel_ll.tree(16).children = 17;
% left leg/right arm
pos2dSkel_rl = pos2dSkel_rr;
tmp = pos2dSkel_rl.tree(12:14);
pos2dSkel_rl.tree(12:14) = pos2dSkel_rl.tree(15:17);
pos2dSkel_rl.tree(15:17) = tmp;
pos2dSkel_rl.tree(13).parent = 12;
pos2dSkel_rl.tree(14).parent = 13;
pos2dSkel_rl.tree(16).parent = 15;
pos2dSkel_rl.tree(17).parent = 16;
pos2dSkel_rl.tree(12).children = 13;
pos2dSkel_rl.tree(13).children = 14;
pos2dSkel_rl.tree(15).children = 16;
pos2dSkel_rl.tree(16).children = 17;

% set opt and init dataset
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% get video ids for visualization
% sidx = dataset.getSampledIdx();
% 
% sid = dataset.getSeqFrId(sidx);
% run = sidx(ismember(sid,2069));
% run = sidx(ismember(sid,2091));

% i = 70365;
% i = 70368;
% i = 70371;
% i = 70374;
% i = 70377;

% i = 70820;
% i = 70824;
i = 70828;    % ** use this **
% i = 70832;
% i = 70837;
% i = 70841;

[sid, fid] = dataset.getSeqFrId(i);

joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',expID,split,i);
preds = load(pred_file);

repos = zeros(opt.seqLength,17,3);
repos(:,joints,:) = preds.repos;
repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
repos(:,11,:) = preds.repos(:,1,:);
repos = permute(repos,[1,3,2]);
trans = preds.trans;
focal = preds.focal;

eval_file = sprintf('./exp/penn-crop/%s/eval_%s/%05d.mat',expID,split,i);
preds = load(eval_file);
pred2 = zeros(opt.seqLength,17,2);
pred2(:,joints,:) = preds.eval;
pred2(:,1,:) = (preds.eval(:,8,:) + preds.eval(:,9,:))/2;
pred2(:,8,:) = (preds.eval(:,2,:) + preds.eval(:,3,:) + preds.eval(:,8,:) + preds.eval(:,9,:))/4;
pred2(:,9,:) = (preds.eval(:,1,:) + preds.eval(:,2,:) + preds.eval(:,3,:))/3;
pred2(:,11,:) = preds.eval(:,1,:);

figure(1);

seq = dataset.getSeq(i);

% copy input image
im_file = sprintf('data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid);
cp_file = [save_root 'input.jpg'];
if ~exist(cp_file,'file')
    copyfile(im_file,cp_file);
end

% run = 1:16;
% len = [2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4];

run = [1 4 8 9 11 16];
len = [2 2 2 3 3 4];

% show skeleton in 2d
im = imread(im_file);
im_ = uint8(255*ones(size(im,1),420,3));
im_(:,1:size(im,2),:) = im;
im = im_;
imshow(im); hold on
pos2dSkel = pos2dSkel_rr;
for j = 1:numel(run);
    pose = permute(pred2(run(j),:,:),[3 2 1]);
    pose(1,:) = pose(1,:) + 25*max((j-1),0) + 20*max((j-2),0) + 15*max((j-3),0) + 10*max((j-4),0) + 5*max((j-5),0);
    % pose(2,:) = pose(2,:) + 2*(j-1);
    % show projected 2D skeleton
    hpos = show2DPose(pose,pos2dSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',len(j));
    end
    xlabel('');
    ylabel('');
    zlabel('');
    axis off;
end
set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
set(gca,'Position',[0.00 0.00 1.00 1.00]);
set(gcf,'PaperPositionMode','auto');
save_file = [save_root 'skel2d.pdf'];
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf','-r0');
end
    
% show skeleton in 3d
len = [2 2 3 3 4 4];
clf;
set(gcf,'Position',[0.00 0.00 560 560]);
set(gca,'Position',[0.05 0.08 0.90 0.90]);
for j = 1:numel(run);
    set(gca,'fontsize',6);
    pred = permute(repos(run(j),:,:),[2 3 1]);
    pred(1,:) = pred(1,:) + max((j-1),0)*120 + max((j-2),0)*100 + max((j-3),0)*80 + max((j-4),0)*60 + max((j-5),0)*40;
    pred(3,:) = pred(3,:) - max((j-1),0)*120 - max((j-2),0)*100 - max((j-3),0)*80 - max((j-4),0)*60 - max((j-5),0)*40;
    pred(1,:) = pred(1,:) - 100;
    pred(3,:) = pred(3,:) + 100;
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',len(j));
    end
    minx =  -700;  maxx = 2200;
    miny = -1000; maxy = 1000;
    minz =  -600; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'XTick', -700:500:2200);
    set(gca,'YTick',-1000:500:1000);
    set(gca,'ZTick', -600:200:1000);
    set(gca,'ZDir','reverse');
    view([-16,18]);
end
xlabel('');
ylabel('');
zlabel('');
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
save_file = [save_root 'skel3d.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end
    
close;