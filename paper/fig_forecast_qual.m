
config;

% add paths
addpath('skeleton2d3d/H36M_utils/H36M');
addpath('skeleton2d3d/H36M_utils/utils');
addpath('skeleton2d3d/H36M_utils/external_utils');
addpath('skeleton2d3d/H36M_utils/external_utils/lawrennd-mocap');
addpath('skeleton2d3d/H36M_utils/external_utils/xml_io_tools');

expID = 'seq16-hg-256-res-clstm-res-64-w1e-6';  mode = 1;

split = 'test';

% set save root
save_root = ['./outputs/figures/forecast_qual_' expID '/'];
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
% posSkel_lr = posSkel;
pos2dSkel_lr = pos2dSkel;
% right leg/right arm
% posSkel_rr = posSkel;
% tmp = posSkel_rr.tree(2:4);
% posSkel_rr.tree(2:4) = posSkel_rr.tree(5:7);
% posSkel_rr.tree(5:7) = tmp;
% posSkel_rr.tree(3).parent = 2;
% posSkel_rr.tree(4).parent = 3;
% posSkel_rr.tree(6).parent = 5;
% posSkel_rr.tree(7).parent = 6;
% posSkel_rr.tree(2).children = 3;
% posSkel_rr.tree(3).children = 4;
% posSkel_rr.tree(5).children = 6;
% posSkel_rr.tree(6).children = 7;
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
% posSkel_ll = posSkel;
% tmp = posSkel_ll.tree(12:14);
% posSkel_ll.tree(12:14) = posSkel_ll.tree(15:17);
% posSkel_ll.tree(15:17) = tmp;
% posSkel_ll.tree(13).parent = 12;
% posSkel_ll.tree(14).parent = 13;
% posSkel_ll.tree(16).parent = 15;
% posSkel_ll.tree(17).parent = 16;
% posSkel_ll.tree(12).children = 13;
% posSkel_ll.tree(13).children = 14;
% posSkel_ll.tree(15).children = 16;
% posSkel_ll.tree(16).children = 17;
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
% posSkel_rl = posSkel_rr;
% tmp = posSkel_rl.tree(12:14);
% posSkel_rl.tree(12:14) = posSkel_rl.tree(15:17);
% posSkel_rl.tree(15:17) = tmp;
% posSkel_rl.tree(13).parent = 12;
% posSkel_rl.tree(14).parent = 13;
% posSkel_rl.tree(16).parent = 15;
% posSkel_rl.tree(17).parent = 16;
% posSkel_rl.tree(12).children = 13;
% posSkel_rl.tree(13).children = 14;
% posSkel_rl.tree(15).children = 16;
% posSkel_rl.tree(16).children = 17;
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

% % get video ids for visualization
% sidx = dataset.getSampledIdx();
% 
% sid = dataset.getSeqFrId(sidx);
% run = sidx(ismember(sid,518));
% run = sidx(ismember(sid,713));
% run = sidx(ismember(sid,730));
% run = sidx(ismember(sid,787));
% run = sidx(ismember(sid,1582));

clear i

% bowling
% i = 14436;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 14439;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 14442;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  % *** use this ***
% i = 14445;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;

% clean and jerk
% i = 19719;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 19736;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 19754;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 19771;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 19788;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;

% i = 22277;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 22287;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 22298;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 22308;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;

% i = 31170;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 31193;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  % *** use this ***
% i = 31215;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 31238;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;

% sit-ups
% i = 52234;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 52239;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  % *** use this ***
% i = 52244;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;
% i = 52249;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;

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

% make directories
save_dir = [save_root sprintf('%04d',sid) '/'];
makedir(save_dir);

% copy input image
im_file = sprintf('data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid);
cp_file = [save_dir 'input.jpg'];
if ~exist(cp_file,'file')
    copyfile(im_file,cp_file);
end

for j = 1:opt.seqLength
    % show gt pose
    save_file = [save_dir sprintf('skelgt_%03d-%02d.pdf',fid,j)];
    if ~exist(save_file,'file')
        clf;
        [~, fid_gt] = dataset.getSeqFrId(seq(j));
        im_file = sprintf('data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid_gt);
        im = imread(im_file);
        imshow(im); hold on;
        % load annotation
        lb_file = [lbdata_root sprintf('%04d.mat',sid)];
        anno = load(lb_file);
        pose = [anno.x(fid_gt,:)' anno.y(fid_gt,:)'];
        % convert to h36m format
        pose_ = zeros(17,2);
        pose_(joints,:) = pose;
        pose_(1,:) = (pose(8,:) + pose(9,:))/2;
        pose_(8,:) = (pose(2,:) + pose(3,:) + pose(8,:) + pose(9,:))/4;
        pose_(9,:) = (pose(1,:) + pose(2,:) + pose(3,:))/3;
        pose_(11,:) = pose(1,:);
        pose = pose_';
        vis = zeros(17,1);
        vis(joints) = anno.visibility(fid_gt,:);
        vis(1) = anno.visibility(fid_gt,8) && anno.visibility(fid_gt,9);
        vis(8) = anno.visibility(fid_gt,2) && anno.visibility(fid_gt,3) && ...
                 anno.visibility(fid_gt,8) && anno.visibility(fid_gt,9);
        vis(9) = anno.visibility(fid_gt,1) && anno.visibility(fid_gt,2) && anno.visibility(fid_gt,3);
        vis(11) = anno.visibility(fid_gt,1);
        % show2DPose(pose,pos2dSkel);
        padding = 0;
        pose = [pose zeros(1, padding)];  %#ok
        vals = bvh2xy(pos2dSkel, pose) ; % * 10
        connect = skelConnectionMatrix(pos2dSkel);
        indices = find(connect);
        [I, J] = ind2sub(size(connect), indices);
        hold on
        grid on
        for i = 1:length(indices)
            if vis(I(i)) == 0 || vis(J(i)) == 0
                continue
            end
            % modify with show part (3d geometrical thing)
            if strncmp(pos2dSkel.tree(I(i)).name,'L',1)
                c = 'r';
            elseif strncmp(pos2dSkel.tree(I(i)).name,'R',1)
                c = 'g';
            else
                c = 'b';
            end
            hl = line([vals(I(i),1) vals(J(i),1)],[vals(I(i),2) vals(J(i),2)],'Color',c);
            set(hl, 'linewidth', 3);
        end
        axis equal
        set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
        set(gca,'Position',[0.00 0.00 1.00 1.00]);
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
    end
    % show projected 2D skeleton
    save_file = [save_dir sprintf('skel2d_%03d-%02d.pdf',fid,j)];
    if ~exist(save_file,'file')
        clf;
        im_file = sprintf('data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid);
        im = imread(im_file);
        imshow(im); hold on;
        show2DPose(permute(pred2(j,:,:),[3 2 1]),pos2dSkel);
        axis off;
        set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
        set(gca,'Position',[0.00 0.00 1.00 1.00]);
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
    end
    % show 3D skeleton relative to center
    save_file = [save_dir sprintf('skel3d_%03d-%02d.pdf',fid,j)];
    if ~exist(save_file,'file')
        clf;
        set(gcf,'Position',[0.00 0.00 560 560]);
        set(gca,'Position',[0.05 0.08 0.90 0.90]);
        set(gca,'fontsize',6);
        pred = permute(repos(j,:,:),[2 3 1]);
        V = pred;
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',4);
        end
        minx = -500; maxx = 500;
        miny = -500; maxy = 500;
        minz = -500; maxz = 500;
        axis([minx maxx miny maxy minz maxz]);
        set(gca,'XTick',-500:200:500);
        set(gca,'YTick',-500:200:500);
        set(gca,'ZTick',-500:200:500);
        set(gca,'ZDir','reverse');
        xlabel('');
        ylabel('');
        zlabel('');
        view([az,el]);
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
    end
end

close;