
% add paths
addpath('skeleton2d3d/H36M_utils/H36M');
addpath('skeleton2d3d/H36M_utils/utils');
addpath('skeleton2d3d/H36M_utils/external_utils');
addpath('skeleton2d3d/H36M_utils/external_utils/lawrennd-mocap');
addpath('skeleton2d3d/H36M_utils/external_utils/xml_io_tools');

config;

inp_name = 'hg-256';

split = 'test';

% feat_type = 1;  % skel
% feat_type = 2;  % skel + caffenet
% feat_type = 3;  % skel + oracle

% set visibility threshold for training data
% selected by maximizing validation accuracy
% param.thres_vis = 9;

% set number of candidate images retrieved with caffe feature
% selected by maximizing validation accuracy
% param.K = 22500;

% set directories
switch feat_type
    case 1
        exp_name = sprintf('nn-skel-%s-th%02d',inp_name,param.thres_vis);
    case 2
        exp_name = sprintf('nn-skel-caffenet-%s-th%02d-K%05d',inp_name,param.thres_vis,param.K);
    case 3
        exp_name = sprintf('nn-skel-oracle-%s-th%02d',inp_name,param.thres_vis);
end
curr_dir = sprintf('exp/penn-crop/pose-est-%s/eval_%s/',inp_name,split);
pred_dir = sprintf('exp/penn-crop/%s/eval_%s/',exp_name,split);

% set save root
save_root = sprintf('outputs/figures/nn_qual_%s/',exp_name);
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

% set opt
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset = penn_crop(opt, split);

% get video ids for visualization
sidx = dataset.getSampledIdx();

sid = dataset.getSeqFrId(sidx);
run = sidx(ismember(sid,518));

clear i

% bowling
% i = 14436;  pos2dSkel = pos2dSkel_rr;
% i = 14439;  pos2dSkel = pos2dSkel_rr;  % *** use this ***
% i = 14442;  pos2dSkel = pos2dSkel_rr;

[sid, fid] = dataset.getSeqFrId(i);

joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];

% load current frame estimation
curr_file = sprintf('%s%05d.mat',curr_dir,i);
curr = load(curr_file);
curr = squeeze(curr.eval);

% load prediction
pred_file = sprintf('%s%05d.mat',pred_dir,i);
pred = load(pred_file);
nn_sid = pred.sid;
nn_fid = pred.fid;
nn_mse = pred.mse;
pred = pred.eval;
    
% clear figure
clf; clear h;

% load input im
im_file = [frdata_root sprintf('%04d/%06d.jpg',sid,fid)];
im = imread(im_file);

% plot predicted pose
save_file = [save_root sprintf('%04d-%03d-hg.pdf',sid,fid)];
if ~exist(save_file,'file')
    clf;
    % show input image
    imshow(im); hold on;
    % convert to h36m format
    curr_ = zeros(17,2);
    curr_(joints,:) = curr;
    curr_(1,:) = (curr(8,:) + curr(9,:))/2;
    curr_(8,:) = (curr(2,:) + curr(3,:) + curr(8,:) + curr(9,:))/4;
    curr_(9,:) = (curr(1,:) + curr(2,:) + curr(3,:))/3;
    curr_(11,:) = curr(1,:);
    curr = curr_';
    % show2DPose(pose,pos2dSkel);
    padding = 0;
    curr = [curr zeros(1, padding)];
    vals = bvh2xy(pos2dSkel, curr) ; % * 10
    connect = skelConnectionMatrix(pos2dSkel);
    indices = find(connect);
    [I, J] = ind2sub(size(connect), indices);
    hold on
    grid on
    for i = 1:length(indices)
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

% load nn training image
save_file = [save_root sprintf('%04d-%03d-nn.pdf',sid,fid)];
if ~exist(save_file,'file')
    clf;
    % load nn image
    nn_im_file = [frdata_root sprintf('%04d/%06d.jpg',nn_sid,nn_fid)];
    nn_im = imread(nn_im_file);
    imshow(nn_im);
    % load nn gt pose
    nn_lb_file = [lbdata_root sprintf('%04d.mat',nn_sid)];
    nn_anno = load(nn_lb_file);
    pose = [nn_anno.x(nn_fid,:)' nn_anno.y(nn_fid,:)'];
    % convert to h36m format
    pose_ = zeros(17,2);
    pose_(joints,:) = pose;
    pose_(1,:) = (pose(8,:) + pose(9,:))/2;
    pose_(8,:) = (pose(2,:) + pose(3,:) + pose(8,:) + pose(9,:))/4;
    pose_(9,:) = (pose(1,:) + pose(2,:) + pose(3,:))/3;
    pose_(11,:) = pose(1,:);
    pose = pose_';
    vis = zeros(17,1);
    vis(joints) = nn_anno.visibility(nn_fid,:);
    vis(1) = nn_anno.visibility(nn_fid,8) && nn_anno.visibility(nn_fid,9);
    vis(8) = nn_anno.visibility(nn_fid,2) && nn_anno.visibility(nn_fid,3) && ...
        nn_anno.visibility(nn_fid,8) && nn_anno.visibility(nn_fid,9);
    vis(9) = nn_anno.visibility(nn_fid,1) && nn_anno.visibility(nn_fid,2) && nn_anno.visibility(nn_fid,3);
    vis(11) = nn_anno.visibility(nn_fid,1);
    % show2DPose(pose,pos2dSkel);
    padding = 0;
    pose = [pose zeros(1, padding)];
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

% display pred pose
for j = 1:opt.seqLength
    save_file = [save_root sprintf('%04d-%03d-pred-%02d.pdf',sid,fid,j)];
    if exist(save_file,'file')
        continue
    end
    % plot pred pose on input
    clf;
    % show input image
    imshow(im); hold on;
    % convert to h36m format
    pose = squeeze(pred(j,:,:));
    pose_ = zeros(17,2);
    pose_(joints,:) = pose;
    pose_(1,:) = (pose(8,:) + pose(9,:))/2;
    pose_(8,:) = (pose(2,:) + pose(3,:) + pose(8,:) + pose(9,:))/4;
    pose_(9,:) = (pose(1,:) + pose(2,:) + pose(3,:))/3;
    pose_(11,:) = pose(1,:);
    pose = pose_';
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
        if pose(1,I(i)) == Inf || pose(2,J(i)) == Inf
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

close;