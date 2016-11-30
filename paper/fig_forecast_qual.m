
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

% l-leg/r-arm (up)
pos2dSkel_lr = pos2dSkel;
% r-leg/r-arm (up)
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
% l-leg/l-arm (up)
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
% r-leg/l-arm (up)
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
% 
% run = sidx(ismember(sid,1300));
% run = sidx(ismember(sid,268));
% run = sidx(ismember(sid,295));
% run = sidx(ismember(sid,2027));
% run = sidx(ismember(sid,1098));
% run = sidx(ismember(sid,360));
% run = sidx(ismember(sid,2144));
% run = sidx(ismember(sid,2308));

clear i

% bowling
% i = 14436;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 14439;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 14442;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 14445;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% clean and jerk
% i = 19719;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 19736;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 19754;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 19771;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 19788;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% i = 22277;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 22287;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 22298;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 22308;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% i = 31170;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 31193;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 31215;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 31238;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% sit-ups
% i = 52234;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 52239;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 52244;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 52249;  pos2dSkel = pos2dSkel_rr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% pull-ups
% i = 43877;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 43879;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;  % *** use this ***
% i = 43882;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 43884;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;

% basebasll swing
% i = 7384;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 7387;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 7390;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 7393;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% i = 7809;  pos2dSkel = pos2dSkel_rl;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 7812;  pos2dSkel = pos2dSkel_rl;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 7815;  pos2dSkel = pos2dSkel_rl;  az = 33;  el = 20;  minc = -600; maxc = 600;  % *** use this ***
% i = 7818;  pos2dSkel = pos2dSkel_rl;  az = 33;  el = 20;  minc = -600; maxc = 600;

% tennis forehand
% i = 69442;  pos2dSkel = pos2dSkel_ll;  az = 33;  el = 20;  minc = -600; maxc = 600;  % *** use this ***
% i = 69446;  pos2dSkel = pos2dSkel_ll;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 69449;  pos2dSkel = pos2dSkel_ll;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 69453;  pos2dSkel = pos2dSkel_ll;  az = 33;  el = 20;  minc = -600; maxc = 600;

% jumping jacks
% i = 38632;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -800; maxc = 800;
% i = 38633;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -800; maxc = 800;  % *** use this ***
% i = 38635;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -800; maxc = 800;
% i = 38636;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -800; maxc = 800;

% bench press
% i = 8811;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 8814;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 8817;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 8820;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;

% tennis serve
% i = 72221;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 72223;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;  % *** use this ***
% i = 72226;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;
% i = 72229;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -600; maxc = 600;

% i = 75806;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 75810;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;  % *** use this ***
% i = 75815;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;
% i = 75820;  pos2dSkel = pos2dSkel_lr;  az = 33;  el = 20;  minc = -500; maxc = 500;


% for generating videos
render_dir_0 = sprintf('./outputs/render_scape_penn/%s/texture_%1d/%s/', expID, 0, split);
render_dir_1 = sprintf('./outputs/render_scape_penn/%s/texture_%1d/%s/', expID, 1, split);
fr_len = 235;

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
im = imread(im_file);

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
        % set coordinates of invisible joints to 0
        %   do not use vis anymore, since the joint order has been changed
        pose(:, vis == 0) = 0;
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
            if vals(I(i),1) == 0 || vals(J(i),1) == 0
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
        set(gcf,'color','w');
        print(gcf,save_file,'-dpdf','-r0');
        % also save it in png
        print(gcf,strrep(save_file,'.pdf','.png'),'-dpng','-r0');
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
        set(gcf,'color','w');
        print(gcf,save_file,'-dpdf','-r0');
        % also save it in png
        print(gcf,strrep(save_file,'.pdf','.png'),'-dpng','-r0');
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
        % minx = -500; maxx = 500;
        % miny = -500; maxy = 500;
        % minz = -500; maxz = 500;
        % axis([minx maxx miny maxy minz maxz]);
        axis([minc maxc minc maxc minc maxc]);
        set(gca,'XTick',minc:200:maxc);
        set(gca,'YTick',minc:200:maxc);
        set(gca,'ZTick',minc:200:maxc);
        set(gca,'ZDir','reverse');
        xlabel('');
        ylabel('');
        zlabel('');
        view([az,el]);
        set(gcf,'color','w');
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
        % also save it in png
        print(gcf,strrep(save_file,'.pdf','.png'),'-dpng','-r0');
    end
    % video frames
    save_file = [save_dir sprintf('vid_frame_%03d-%02d.png',fid,j)];
    if ~exist(save_file,'file')
        file_1 = [save_dir sprintf('skelgt_%03d-%02d.png',fid,j)];
        file_2 = [save_dir sprintf('skel2d_%03d-%02d.png',fid,j)];
        file_3 = [save_dir sprintf('skel3d_%03d-%02d.png',fid,j)];
        file_4 = [render_dir_0 sprintf('%04d/%03d-%02d.png',sid,fid,j)];
        file_5 = [render_dir_1 sprintf('%04d/%03d-%02d.png',sid,fid,j)];
        assert(exist(file_4,'file') && exist(file_5,'file'));
        im_all = cell(3,1);
        im_all{1} = imread(file_1);
        im_all{2} = imread(file_2);
        im_all{3} = imread(file_3);
        im_all{3} = imresize(im_all{3},[fr_len,NaN]);
        im_all{4} = imread(file_4);
        im_all{5} = imread(file_5);
        assert(all(size(im_all{4}) == size(im_all{1})));
        assert(all(size(im_all{5}) == size(im_all{1})));
        if size(im_all{1},1) > size(im_all{1},2)
            im_all{1} = imresize(im_all{1},[fr_len,NaN]);
            im_all{2} = imresize(im_all{2},[fr_len,NaN]);
            im_all{4} = imresize(im_all{4},[fr_len,NaN]);
            im_all{5} = imresize(im_all{5},[fr_len,NaN]);
            p1 = ceil((size(im_all{1},1) - size(im_all{1},2))/2);
            p2 = floor((size(im_all{1},1) - size(im_all{1},2))/2);
            im_all{1} = padarray(im_all{1},[0 p1],0,'pre');
            im_all{1} = padarray(im_all{1},[0 p2],0,'post');
            im_all{2} = padarray(im_all{2},[0 p1],0,'pre');
            im_all{2} = padarray(im_all{2},[0 p2],0,'post');
            im_all{4} = padarray(im_all{4},[0 p1],0,'pre');
            im_all{4} = padarray(im_all{4},[0 p2],0,'post');
            im_all{5} = padarray(im_all{5},[0 p1],0,'pre');
            im_all{5} = padarray(im_all{5},[0 p2],0,'post');
        else
            im_all{1} = imresize(im_all{1},[NaN,fr_len]);
            im_all{2} = imresize(im_all{2},[NaN,fr_len]);
            im_all{4} = imresize(im_all{4},[NaN,fr_len]);
            im_all{5} = imresize(im_all{5},[NaN,fr_len]);
            p1 = ceil((size(im_all{1},2) - size(im_all{1},1))/2);
            p2 = floor((size(im_all{1},2) - size(im_all{1},1))/2);
            im_all{1} = padarray(im_all{1},[p1 0],0,'pre');
            im_all{1} = padarray(im_all{1},[p2 0],0,'post');
            im_all{2} = padarray(im_all{2},[p1 0],0,'pre');
            im_all{2} = padarray(im_all{2},[p2 0],0,'post');
            im_all{4} = padarray(im_all{4},[p1 0],0,'pre');
            im_all{4} = padarray(im_all{4},[p2 0],0,'post');
            im_all{5} = padarray(im_all{5},[p1 0],0,'pre');
            im_all{5} = padarray(im_all{5},[p2 0],0,'post');
        end
        assert(size(im_all{3},1) == size(im_all{3},2));
        row = size(im_all{1},1);
        col = sum(cellfun(@(x)size(x,2),im_all));
        im_con = uint8(zeros(row,col,3));
        for k = 1:3
            for m = 1:numel(im_all)
                ind1 = sum(cellfun(@(x)size(x,2),im_all(1:m-1)))+1;
                ind2 = sum(cellfun(@(x)size(x,2),im_all(1:m-1)))+size(im_all{m},2);
                im_con(:,ind1:ind2,k) = im_all{m}(:,:,k);
            end
        end
        clf;
        imshow(im_con);
        set(gcf,'Position',[0.00 0.00 size(im_con,2) size(im_con,1)/0.85]);
        set(gca,'Position',[0.00 0.15 1.00 0.85]);
        set(gcf,'color','k');
        set(gcf,'inverthardcopy','off')
        xh = xlabel('GT Frame & Pose                    Forecasted 2D Pose                  Forecasted 3D Pose                                   Rendered Human Character                  ');
        set(xh,'Color','w');
        set(xh,'Position',get(xh,'Position') + [0, -20, 0]);
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpng','-r0');
    end
end

% input frame
save_file = [save_dir sprintf('vid_input_%03d.png',fid)];
if ~exist(save_file,'file')
    im_file = [save_dir 'input.jpg'];
    im = imread(im_file);
    if size(im,1) > size(im,2)
        im = imresize(im,[fr_len,NaN]);
        p1 = ceil((size(im,1) - size(im,2))/2);
        p2 = floor((size(im,1) - size(im,2))/2);
        im = padarray(im,[0 p1],0,'pre');
        im = padarray(im,[0 p2],0,'post');
    else
        im = imresize(im,[NaN,fr_len]);
        p1 = ceil((size(im,2) - size(im,1))/2);
        p2 = floor((size(im,2) - size(im,1))/2);
        im = padarray(im,[p1 0],0,'pre');
        im = padarray(im,[p2 0],0,'post');
    end
    p1 = fr_len*2;
    p2 = fr_len*2;
    im = padarray(im,[0 p1],0,'pre');
    im = padarray(im,[0 p2],0,'post');
    clf;
    imshow(im);
    set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)/0.85]);
    set(gca,'Position',[0.00 0.15 1.00 0.85]);
    set(gcf,'color','k');
    set(gcf,'inverthardcopy','off')
    xh = xlabel('Input Image');
    set(xh,'Color','w');
    set(xh,'Position',get(xh,'Position') + [0, -20, 0]);
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r0');
end

close;

% generate video
vid_file = [save_dir sprintf('%04d-%03d',sid,fid) '.avi'];
FrameRate = 2;
InpFrame = 4;
GapFrame = 2;
if ~exist(vid_file,'file')
    % intialize video writer
    v = VideoWriter(vid_file,'Uncompressed AVI');
    v.FrameRate = FrameRate;
    % open new video
    open(v);
    % input frame
    im_inp = imread([save_dir sprintf('vid_input_%03d.png',fid)]);
    for j = 1:InpFrame
        writeVideo(v,im_inp);
    end
    % gap frame
    im_gap = uint8(zeros(size(im_inp)));
    for j = 1:GapFrame
        writeVideo(v,im_gap);
    end
    % video frames
    for j = 1:opt.seqLength
        % read image
        file_im = [save_dir sprintf('vid_frame_%03d-%02d.png',fid,j)];
        im = imread(file_im);
        assert(all(size(im) == size(im_inp)));
        writeVideo(v,im);
    end
    % close video
    close(v);
end
