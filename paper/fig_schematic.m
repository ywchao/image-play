
% add paths
addpath('skeleton2d3d/H36M_utils/H36M');
addpath('skeleton2d3d/H36M_utils/utils');
addpath('skeleton2d3d/H36M_utils/external_utils');
addpath('skeleton2d3d/H36M_utils/external_utils/lawrennd-mocap');
addpath('skeleton2d3d/H36M_utils/external_utils/xml_io_tools');

% expID = 'res-64-t2';
expID = 'hg-256-res-64-hg-pred';

split = 'train';

% set save root
save_root = './outputs/figures/schematic/';
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

% init camera
CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};
CameraVertex = CameraVertex / 4;

% set opt and init dataset
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% load libraries
libimg = ip_img();

figure(1);

% i = 1;
% 
% expID = 'seq16-hg-256-res-clstm-res-64-w1e-7';
% % load predictions
% pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',expID,split,i);
% preds = load(pred_file);
% hmap = preds.hmap;
% 
% for j = 1:1
%     hm = squeeze(hmap(j,:,:,:));
%     for k = 1:size(hm,1)
%         % save_file = [save_root sprintf('hmap_%02d_%02d.pdf',j,k)];
%         save_file = [save_root sprintf('hmap_%02d_%02d.png',j,k)];
%         if exist(save_file,'file')
%             continue
%         end
%         hmap_k = libimg.colorHM(squeeze(hm(k,:,:))) * 255;
%         hmap_k = permute(hmap_k,[2 3 1]);
%         hmap_k = uint8(hmap_k);
%         clf;
%         imshow(hmap_k);
%         set(gcf,'Position',[0.00 0.00 size(hmap_k,1) size(hmap_k,2)]);
%         set(gca,'Position',[0.00 0.00 1.00 1.00]);
%         set(gcf,'PaperPositionMode','auto');
%         % print(gcf,save_file,'-dpdf','-r0');
%         print(gcf,save_file,'-dpng','-r0');
%     end
% end

ind = 1:21:64;

% heatmap
for j = ind
    hmap_file = sprintf('./skeleton2d3d/exp/penn-crop/%s/hmap_%s/%05d.mat',expID,split,j);
    hmap = load(hmap_file);
    hmap = hmap.hmap;
    for k = 1:size(hmap,1)
        save_file = [save_root sprintf('hmap_%02d_%02d.png',j,k)];
        if exist(save_file,'file')
            continue
        end
        hm = libimg.colorHM(squeeze(hmap(k,:,:))) * 255;
        hm = permute(hm,[2 3 1]);
        hm = uint8(hm);
        clf;
        imshow(hm);
        set(gcf,'Position',[0.00 0.00 size(hm,2) size(hm,1)]);
        set(gca,'Position',[0.00 0.00 1.00 1.00]);
        set(gcf,'PaperPositionMode','auto');
        
        print(gcf,save_file,'-dpng','-r0');
    end
end

% skeleton
preds = load(['./skeleton2d3d/exp/penn-crop/' expID '/preds_' split '.mat']);
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
repos = zeros(size(preds.repos,1),17,3);
repos(:,joints,:) = preds.repos;
repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
repos(:,11,:) = preds.repos(:,1,:);
repos = permute(repos,[1,3,2]);
trans = preds.trans;
focal = preds.focal;

set(gcf,'Position',[0.00 0.00 560 560]);

for j = ind
    save_file = [save_root sprintf('skel_%02d.png',j)];
    if exist(save_file,'file')
        continue
    end
    clf;
    set(gca,'Position',[0.05 0.08 0.90 0.90]);
    set(gca,'fontsize',6);
    pred = permute(repos(j,:,:),[2 3 1]);
    pred = pred + repmat(permute(trans(j,:),[2 1]),[1 size(pred,2)]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for l = 1:numel(hpos)-1
        set(hpos(l+1),'linewidth',5);
    end
    minx =  -800; maxx =  800;
    miny =     0; maxy = 3500;
    minz =  -800; maxz =  800;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'XTick', -800: 200: 800);
    set(gca,'YTick',    0:1000:3500);
    set(gca,'ZTick', -800: 200: 800);
    set(gca,'ZDir','reverse');
    % view([6,10]);
    view([-8,10]);
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
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r150');
end

close;
