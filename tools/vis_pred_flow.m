
addpath('./ijcv_flow_code/utils/flowColorCode');

% set parameters

% split = 'val';
% split = 'train';

exp = 'seq16-hg-pf-clstm';

opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% set flow file
h5_flow = ['./exp/penn-crop/' exp '/flows_' split '.h5'];

% set output dir
vis_root = ['./outputs/flows_' split '/' exp '/'];

% load flow and input
flows = hdf5read(h5_flow,'flows');
flows = permute(flows, [2 1 3 4 5]);
targets = hdf5read(h5_flow,'targets');
targets = permute(targets, [2 1 3 4 5]);
targets = targets * 255;

% init dataset
dataset = penn_crop(opt, split);

% get sample indeices
sidx = dataset.getSampleIdx();

figure(1);

upfactor = 4;

for i = 1:numel(sidx)
    tic_print(sprintf('  %04d/%04d\n',i,numel(sidx)));
    [sid, fid] = dataset.getSeqFrId(sidx(i));
    % set vis dir
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    if ~exist(vis_dir,'dir')
        makedir(vis_dir);
    end
    % get the first frame
    ip = targets(:, :, :, 1, i);
    ip = imresize(ip, upfactor);
    for j = 1:opt.seqLength
        % set vis file
        vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];
        if exist(vis_file,'file')
            continue
        end
        % get backward flow
        fl = flows(:, :, :, j ,i);
        % fl = imresize(fl, upfactor);
        fl(:,:,1) = (size(fl,1)-1) * ((fl(:,:,1) - (-1)) / (1-(-1))) + 1;  % y
        fl(:,:,2) = (size(fl,2)-1) * ((fl(:,:,2) - (-1)) / (1-(-1))) + 1;  % x
        [X, Y] = meshgrid(1:size(fl,2), 1:size(fl,1));
        fl(:,:,1) = fl(:,:,1)-Y;
        fl(:,:,2) = fl(:,:,2)-X;
        % get backward flow (downsampled)
        rfactor = 0.1 * upfactor;
        vec = flows(:, :, :, j ,i);
        vec = imresize(vec, rfactor);
        vec(:,:,1) = (size(vec,1)-1) * ((vec(:,:,1) - (-1)) / (1-(-1))) + 1;  % y
        vec(:,:,2) = (size(vec,2)-1) * ((vec(:,:,2) - (-1)) / (1-(-1))) + 1;  % x
        [X, Y] = meshgrid(1:size(vec,2), 1:size(vec,1));
        vec(:,:,1) = vec(:,:,1)-Y;
        vec(:,:,2) = vec(:,:,2)-X;
        % get ground-truth
        gt = targets(:, :, :, j, i);
        gt = imresize(gt, upfactor);
        % get blended image
        alpha = 1/2;
        im_c = uint8(double(ip)*alpha+double(gt)*(1-alpha));
        % plot
        subplot('Position',[0 0 1/3 1]); imshow(im_c);
        subplot('Position',[1/3 0 1/3 1]); imshow(flowToColor(fl));
        subplot('Position',[2/3 0 1/3 1]); quiver(X,Y,vec(:,:,1),vec(:,:,2));  
        set(gca,'YDir','Reverse');
        axis image; axis off;
        % get figure properties
        set(gcf,'Position',[0 0 size(im_c,2)*3 size(im_c,1)]);
        set(gcf,'PaperPositionMode','auto');
        set(gcf,'color',[1 1 1]);
        set(gca,'color',[1 1 1]);
        % save figure
        print(gcf,vis_file,'-dpng','-r150');
    end
end

close;
