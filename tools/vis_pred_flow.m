
addpath('./ijcv_flow_code/utils/flowColorCode');

% set parameters

% split = 'train';
% split = 'val';

% exp = 'seq16-hg-pf-res-clstm';
% opt.seqType = 'phase';

% exp = 'seq16raw-hg-pf-res-clstm';
% opt.seqType = 'raw';

opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% set flow file
h5_flow = ['./exp/penn-crop/' exp '/flows_' split '.h5'];

% set output dir
vis_root = ['./outputs/flows_' opt.seqType '_' split '/' exp '/'];

% load flow and gt flow
flows = hdf5read(h5_flow,'flows');
flows = permute(flows, [5 4 3 2 1]);
gtflows = hdf5read(h5_flow,'gtflows');
gtflows = permute(gtflows, [5 4 3 2 1]);

% init dataset
dataset = penn_crop(opt, split);

% get sample indeices
sidx = dataset.getSampledIdx();

figure(1);

for i = 1:numel(sidx)
    tic_print(sprintf('  %04d/%04d\n',i,numel(sidx)));
    [sid, fid] = dataset.getSeqFrId(sidx(i));
    % set vis dir
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    makedir(vis_dir);
    % skip if all exist
    list_file = dir([vis_dir sprintf('%03d-',fid) '*']);
    list_file = {list_file.name}';
    if numel(list_file) == opt.seqLength
        continue
    end
    % load input
    input = dataset.get(sidx(i));
    for j = 1:opt.seqLength
        % set vis file
        vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];

        % show image pair
        im_1 = input(j,:,:,:);
        if j == opt.nPhase
            im_2 = input(j,:,:,:);
        else
            im_2 = input(j+1,:,:,:);
        end
        im_1 = permute(im_1, [3 4 2 1]);
        im_2 = permute(im_2, [3 4 2 1]);
        alpha = 1/2;
        im_c = uint8(double(im_1)*alpha+double(im_2)*(1-alpha));
        subplot('Position',[0 0 1/9 1]); imshow(im_c);

        % resize im for warping
        im_1_r = imresize(im_1, opt.outputRes/opt.inputRes);

        % show gt flow
        gtfl = gtflows(i, j,: , :, :);
        gtfl = permute(gtfl, [4 5 3 2 1]);
        mfactor = 0.1;
        maxFlow = max(size(gtfl,1),size(gtfl,2)) * mfactor;
        subplot('Position',[1/9 0 1/9 1]); imshow(flowToColor(gtfl,maxFlow));
        rfactor = 0.25;
        gtfl_v = imresize(gtfl, rfactor);
        gtfl_v = gtfl_v * rfactor;
        [X, Y] = meshgrid(1:size(gtfl_v,2), 1:size(gtfl_v,1));
        subplot('Position',[2/9 0 1/9 1]);
        quiver(X,Y,gtfl_v(:,:,1),gtfl_v(:,:,2),0);
        set(gca,'YDir','Reverse');
        axis image; axis off;

        % show warped image with gt flow
        [X, Y] = meshgrid(1:size(gtfl,2), 1:size(gtfl,1));
        im_w = zeros(size(im_1_r));
        im_w(:,:,1) = interp2(double(im_1_r(:,:,1)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        im_w(:,:,2) = interp2(double(im_1_r(:,:,2)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        im_w(:,:,3) = interp2(double(im_1_r(:,:,3)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        assert(all(uint8(round(im_w(:))) == uint8(im_w(:))) == 1);
        im_w = uint8(im_w);
        im_w = imresize(im_w, opt.inputRes/opt.outputRes);
        subplot('Position',[3/9 0 1/9 1]); imshow(im_w);

        % warp propagated im_1 with gt flow
        if j == 1
            im_1_p = im_1_r;
        else
            im_1_p = im_p_gt;
        end
        [X, Y] = meshgrid(1:size(gtfl,2), 1:size(gtfl,1));
        im_p_gt = zeros(size(im_1_p));
        im_p_gt(:,:,1) = interp2(double(im_1_p(:,:,1)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        im_p_gt(:,:,2) = interp2(double(im_1_p(:,:,2)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        im_p_gt(:,:,3) = interp2(double(im_1_p(:,:,3)),X+gtfl(:,:,1),Y+gtfl(:,:,2));
        assert(all(uint8(round(im_p_gt(:))) == uint8(im_p_gt(:))) == 1);
        im_p_gt = uint8(im_p_gt);
        im_v_gt = imresize(im_p_gt, opt.inputRes/opt.outputRes);
        subplot('Position',[4/9 0 1/9 1]); imshow(im_v_gt);

        % show pred flow
        fl = flows(i, j,: , :, :);
        fl = permute(fl, [4 5 3 2 1]);
        mfactor = 0.1;
        maxFlow = max(size(fl,1),size(fl,2)) * mfactor;
        subplot('Position',[5/9 0 1/9 1]); imshow(flowToColor(fl,maxFlow));
        rfactor = 0.25;
        fl_v = imresize(fl, rfactor);
        fl_v = fl_v * rfactor;
        [X, Y] = meshgrid(1:size(fl_v,2), 1:size(fl_v,1));
        subplot('Position',[6/9 0 1/9 1]);
        quiver(X,Y,fl_v(:,:,1),fl_v(:,:,2),0);
        set(gca,'YDir','Reverse');
        axis image; axis off;

        % show warped image with pred flow
        [X, Y] = meshgrid(1:size(fl,2), 1:size(fl,1));
        im_w = zeros(size(im_1_r));
        im_w(:,:,1) = interp2(double(im_1_r(:,:,1)),X+fl(:,:,1),Y+fl(:,:,2));
        im_w(:,:,2) = interp2(double(im_1_r(:,:,2)),X+fl(:,:,1),Y+fl(:,:,2));
        im_w(:,:,3) = interp2(double(im_1_r(:,:,3)),X+fl(:,:,1),Y+fl(:,:,2));
        assert(all(uint8(round(im_w(:))) == uint8(im_w(:))) == 1);
        im_w = uint8(im_w);
        im_w = imresize(im_w, opt.inputRes/opt.outputRes);
        subplot('Position',[7/9 0 1/9 1]); imshow(im_w);

        % warp propagated im_1 with pred flow
        if j == 1
            im_1_p = im_1_r;
        else
            im_1_p = im_p_pr;
        end
        [X, Y] = meshgrid(1:size(fl,2), 1:size(fl,1));
        im_p_pr = zeros(size(im_1_p));
        im_p_pr(:,:,1) = interp2(double(im_1_p(:,:,1)),X+fl(:,:,1),Y+fl(:,:,2));
        im_p_pr(:,:,2) = interp2(double(im_1_p(:,:,2)),X+fl(:,:,1),Y+fl(:,:,2));
        im_p_pr(:,:,3) = interp2(double(im_1_p(:,:,3)),X+fl(:,:,1),Y+fl(:,:,2));
        assert(all(uint8(round(im_p_pr(:))) == uint8(im_p_pr(:))) == 1);
        im_p_pr = uint8(im_p_pr);
        im_v_pr = imresize(im_p_pr, opt.inputRes/opt.outputRes);
        subplot('Position',[8/9 0 1/9 1]); imshow(im_v_pr);

        % get figure properties
        set(gcf,'Position',[0 0 size(im_c,2)*9 size(im_c,1)]);
        set(gcf,'PaperPositionMode','auto');
        set(gcf,'color',[1 1 1]);
        set(gca,'color',[1 1 1]);

        % save figure
        print(gcf,vis_file,'-dpng','-r150');
    end
end

close;
