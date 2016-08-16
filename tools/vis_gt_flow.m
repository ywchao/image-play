
config;

addpath('./ijcv_flow_code/utils');
addpath('./ijcv_flow_code/utils/flowColorCode');

% set parameters
samp = true;

n_phase = 16;

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

res_root = './flownet-release/models/flownet/res_penn-crop_samp/';
vis_root = './flownet-release/models/flownet/vis_penn-crop_samp/';

% first K videos for each action
K = 3;
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
seq = zeros(1,numel(list_act)*K);
for i = 1:numel(list_act)
    ii = find(ia == i);
    seq((i-1)*K+1:i*K) = ii(1:K);
end

figure(1);

% for i = 1:num_seq
for i = seq
    tic_print(sprintf('  %04d/%04d\n',i,num_seq));
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frdata_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    num_fr = numel(list_fr);
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    for j = 1:anno.nframes
        % for now just consider one sequence
        if samp && j > 1
            continue
        end
        % set and make directories
        res_dir = [res_root name_seq '/' num2str(j,'%03d') '/'];
        vis_dir = [vis_root name_seq '/' num2str(j,'%03d') '/'];
        makedir(vis_dir);
        % get seq ind
        ind = linspace(j,j+num_fr-1,n_phase);
        ind = round(ind);
        % remove overlength indices
        rep_ind = ind > num_fr;
        rep_val = max(ind(rep_ind == 0));
        ind(rep_ind) = [];
        % set count
        count = -1;
        for k = 1:numel(ind)-1
            count = count + 1;
            vis_file = [vis_dir num2str(ind(k),'%04d') '_' num2str(ind(k+1),'%04d') '.png'];
            % skip if existed
            if exist(vis_file,'file')
                continue
            end
            im_file_1 = [fr_dir list_fr{ind(k)}];
            im_file_2 = [fr_dir list_fr{ind(k+1)}];
            im_file_f = [res_dir 'flownets-pred-' num2str(count,'%07d') '.flo'];
            im_1 = imread(im_file_1);
            im_2 = imread(im_file_2);
            im_f = readFlowFile(im_file_f);
            
            % blend im_1 and im_2
            alpha = 1/2;
            im_c = uint8(double(im_1)*alpha+double(im_2)*(1-alpha));
            subplot('Position',[0 0 1/5 1]); imshow(im_c);
            
            % compute max flow for color visualization
            mfactor = 0.1;
            maxFlow = max(size(im_f,1),size(im_f,2)) * mfactor;
            subplot('Position',[1/5 0 1/5 1]); imshow(flowToColor(im_f,maxFlow));

            % downsample flow field image
            rfactor = 0.1;
            im_v = imresize(im_f, rfactor);
            im_v = im_v * rfactor;
            [X, Y] = meshgrid(1:size(im_v,2), 1:size(im_v,1));
            subplot('Position',[2/5 0 1/5 1]);
            quiver(X,Y,im_v(:,:,1),im_v(:,:,2),0);
            set(gca,'YDir','Reverse');
            axis image; axis off;

            % warp gt im_1
            [X, Y] = meshgrid(1:size(im_f,2), 1:size(im_f,1));
            im_w = zeros(size(im_1));
            im_w(:,:,1) = interp2(double(im_1(:,:,1)),X+im_f(:,:,1),Y+im_f(:,:,2));
            im_w(:,:,2) = interp2(double(im_1(:,:,2)),X+im_f(:,:,1),Y+im_f(:,:,2));
            im_w(:,:,3) = interp2(double(im_1(:,:,3)),X+im_f(:,:,1),Y+im_f(:,:,2));
            assert(all(uint8(round(im_w(:))) == uint8(im_w(:))) == 1);
            im_w = uint8(im_w);
            subplot('Position',[3/5 0 1/5 1]); imshow(im_w);

            % warp propagated im_1
            if k == 1
                im_1_p = im_1;
            else
                im_1_p = im_p;
            end
            [X, Y] = meshgrid(1:size(im_f,2), 1:size(im_f,1));
            im_p = zeros(size(im_1_p));
            im_p(:,:,1) = interp2(double(im_1_p(:,:,1)),X+im_f(:,:,1),Y+im_f(:,:,2));
            im_p(:,:,2) = interp2(double(im_1_p(:,:,2)),X+im_f(:,:,1),Y+im_f(:,:,2));
            im_p(:,:,3) = interp2(double(im_1_p(:,:,3)),X+im_f(:,:,1),Y+im_f(:,:,2));
            assert(all(uint8(round(im_p(:))) == uint8(im_p(:))) == 1);
            im_p = uint8(im_p);
            subplot('Position',[4/5 0 1/5 1]); imshow(im_p);

            set(gcf,'Position',[0 0 size(im_1,2)*5 size(im_1,1)]);
            set(gcf,'PaperPositionMode','auto');
            set(gcf,'color',[1 1 1]);
            set(gca,'color',[1 1 1]);
            
            print(gcf,vis_file,'-dpng','-r150');
        end
    end
end

close;
