
frame_root = './external/Penn_Action/frames/';
label_root = './external/Penn_Action/labels/';

%  1: head
%  2: right-shoulder
%  3: left-shoulder
%  4: right-elbow
%  5: left-elbow
%  6: right-wrist
%  7: left-wrist
%  8: right-hip
%  9: left-hip
% 10: right-knee
% 11: left-knee
% 12: right-ankle
% 13: left-ankle

% set body joint config
pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
p_no = numel(pa);

% set vis params
msize = 4;
partcolor = {'g','g','g','r','b','r','b','y','y','m','c','m','c'};
frame_rate = 30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose source
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. image only
% vis_mode = 0;
% vis_fr_root = frame_root;
% vis_vd_root = './outputs/vis_dataset/videos_original/';

% 1. annotation with image
vis_mode = 1;
vis_fr_root = './outputs/vis_dataset/frames_joints_w_image/';
vis_vd_root = './outputs/vis_dataset/videos_joints_w_image/';

% 2. annotation without image
% vis_mode = 2;
% vis_fr_root = './outputs/vis_dataset/frames_joints_only/';
% vis_vd_root = './outputs/vis_dataset/videos_joints_only/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

makedir(vis_vd_root);

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% limit to first K videos for each action
K = 10;
action = cell(num_seq,1);
train = zeros(num_seq,1);
for i = 1:num_seq
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
    train(i) = anno.train;
end
[list_act,~,ia] = unique(action, 'stable');
seq = zeros(1,numel(list_act)*K);
for i = 1:numel(list_act)
    % training set only
    ii = find(ia == i & train == 1);
    seq((i-1)*K+1:i*K) = ii(1:K);
end

figure(1);

fprintf('visualizing annotations ... \n');
% for i = 1:num_seq
for i = seq
    % fprintf('%04d/%04d\n',i,num_seq);
    fprintf('%04d/%04d\n',find(i == seq),numel(seq));
    
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    
    % set vis dir for frames
    vis_fr_dir = [vis_fr_root name_seq '/'];
    % output frames
    if vis_mode ~= 0
        makedir(vis_fr_dir);
        for j = 1:anno.nframes
            tic_print(sprintf('  %03d/%03d\n',j,anno.nframes));
            % skip if vis file exists
            vis_file = [vis_fr_dir list_fr{j}];
            if exist(vis_file,'file')
                continue
            end
            % read image
            file_im = [fr_dir list_fr{j}];
            im = imread(file_im);
            % plot body joints on the image
            if exist('h','var')
                delete(h);
            end
            switch vis_mode
                case 1
                    h = imshow(im); hold on;
                case 2
                    im_bg = uint8(64*ones(size(im,1),size(im,2),1));
                    h =  imshow(im_bg); hold on;
            end
            setup_im_gcf(size(im,1),size(im,2));
            for child = 2:p_no
                x1 = anno.x(j,pa(child));
                y1 = anno.y(j,pa(child));
                x2 = anno.x(j,child);
                y2 = anno.y(j,child);
                % skip invisible joints
                if anno.visibility(j,child)
                    plot(x2, y2, 'o', ...
                        'color', partcolor{child}, ...
                        'MarkerSize', msize, ...
                        'MarkerFaceColor', partcolor{child});
                    if anno.visibility(j,pa(child))
                        plot(x1, y1, 'o', ...
                            'color', partcolor{child}, ...
                            'MarkerSize', msize, ...
                            'MarkerFaceColor', partcolor{child});
                        line([x1 x2], [y1 y2], ...
                            'color', partcolor{child}, ...
                            'linewidth',round(msize/2));
                    end
                end
            end
            drawnow;
            % save vis to file
            print(gcf,vis_file,'-djpeg','-r0');
        end
        clf;
    end
    
    % set vis file for video
    vis_file = [vis_vd_root name_seq '.avi'];
    if exist(vis_file,'file')
        continue
    end
    % intialize video writer
    v = VideoWriter(vis_file);
    v.FrameRate = frame_rate;
    % open new video
    open(v);
    for j = 1:anno.nframes
        % read image
        file_im = [vis_fr_dir list_fr{j}];
        im = imread(file_im);
        writeVideo(v,im);
    end
    % close video
    close(v);
end
fprintf('\n');

close;
