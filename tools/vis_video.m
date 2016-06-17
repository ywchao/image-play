
config;

% choose source

% 1. original frames
% FrameRate = 30;
% vis_root = ['./outputs/dataset_vis/videos_fr' num2str(FrameRate,'%03d') '/'];

% 2. frames with gt human poses; need to run vis_body_joint_anno.m first
% FrameRate = 30;
% frame_root = './outputs/dataset_vis/body_joints/';
% vis_root = ['./outputs/dataset_vis/videos_fr' num2str(FrameRate,'%03d') '_body_joints/'];

% 3. only gt human poses; need to run vis_body_joint_anno.m first
% FrameRate = 30;
% frame_root = './outputs/dataset_vis/body_joints_only/';
% vis_root = ['./outputs/dataset_vis/videos_fr' num2str(FrameRate,'%03d') '_body_joints_only/'];

% 4. frames with estimated human poses on cropped image; need to have penn_action_vis/ under caches/
FrameRate = 10;
frame_root = './caches/pose_penn_vis/';
vis_root = ['./outputs/pose_penn_vis/videos_fr' num2str(FrameRate,'%03d') '/'];

makedir(vis_root);

list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% strum_guitar
% seq = 1890:1983;

% first K videos for each action
K = 10;
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

% reading annotations
fprintf('visualizing videos ... \n');
% for i = 1:num_seq
for i = seq
    fprintf('%04d/%04d\n',i,num_seq);
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    % set vis file; skip if existed
    vis_file = [vis_root name_seq '.avi'];
    if exist(vis_file,'file')
        continue
    end
    % intialize video writer
    v = VideoWriter(vis_file,'Uncompressed AVI');
    v.FrameRate = FrameRate;
    % open new video
    open(v);
    for j = 1:anno.nframes
        % read image
        file_im = [fr_dir list_fr{j}];
        im = imread(file_im);
        writeVideo(v,im);
    end
    % close video
    close(v);
end
fprintf('\n');