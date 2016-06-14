
config;

FrameRate = 30;

seq = 1890:1983;  % strum_guitar

% choose source
% 1. original frames
% vis_root = ['./outputs/dataset_vis/videos_fr' num2str(FrameRate,'%03d') '/'];
% 2. frames with human poses; need to run vis_body_joint_anno.m first
frame_root = './outputs/dataset_vis/body_joints/';
vis_root = ['./outputs/dataset_vis/videos_fr' num2str(FrameRate,'%03d') '_body_joints/'];

makedir(vis_root);

list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

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