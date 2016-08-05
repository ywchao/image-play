
config;

% choose source

% % 1. 
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-single-no-skip-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-single-no-skip-clstm/'];

% % 2.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-single-no-skip-res-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-single-no-skip-res-clstm/'];

% % 3.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-256-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-256-clstm/'];

% % 4.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-256-res-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-256-res-clstm/'];

% % 5.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-pf-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-pf-clstm/'];

% % 6.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-pf-res-clstm/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-pf-res-clstm/'];

% % 7.
% FrameRate = 4;
% frame_root = './exp/penn-crop/seq16-hg-pf-res-clstm-resf/preds_val_vis/';
% vis_root = ['./outputs/preds_val_fr' num2str(FrameRate,'%03d') '/seq16-hg-pf-res-clstm-resf/'];


makedir(vis_root);

list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% validation set
valid_ind_file = './data/Penn_Action_cropped/valid_ind.txt';
seq = read_file_lines(valid_ind_file);
seq = cellfun(@(x)str2double(x),seq)';

% limited to first min(K,len) videos for each action
K = 5;
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
keep = false(numel(seq),1);
for i = 1:numel(list_act)
    ii = find(ismember(seq,find(ia == i)));
    keep(ii(1:min(numel(ii),K))) = true;
end
seq = seq(keep);

% reading annotations
fprintf('visualizing videos ... \n');
for i = seq
    fprintf('%04d/%04d\n',i,num_seq);
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    list_ip = cellfun(@(x)x(1:end-7),list_fr,'UniformOutput',false);
    [list_ip,~,ia] = unique(list_ip,'stable');
    % make directory
    vis_dir = [vis_root name_seq '/'];
    makedir(vis_dir);
    % check each frame
    for j = 1:numel(list_ip)        
        % set vis file; skip if existed
        vis_file = [vis_dir list_ip{j} '.avi'];
        if exist(vis_file,'file')
            continue
        end
        % intialize video writer
        v = VideoWriter(vis_file,'Uncompressed AVI');
        v.FrameRate = FrameRate;
        % open new video
        open(v);
        for k = find(ia' == j)
            % read image
            file_im = [fr_dir list_fr{k}];
            im = imread(file_im);
            writeVideo(v,im);
        end
        % close video
        close(v);
    end
end
    
fprintf('\n');