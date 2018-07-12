
frdata_root = './data/penn-crop/frames/';
lbdata_root = './data/penn-crop/labels/';

n_phase = 16;

% get list of sequences
list_seq = dir([lbdata_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

vis_root = './outputs/vis_action_phase/';

% reading annotations
fprintf('visualizing action phase ... \n');
for i = 1:num_seq
    tic_print(sprintf('  %04d/%04d\n',i,num_seq));
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frdata_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    % load annotation
    lb_file = [lbdata_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    % make directory
    vis_dir = [vis_root name_seq '/'];
    makedir(vis_dir);
    % get sample id
    num_fr = numel(list_fr);
    t_stride = (num_fr-1)/(n_phase-1);
    samp_id = 1:t_stride:num_fr;
    assert(numel(samp_id) == n_phase);
    % save sampled frame
    for j = samp_id
        id = round(j);
        % skip if vis file exists
        vis_file = [vis_dir list_fr{id}];
        if exist(vis_file,'file')
            continue
        end
        % copy image
        file_im = [fr_dir list_fr{id}];
        copyfile(file_im, vis_file);
    end
end
fprintf('done.\n');
