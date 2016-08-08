
config;

% set parameters

% sample == true is used for visualization only
% samp = true; num_batch = 1;
% samp = false; num_batch = 20;

n_phase = 16;

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

postfix = '';
if samp
    postfix = '_samp';
end

input_root  = ['./data/Penn_Action_cropped/flownet' postfix '/flownet_input/'];
script_temp = ['./data/Penn_Action_cropped/flownet' postfix '/flownet_run_%02d.sh'];
output_root = ['./res_penn-crop' postfix '/'];
makedir(input_root);

C = cell(0,1);

for i = 1:num_seq
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
    % make directories
    input_dir = [input_root name_seq '/'];
    if ~exist(input_dir,'dir')
        makedir(input_dir);
    end
    % get absolute path
    % use symlink path on ilcomp
    list = cellfun(@(x)['./penn-crop/frames/' name_seq '/' x],list_fr,'UniformOutput',false);
    for j = 1:anno.nframes
        % for now just consider one sequence
        if samp && j > 1
            continue
        end
        save_list_1 = [input_dir num2str(j,'%0d') '_list_1.txt'];
        save_list_2 = [input_dir num2str(j,'%0d') '_list_2.txt'];
        % add to script
        % use symlink path on ilcomp
        output_dir = [output_root name_seq '/' num2str(j,'%03d')];
        C{end+1,1} = [ ...
            strrep(save_list_1,input_root,['./penn-crop/flownet' postfix '/flownet_input/']) ' ' ...
            strrep(save_list_2,input_root,['./penn-crop/flownet' postfix '/flownet_input/']) ' ' ...
            output_dir];
        % skip if files exist
        if exist(save_list_1,'file') && exist(save_list_2,'file')
            continue
        end
        % get seq ind
        ind = linspace(j,j+num_fr-1,n_phase);
        ind = round(ind);
        ind(ind > num_fr) = num_fr;
        % get list
        list_1 = list(ind(1:end-1));
        list_2 = list(ind(2:end));
        % write to file
        write_file_lines(save_list_1, list_1);
        write_file_lines(save_list_2, list_2);
    end
end
fprintf('done.\n');

% generate scripts (batch mode)
C = cellfun(@(x)['./demo_flownet.py S ' x],C,'UniformOutput',false);
len = numel(C);
interval = round(len / num_batch);
ss = 1:interval:len;
sid = ss(1:num_batch);
eid = [ss(2:num_batch)-1 len];
for i = 1:num_batch
    script_file = sprintf(script_temp,i);
    if ~exist(script_file,'file')
        Cb = [{'#!/bin/bash'}; C(sid(i):eid(i))];
        write_file_lines(script_file,Cb);
        edit_file_permission(script_file,'755');
    end
end
