
frame_root = './external/Penn_Action/frames/';
label_root = './external/Penn_Action/labels/';

save_dir = 'outputs/vis_dataset/';
makedir(save_dir);

list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% reading annotations
action = cell(num_seq,1);
train = zeros(num_seq,1);
num_fr = zeros(num_seq,1);

fprintf('reading annotations ... \n');
for i = 1:num_seq
    tic_print(sprintf('%04d/%04d\n',i,num_seq));
    
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    num_fr(i) = numel(list_fr);
    
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    
    assert(ischar(anno.action));
    action{i} = anno.action;
    
    assert(anno.train == 1 || anno.train == -1);
    train(i) = anno.train;
end
fprintf('\n');

% show action list
[list_act,~,ia] = unique(action, 'stable');
num_act = numel(list_act);
fprintf('action list:\n');
for i = 1:numel(list_act)
    fprintf('  %02d %s\n',i,list_act{i});
end
fprintf('\n');

% show number of sequences
fprintf('number of sequences:\n');
fprintf('  training:  %4d\n',sum(train == 1));
fprintf('  test:      %4d\n',sum(train == -1));
fprintf('  total:     %4d\n',num_seq);
fprintf('\n');

% show number of frames per sequence
figure(1);
bar(sort(num_fr));
grid on;
axis([0.5 num_seq+0.5 ylim]);
save_file = [save_dir 'hist_frame.pdf'];
title(sprintf('max: %d / min: %d / mean: %5.2f / total: %d', ...
    max(num_fr),min(num_fr),mean(num_fr),sum(num_fr)));
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf');
end
close;

% show number statistics for actions
num_seq_act_tr = zeros(num_act,1);
num_seq_act_ts = zeros(num_act,1);
num_fr_act_tr = zeros(num_act,1);
num_fr_act_ts = zeros(num_act,1);
for i = 1:numel(list_act)
    num_seq_act_tr(i) = sum(ia == i & train == 1);
    num_seq_act_ts(i) = sum(ia == i & train == -1);
    num_fr_act_tr(i) = sum(num_fr(ia == i & train == 1));
    num_fr_act_ts(i) = sum(num_fr(ia == i & train == -1));
end

figure(1);
list_act_vis = cellfun(@(x)strrep(x,'_',' '),list_act,'UniformOutput',false);

subplot(3,3,1); barh(num_seq_act_tr); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 250 0.5 numel(list_act)+0.5]);
title('# sequences: training');
subplot(3,3,2); barh(num_seq_act_ts); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 250 0.5 numel(list_act)+0.5]);
title('# sequences: test');
subplot(3,3,3); barh(num_seq_act_tr+num_seq_act_ts); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 250 0.5 numel(list_act)+0.5]);
title('# sequences: total');

subplot(3,3,4); barh(num_fr_act_tr); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 25000 0.5 numel(list_act)+0.5]);
title('# frames: training');
subplot(3,3,5); barh(num_fr_act_ts); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 25000 0.5 numel(list_act)+0.5]);
title('# frames: test');
subplot(3,3,6); barh(num_fr_act_tr+num_fr_act_ts); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 25000 0.5 numel(list_act)+0.5]);
title('# frames: total');

subplot(3,3,7); barh(num_fr_act_tr ./ num_seq_act_tr); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 300 0.5 numel(list_act)+0.5]);
title('# frames / sequence: training');
subplot(3,3,8); barh(num_fr_act_ts ./ num_seq_act_ts); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 300 0.5 numel(list_act)+0.5]);
title('# frames / sequence: test');
subplot(3,3,9); barh((num_fr_act_tr+num_fr_act_ts) ./ (num_seq_act_tr+num_seq_act_ts)); grid on;
set(gca,'Ydir','reverse');
set(gca,'YTickLabel',list_act_vis);
set(gca,'fontsize',4);
axis([0 300 0.5 numel(list_act)+0.5]);
title('# frames / sequence: total');

save_file = [save_dir 'stats_action.pdf'];
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf');
end

close;
