% eval one
%   eval_seq:      eval sequences
%   list_action:   list of action names
%   aid:           action ids
%   pred_dir:      pred file directory
%   seq_len:       sequence lenth

num_seq = size(eval_seq,1);
num_pt = 13;
pck_th = 0.05;

% compute distance
dist = cell(seq_len,1);
for i = 1:seq_len
    dist{i} = zeros(num_seq,num_pt);
end
fprintf('computing distance ... \n');
for i = 1:num_seq
    if mod(i,10000) == 0 || i == num_seq
        fprintf('  %05d/%05d\n',i,num_seq);
    end
    sid = eval_seq(i,1);
    fid = eval_seq(i,2:end);
    % load annotation
    lb_file = ['./data/penn-crop/labels/' num2str(sid,'%04d') '.mat'];
    anno = load(lb_file);
    % load prediction
    pred_file = sprintf('%s/%05d.mat',pred_dir,i);
    pred = load(pred_file);
    pred = pred.eval;
    % get ref
    ref = max(anno.dimensions(1:2));
    % compute distance
    for t = 1:seq_len
        gtpts = [anno.x(fid(t),:)' anno.y(fid(t),:)'];
        gtpts(anno.visibility(fid(t),:) == 0,:) = NaN;
        dist{t}(i,:) = sqrt(sum((squeeze(pred(t,:,:)) - gtpts).^2,2)) / ref;
    end
end
fprintf('done.\n');

% print accuracy
fprintf('pck@0.05:\n');
fprintf('%s',repmat(' ',[17,1]));
for i = 1:seq_len
    fprintf('ts %02d  ',i);
end
fprintf('\n');
fprintf('%s',repmat('-',[17+7*seq_len,1]));
fprintf('\n');
for a = 1:numel(list_act)
    fprintf('%15s  ',list_act{a})
    for i = 1:seq_len
        d = dist{i}(aid == a,:);
        d = d(~isnan(d));
        acc = 100 * mean(d < pck_th);
        fprintf('%4.1f %% ',acc);
    end
    fprintf('\n');
end
fprintf('%s',repmat('-',[17+7*seq_len,1]));
fprintf('\n');
fprintf('%15s  ','all');
for i = 1:seq_len
    d = dist{i}(~isnan(dist{i}));
    acc = 100 * mean(d < pck_th);
    fprintf('%4.1f %% ',acc);
end
fprintf('\n');
