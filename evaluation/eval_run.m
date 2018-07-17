
% split = 'val';
split = 'test';

% cache eval sequences
cache_eval_seq;

% load eval sequences
fprintf('loading eval sequences ... \n');
seq_file = ['./evaluation/seq_' split '.txt'];
eval_seq = dlmread(seq_file);
fprintf('done.\n');

% load action labels (for per action eval)
fprintf('loading action labels ...\n');
[list_seq,~,ii] = unique(eval_seq(:,1));
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/penn-crop/labels/' num2str(list_seq(i),'%04d') '.mat'];
    anno = load(lb_file);
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
aid = ia(ii);
fprintf('done.\n');

% create var to keep all dist
dist_all = cell(4,1);

% print pck@0.05 for hourglass [19]
fprintf('\n');
fprintf('exp: hourglass [19]\n');
pred_dir = ['./exp/penn-crop/hg-256/eval_' split];
if exist(pred_dir,'dir')
    seq_len = 1;
    eval_one;  % eval routine
    dist_all{1} = dist;
else
    fprintf('no result found.\n');
end

% print pck@0.05 for nn-all
fprintf('\n');
fprintf('exp: nn-all\n');
pred_dir = ['./exp/penn-crop/nn-all-th09/eval_' split];
if exist(pred_dir,'dir')
    seq_len = 16;
    eval_one;  % eval routine
    dist_all{2} = dist;
else
    fprintf('no result found.\n');
end

% print pck@0.05 for nn-oracle
fprintf('\n');
fprintf('exp: nn-oracle\n');
pred_dir = ['./exp/penn-crop/nn-oracle-th09/eval_' split];
if exist(pred_dir,'dir')
    seq_len = 16;
    eval_one;  % eval routine
    dist_all{3} = dist;
else
    fprintf('no result found.\n');
end

% print pck@0.05 for 3d-pfnet
fprintf('\n');
fprintf('exp: 3d-pfnet\n');
pred_dir = ['./exp/penn-crop/hg-256-res-clstm-res-64/eval_' split];
if exist(pred_dir,'dir')
    seq_len = 16;
    eval_one;  % eval routine
    dist_all{4} = dist;
else
    fprintf('no result found.\n');
end

% plot pck curves
pck_range = 0:0.01:0.1;
figure(1);
set(gcf,'Position',[2 26 700 700]);
leg = {'Hourglass [19]','NN-all','NN-oracle','3D-PFNet'};
clr = {'c','r','k','b'};
ind = find(cellfun(@(x)~isempty(x),dist_all))';
for s = ind
    dist = dist_all{s};
    for i = 1:min(numel(dist),16)
        d = dist{i}(~isnan(dist{i}));
        acc = zeros(numel(pck_range),1);
        for j = 1:numel(pck_range)
            acc(j) = 1 * mean(d < pck_range(j));
        end
        row = ceil(i/4);
        col = mod(i-1,4)+1;
        subplot('Position',[(col-1)/4+0.025 (-row+4)/4+0.025 1/4-0.035 1/4-0.055]);
        set(gca,'fontsize',6);
        plot(pck_range,acc,'color',clr{s},'LineStyle','-','LineWidth',1);
        hold on;
        grid on;
        axis([0 0.1 0 1]);
        set(gca,'xtick',0:0.02:0.1);
        title(sprintf('t = %2d',i));
        if i == 1 && s == ind(end)
            hl = legend(leg(ind));
            set(hl,'Location','southeast');
            set(hl,'FontSize',6);
        end
    end
end
