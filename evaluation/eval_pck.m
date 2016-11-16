
config;

% exp_name = 'seq16-hg-256-res-clstm-base16';                       clr = 'r';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-base16-w1e-7';          clr = 'b';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-fts3-hg-base16-w1e-7';  clr = 'g';  opt.seqLength = 16;

% exp_name = 'seq16-hg-256-res-clstm';                              clr = 'r';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-w1e-5';                 clr = 'b';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-w1e-6';                 clr = 'b';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-w1e-7';                 clr = 'b';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-w1e-8';                 clr = 'b';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-fts3-hg-w1e-5';         clr = 'g';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-fts3-hg-w1e-6';         clr = 'g';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-fts3-hg-w1e-7';         clr = 'g';  opt.seqLength = 16;
% exp_name = 'seq16-hg-256-res-clstm-res-64-fts3-hg-w1e-8';         clr = 'g';  opt.seqLength = 16;

% exp_name = 'nn-skel-hg-256-th09';                                 clr = 'g';  opt.seqLength = 16;
% exp_name = 'nn-skel-caffenet-hg-256-th09-K22500';                 clr = 'r';  opt.seqLength = 16;
% exp_name = 'nn-skel-oracle-hg-256-th09';                          clr = 'k';  opt.seqLength = 16;

% exp_name = 'pose-est-hg-256';                                     clr = 'c';  opt.seqLength = 1;

% split = 'val';
% split = 'test';

% set opt and init dataset
% THIS SHOULD BE REMOVED LATER. THE EVALUATION SCRIPT ONLY NEED TO THE
% LABEL FILES AND A LIST OF ID SEQUENCES
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% % load action label
% list_seq = dir('./data/Penn_Action_cropped/labels/*.mat');
% list_seq = {list_seq.name}';
% num_seq = numel(list_seq);
% action = cell(num_seq,1);
% for i = 1:num_seq
%     lb_file = ['./data/Penn_Action_cropped/labels/' list_seq{i}];
%     anno = load(lb_file);
%     assert(ischar(anno.action));
%     action{i} = anno.action;
% end
% [list_act,~,ia] = unique(action, 'stable');
% aid = zeros(dataset.size(),1);
% for i = 1:dataset.size()
%     aid(i) = ia(dataset.getSeqFrId(i));
% end

dist_root = 'evaluation/cache_dist/';
dist_file = [dist_root exp_name '_' split '.mat'];
makedir(dist_root);

if exist(dist_file,'file')
    load(dist_file);
    fprintf('dist loaded.\n');
else
    % init dist
    num_pt = 13;
    dist = cell(opt.seqLength,1);
    for i = 1:opt.seqLength
        dist{i} = zeros(dataset.size(),num_pt);
    end
    fprintf('computing dist ... \n');
    for i = 1:dataset.size();
        tic_print(sprintf('  %05d/%05d\n',i,dataset.size()));
        % load annotation
        sid = dataset.getSeqFrId(i);
        lb_file = [lbdata_root num2str(sid,'%04d') '.mat'];
        anno = load(lb_file);
        % get sequence id
        seqIdx = dataset.getSeq(i);
        assert(numel(seqIdx) == opt.seqLength);
        % load prediction
        pred_file = sprintf('./exp/penn-crop/%s/eval_%s/%05d.mat',exp_name,split,i);
        pred = load(pred_file);
        pred = pred.eval;
        % get ref
        ref = max(anno.dimensions(1:2));
        % compute distance
        for t = 1:opt.seqLength
            ind = seqIdx(t);
            [~, fid] = dataset.getSeqFrId(ind);
            gtpts = [anno.x(fid,:)' anno.y(fid,:)'];
            gtpts(anno.visibility(fid,:) == 0,:) = NaN;
            % c1 = gtpts(:,1) < 1;
            % c2 = gtpts(:,2) < 1;
            % c3 = gtpts(:,1) > anno.dimensions(1);
            % c4 = gtpts(:,1) > anno.dimensions(2);
            % gtpts(c1|c2|c3|c4,:) = NaN;
            dist{t}(i,:) = sqrt(sum((squeeze(pred(t,:,:)) - gtpts).^2,2)) / ref;
        end
    end
    fprintf('done.\n');
    % save dist
    save(dist_file,'dist');
end

% print acc
th = 0.05;
for i = 1:opt.seqLength
    d = dist{i}(~isnan(dist{i}));
    d = d(:);
    err = mean(d(~isinf(d)));
    acc = 100 * mean(d < th);
    fprintf('fr %02d  err: %7.5f  acc: %5.2f %%\n',i,err,acc);
end

range = 0:0.01:0.1;

err = zeros(opt.seqLength,1);
acc = zeros(opt.seqLength,numel(range));

for i = 1:opt.seqLength
    % dist{i} = dist{i}(:,1);        % head
    % dist{i} = dist{i}(:,[ 2  3]);  % shoulder
    % dist{i} = dist{i}(:,[ 4  5]);  % elbow
    % dist{i} = dist{i}(:,[ 6  7]);  % wrist
    % dist{i} = dist{i}(:,[ 8  9]);  % hip
    % dist{i} = dist{i}(:,[10 11]);  % knee
    % dist{i} = dist{i}(:,[12 13]);  % ankle

    % act = 7;  % jump_rope
    % act = 8;  % jumping_jacks
    % dist{i} = dist{i}(aid == act,:);
    
    % mean over all joints
    d = dist{i}(~isnan(dist{i}));
    err(i) = mean(d);
    for j = 1:numel(range)
        acc(i,j) = 1 * mean(d < range(j));
    end
    
    % % mean of per image mean
    % d = cell(size(dist{i},1),1);
    % for k = 1:size(dist{i})
    %     d{k} = dist{i}(k,~isnan(dist{i}(k,:)));
    % end
    % for j = 1:numel(range)
    %     m = 0;
    %     n = 0;
    %     for k = 1:size(d,1);
    %         if ~isempty(d{k})
    %             m = m + 1 * mean(d{k} < range(j));
    %             n = n + 1;
    %         end
    %     end
    %     acc(i,j) = m/n;
    % end
end
sum(acc(:))

% figure(act);

set(gcf,'Position',[2 26 700 700]);
% set(gcf,'Position',[2 26 1468 1468]);

for i = 1:opt.seqLength
    row = ceil(i/4);
    col = mod(i-1,4)+1;
    subplot('Position',[(col-1)/4+0.02 (-row+4)/4+0.025 1/4-0.03 1/4-0.055]);
    set(gca,'fontsize',6);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
    title(sprintf('t = %2d',i));
end

% set(gcf,'PaperPositionMode','auto');
% print(gcf,'outputs/pck.pdf','-dpdf');