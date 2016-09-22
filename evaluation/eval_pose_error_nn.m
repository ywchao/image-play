
exp_name = 'hg-256';

param.thres_vis = 13;

fig_on = false;

% set body joint config
param.pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
param.p_no = numel(param.pa);

% set vis params
param.msize = 4;
param.partcolor = {'g','g','g','r','b','r','b','y','y','m','c','m','c'};

% set output dir
vis_root = sprintf('./outputs/nn_thres%02d_%s/',param.thres_vis,exp_name);

% load training data
annot_file = './data/Penn_Action_cropped/train.h5';
anno_tr.ind2sub = permute(hdf5read(annot_file,'ind2sub'),[2 1]);
anno_tr.visible = permute(hdf5read(annot_file,'visible'),[2 1]);
anno_tr.part = permute(hdf5read(annot_file,'part'),[3 2 1]);

% process training data
tr.id = (1:size(anno_tr.ind2sub,1))';
tr.visible = anno_tr.visible;
[tr.part, tr.w] = normalize_part(anno_tr.part, anno_tr.visible);
% remove samples with limited visible joints
is_rm = tr.w < param.thres_vis;
tr.id(is_rm, :) = [];
tr.visible(is_rm, :) = [];
tr.part(is_rm, :, :) = [];
tr.w(is_rm) = [];
% assertions
assert(all(tr.part(repmat(tr.visible,[1 1 2]) == 0) == 0) == 1);
assert(all(~isnan(tr.part(:))) == 1);

% load valid data
annot_file = './data/Penn_Action_cropped/val.h5';
anno_va.ind2sub = permute(hdf5read(annot_file,'ind2sub'),[2 1]);
anno_va.visible = permute(hdf5read(annot_file,'visible'),[2 1]);
anno_va.part = permute(hdf5read(annot_file,'part'),[3 2 1]);

% set opt
opt.data = './data/Penn_Action_cropped';
opt.seqType = 'phase';
opt.nPhase = 16;
opt.seqLength = 1;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset_tr = penn_crop(opt, 'train');
dataset_va = penn_crop(opt, 'val');

% get sampled indices
sidx = dataset_va.getSampledIdx();

% load estimated pose
h5_pred = ['./exp/penn-crop/pose-val-' exp_name '/preds_val.h5'];
heatmaps = hdf5read(h5_pred,'heatmaps');
heatmaps = permute(heatmaps, [5 4 3 2 1]);
assert(size(heatmaps,1) == numel(sidx));

% load libs
libeval = ip_eval();
libimg = ip_img();

if fig_on
    figure(1);
end

pred_seqLength = 16;
acc = zeros(numel(sidx),pred_seqLength);

for i = 1:numel(sidx);
    tic_print(sprintf('  %04d/%04d\n',i,numel(sidx)));
    [sid, fid] = dataset_va.getSeqFrId(sidx(i));
    % set vis dir and file
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    if ~exist(vis_dir,'dir')
        mkdir(vis_dir);
    end
    % skip if all exist
    list_file = dir([vis_dir sprintf('%03d-',fid) '*']);
    list_file = {list_file.name}';
    if numel(list_file) == pred_seqLength
        fig_skip = true;
    else
        fig_skip = false;
    end
    
    if fig_on
        clf;
    end
    
    % get prediction
    hm = squeeze(heatmaps(i, opt.seqLength, :, :, :));
    hm(hm < 0) = 0;
    preds = libeval.getPreds(reshape(hm, [1 size(hm)]));
    preds = squeeze(preds);
    preds = preds * 4;
   
    % load input
    [input, seq, center, scale] = dataset_va.get(sidx(i));
    inp = input(1,:,:,:);
    inp = permute(inp,[3 4 2 1]);
    
    % plot predicted pose
    if fig_on && ~fig_skip
        subplot('Position',[0 0 1/4 1]); imshow(inp); hold on;
        for child = 2:param.p_no
            if mean(reshape(hm(param.pa(child),:,:),[opt.outputRes*opt.outputRes 1])) > 0.002 && ...
                    mean(reshape(hm(child,:,:),[opt.outputRes*opt.outputRes 1])) > 0.002
                x1 = preds(param.pa(child),1);
                y1 = preds(param.pa(child),2);
                x2 = preds(child,1);
                y2 = preds(child,2);
                
                plot(x2, y2, 'o', ...
                    'color', param.partcolor{child}, ...
                    'MarkerSize', param.msize, ...
                    'MarkerFaceColor', param.partcolor{child});
                plot(x1, y1, 'o', ...
                    'color', param.partcolor{child}, ...
                    'MarkerSize', param.msize, ...
                    'MarkerFaceColor', param.partcolor{child});
                line([x1 x2], [y1 y2], ...
                    'color', param.partcolor{child}, ...
                    'linewidth',round(param.msize/2));
            end
        end
        text(1,10,'Pred','Color','k','FontSize',10,'BackgroundColor','w');
    end
    
    % get gt (before normalized)
    gt_pa = anno_va.part(sidx(i),:,:);
    gt_pa = permute(gt_pa, [2 3 1]);
    gt_vis = anno_va.visible(sidx(i),:);
    gt_vis = permute(gt_vis, [2 1]);
    for j = 1:size(gt_pa,1)
        gt_pa(j,:) = libimg.transform(gt_pa(j,:), center, scale, 0, opt.inputRes);
    end
    
    % plot gt pose
    if fig_on && ~fig_skip
        subplot('Position',[1/4 0 1/4 1]); imshow(inp); hold on;
        for child = 2:param.p_no
            x1 = gt_pa(param.pa(child),1);
            y1 = gt_pa(param.pa(child),2);
            x2 = gt_pa(child,1);
            y2 = gt_pa(child,2);
            % skip invisible joints
            if gt_vis(child)
                plot(x2, y2, 'o', ...
                    'color', param.partcolor{child}, ...
                    'MarkerSize', param.msize, ...
                    'MarkerFaceColor', param.partcolor{child});
                if gt_vis(param.pa(child))
                    plot(x1, y1, 'o', ...
                        'color', param.partcolor{child}, ...
                        'MarkerSize', param.msize, ...
                        'MarkerFaceColor', param.partcolor{child});
                    line([x1 x2], [y1 y2], ...
                        'color', param.partcolor{child}, ...
                        'linewidth',round(param.msize/2));
                end
            end
        end
        
        % compute gt mse
        pa_ = permute(gt_pa,[3 1 2]);
        vs_ = permute(gt_vis,[2 1]);
        [gt_part, w] = normalize_part(pa_, vs_);
        if w >= param.thres_vis
            pa_ = permute(preds,[3 1 2]);
            pr_part = normalize_part(pa_, vs_);
            mse_gt = sum((pr_part(:) - gt_part(:)).^2) / sum(vs_);
        else
            mse_gt = NaN;
        end
        
        % display gt mse
        text(1,10,['GT: MSE ' num2str(mse_gt,'%.4f')],'Color','k','FontSize',10,'BackgroundColor','w');
    end
        
    % normalize pred pose
    pa_ = repmat(permute(preds,[3 1 2]),[numel(tr.id) 1 1]);
    [pr_part, ~, mu, sc] = normalize_part(pa_, tr.visible);
    assert(all(pr_part(repmat(tr.visible,[1 1 2]) == 0) == 0) == 1);
    assert(all(~isnan(pr_part(:))) == 1);
    
    % find nearest neighbor in training set and compute pred mse
    mse_all = sum(sum((pr_part - tr.part) .^ 2,3),2) ./ tr.w;
    [mse_sort, ii] = sort(mse_all,'ascend');
    mse_nn = mse_sort(1);
    ind_nn = ii(1);
    
    % load training image with nn pose
    [nn_input, nn_seq, nn_center, nn_scale] = dataset_tr.get(tr.id(ind_nn));
    nn_inp = nn_input(1,:,:,:);
    nn_inp = permute(nn_inp, [3 4 2 1]);
    nn_part = anno_tr.part(tr.id(ind_nn),:,:);
    nn_part = permute(nn_part, [2 3 1]);
    nn_vis = anno_tr.visible(tr.id(ind_nn),:);
    nn_vis = permute(nn_vis, [2 1]);
    for j = 1:size(nn_part,1)
        nn_part(j,:) = libimg.transform(nn_part(j,:), nn_center, nn_scale, 0, opt.inputRes);
    end
    
    % plot nn pose on training image
    if fig_on && ~fig_skip
        subplot('Position',[2/4 0 1/4 1]); imshow(nn_inp); hold on;
        for child = 2:param.p_no
            x1 = nn_part(param.pa(child),1);
            y1 = nn_part(param.pa(child),2);
            x2 = nn_part(child,1);
            y2 = nn_part(child,2);
            % skip invisible joints
            if nn_vis(child)
                plot(x2, y2, 'o', ...
                    'color', param.partcolor{child}, ...
                    'MarkerSize', param.msize, ...
                    'MarkerFaceColor', param.partcolor{child});
                if nn_vis(param.pa(child))
                    plot(x1, y1, 'o', ...
                        'color', param.partcolor{child}, ...
                        'MarkerSize', param.msize, ...
                        'MarkerFaceColor', param.partcolor{child});
                    line([x1 x2], [y1 y2], ...
                        'color', param.partcolor{child}, ...
                        'linewidth',round(param.msize/2));
                end
            end
        end
        
        % display nn mse
        text(1,10,['NN: MSE ' num2str(mse_nn,'%.4f')],'Color','k','FontSize',10,'BackgroundColor','w');
    end
    
    % transform nn pose for input
    for j = 1:pred_seqLength
        tf_part_ = anno_tr.part(nn_seq(j),:,:);
        tf_vis = anno_tr.visible(nn_seq(j),:);
        % normalize
        if j == 1
            [tf_part, tf_w, tf_mu, tf_sc] = normalize_part(tf_part_, tf_vis);
            assert(tf_w >= param.thres_vis);
            assert(numel(find(tr.id == nn_seq(j))) == 1);
            assert(all(tf_part(:) == reshape(tr.part(ind_nn,:,:),[numel(tr.part(ind_nn,:,:)), 1])) == 1);
            assert(all(tf_vis == tr.visible(ind_nn,:,:)) == 1);
        else
            tf_part = tf_part_ - repmat(tf_mu, [1 size(tf_part_,2) 1]);
            tf_part = tf_part .* repmat(tf_vis, [1 1 2]);
            tf_part = tf_part ./ repmat(tf_sc, [1 size(tf_part,2) 2]);
        end
        % de-normalize to input scale
        if tf_w > 1
            assert(all(~isnan(tf_part(:))) == 1);
            tf_part = tf_part .* repmat(sc(ind_nn), [1 size(tf_part,2) 2]);
            tf_part = tf_part + repmat(mu(ind_nn,:,:), [1 size(tf_part,2) 1]);
            tf_part = permute(tf_part, [2 3 1]);
            tf_vis = permute(tf_vis, [2 1]);
            
            % compute accuracy
            % modified from train.lua; assume batch size == 1 here
            k = 1;
            gt_pa = anno_va.part(seq(j),:,:);
            gt_pa = permute(gt_pa, [2 3 1]);
            for p = 1:size(gt_pa,1)
                gt_pa(p,:) = libimg.transform(gt_pa(p,:)+1, center, scale, 0, opt.outputRes);
            end
            if any(reshape(gt_pa(k,:,:),[numel(gt_pa(k,:,:)) 1]) ~= 0)
                % hm = heatmaps(i, j, :, :, :);
                % hm = permute(hm, [1 3 4 5 2]);
                pr_pa_ = round(tf_part / 4);
                pr_pa_ = permute(pr_pa_, [3 1 2]);
                if j == 1
                    assert(seq(j) == sidx(i));
                end
                gt_pa_ = permute(gt_pa, [3 1 2]);
                % acc_ = libeval.heatmapAccuracy(hm, gt_pa_, [], [], opt.outputRes);
                acc_ = libeval.heatmapAccuracy(pr_pa_, gt_pa_, [], [], opt.outputRes);
                acc(i,j) = acc_(1);
            else
                acc(i,j) = NaN;
            end
        else
            error('tf_w == 1, this should be handled separately.');
        end
    
        if exist('h','var')
            delete(h);
        end
        
        % plot transformed nn pose on input
        if fig_on && ~fig_skip
            h = subplot('Position',[3/4 0 1/4 1]); imshow(inp); hold on;
            for child = 2:param.p_no
                x1 = tf_part(param.pa(child),1);
                y1 = tf_part(param.pa(child),2);
                x2 = tf_part(child,1);
                y2 = tf_part(child,2);
                % skip invisible joints
                if tf_vis(child)
                    plot(x2, y2, 'o', ...
                        'color', param.partcolor{child}, ...
                        'MarkerSize', param.msize, ...
                        'MarkerFaceColor', param.partcolor{child});
                    if tf_vis(param.pa(child))
                        plot(x1, y1, 'o', ...
                            'color', param.partcolor{child}, ...
                            'MarkerSize', param.msize, ...
                            'MarkerFaceColor', param.partcolor{child});
                        line([x1 x2], [y1 y2], ...
                            'color', param.partcolor{child}, ...
                            'linewidth',round(param.msize/2));
                    end
                end
            end
            
            % set figure
            set(gcf,'Position',[0 0 opt.inputRes*4 opt.inputRes]);
            set(gcf,'PaperPositionMode','auto');
            set(gcf,'color',[1 1 1]);
            set(gca,'color',[1 1 1]);
            
            % save figure
            vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];
            print(gcf,vis_file,'-dpng','-r150');
        end
    end
end

if fig_on && ~fig_skip
    close;
end

macc = zeros(pred_seqLength,1);
for i = 1:pred_seqLength
    acc_ = acc(:,i);
    macc(i) = mean(acc_(~isnan(acc_)));
end
