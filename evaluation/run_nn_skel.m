
exp_name = 'hg-256';

% split = 'val';
% split = 'test';

% set visibility threshold for training data
% selected by maximizing validation accuracy
param.thres_vis = 9;

% set save directory
save_dir = sprintf('exp/penn-crop/nn-skel-%s-th%02d/eval_%s/',exp_name,param.thres_vis,split);
makedir(save_dir);

% load training data
annot_file = './data/Penn_Action_cropped/train.h5';
anno_tr.ind2sub = permute(hdf5read(annot_file,'ind2sub'),[2 1]);
anno_tr.part = permute(hdf5read(annot_file,'part'),[3 2 1]);

% process training data
tr.id = (1:size(anno_tr.ind2sub,1))';
tr.visible = anno_tr.part(:,:,1) ~= 0 & anno_tr.part(:,:,2) ~= 0;
[tr.part, tr.c] = normalize_part(anno_tr.part, tr.visible);

% remove samples with limited visible joints
is_rm = tr.c < param.thres_vis;
tr.id(is_rm, :) = [];
tr.visible(is_rm, :) = [];
tr.part(is_rm, :, :) = [];
tr.c(is_rm) = [];
assert(all(tr.part(repmat(tr.visible,[1 1 2]) == 0) == 0) == 1);

% set opt
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset_tr = penn_crop(opt, 'train');
dataset_ts = penn_crop(opt, split);

fprintf('running nn skel ... \n');
for i = 1:dataset_ts.size()
    tic_print(sprintf('  %04d/%04d\n',i,dataset_ts.size()));
    
    % skip if exists
    save_file = [save_dir num2str(i,'%05d') '.mat'];
    if exist(save_file,'file')
        continue
    end
    
    % load estimated pose
    pred_file = sprintf('./exp/penn-crop/pose-est-%s/eval_%s/%05d.mat',exp_name,split,i);
    pred = load(pred_file);
    pred = pred.eval;
        
    % normalize pred pose
    pa_ = repmat(pred,[numel(tr.id) 1 1]);
    [pr_part, ~, mu, sc] = normalize_part(pa_, tr.visible);
    assert(all(pr_part(repmat(tr.visible,[1 1 2]) == 0) == 0) == 1);
    
    % find nearest neighbor in training set and compute pred mse
    mse_all = sum(sum((pr_part - tr.part) .^ 2,3),2) ./ tr.c;
    [mse, ind_nn] = min(mse_all);
    
    % load training image with nn pose
    nn_seq = dataset_tr.getSeq(tr.id(ind_nn));
    
    % transform nn pose for test image
    eval = zeros(opt.seqLength, 13, 2);
    for j = 1:opt.seqLength
        tf_part_ = anno_tr.part(nn_seq(j),:,:);
        tf_vis = tf_part_(:,:,1) ~= 0 & tf_part_(:,:,2) ~= 0;
        % normalize
        if j == 1
            [tf_part, tf_c, tf_mu, tf_sc] = normalize_part(tf_part_, tf_vis);
            assert(tf_c >= param.thres_vis);
            assert(numel(find(tr.id == nn_seq(j))) == 1);
            assert(all(tf_part(:) == reshape(tr.part(ind_nn,:,:),[numel(tr.part(ind_nn,:,:)), 1])) == 1);
            assert(all(tf_vis == tr.visible(ind_nn,:,:)) == 1);
        else
            tf_part = tf_part_ - repmat(tf_mu, [1 size(tf_part_,2) 1]);
            tf_part = tf_part .* repmat(tf_vis, [1 1 2]);
            tf_part = tf_part ./ repmat(tf_sc, [1 size(tf_part,2) 2]);
        end
        % de-normalize to test image scale
        if tf_c == 1
            error('tf_c == 1, this should be handled separately.');
        end
        tf_part = tf_part .* repmat(sc(ind_nn), [1 size(tf_part,2) 2]);
        tf_part = tf_part + repmat(mu(ind_nn,:,:), [1 size(tf_part,2) 1]);
        % missing annotation should be treated as mistakes during classification
        tf_part(repmat(tf_vis, [1 1 2]) == 0) = Inf;
        % save to eval
        eval(j,:,:) = tf_part;
    end
    
    % save pred
    sid = anno_tr.ind2sub(tr.id(ind_nn),1);
    fid = anno_tr.ind2sub(tr.id(ind_nn),2);
    save(save_file,'eval','sid','fid','mse');
end
fprintf('done.\n');