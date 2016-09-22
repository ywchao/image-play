
% exp_name = 'seq16-hg-256-res-clstm-nl2';

% load valid data
annot_file = './data/Penn_Action_cropped/val.h5';
anno_va.ind2sub = permute(hdf5read(annot_file,'ind2sub'),[2 1]);
anno_va.visible = permute(hdf5read(annot_file,'visible'),[2 1]);
anno_va.part = permute(hdf5read(annot_file,'part'),[3 2 1]);

% set opt
opt.data = './data/Penn_Action_cropped';
opt.seqType = 'phase';
opt.nPhase = 16;
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset_va = penn_crop(opt, 'val');

% get sampled indices
sidx = dataset_va.getSampledIdx();

% load estimated pose
h5_pred = ['./exp/penn-crop/' exp_name '/preds_val.h5'];
heatmaps = hdf5read(h5_pred,'heatmaps');
heatmaps = permute(heatmaps, [5 4 3 2 1]);
assert(size(heatmaps,1) == numel(sidx));

% load libs
libeval = ip_eval();
libimg = ip_img();

acc = zeros(numel(sidx),opt.seqLength);

for i = 1:numel(sidx);
    tic_print(sprintf('  %04d/%04d\n',i,numel(sidx)));
    [sid, fid] = dataset_va.getSeqFrId(sidx(i));
    
    % load input
    [~, seq, center, scale] = dataset_va.get(sidx(i));
    
    % compute accuracy
    for j = 1:opt.seqLength
        % modified from train.lua; assume batch size == 1 here
        k = 1;
        gt_pa = anno_va.part(seq(j),:,:);
        gt_pa = permute(gt_pa, [2 3 1]);
        for p = 1:size(gt_pa,1)
            gt_pa(p,:) = libimg.transform(gt_pa(p,:)+1, center, scale, 0, opt.outputRes);
        end            
        if any(reshape(gt_pa(k,:,:),[numel(gt_pa(k,:,:)) 1]) ~= 0)
            hm = heatmaps(i, j, :, :, :);
            hm = permute(hm, [1 3 4 5 2]);
            gt_pa_ = permute(gt_pa, [3 1 2]);
            acc_ = libeval.heatmapAccuracy(hm, gt_pa_, [], [], opt.outputRes);
            acc(i,j) = acc_(1);
        else
            acc(i,j) = NaN;
        end
    end
end

macc = zeros(opt.seqLength,1);
for i = 1:opt.seqLength
    acc_ = acc(:,i);
    macc(i) = mean(acc_(~isnan(acc_)));
end
