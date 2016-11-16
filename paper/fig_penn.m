
% split = 'test';   ind = 8027;
% split = 'train';  ind = 13231;
% split = 'test';   ind = 33702;
% split = 'train';  ind = 41530;
% split = 'test';   ind = 40997;
% split = 'test';   ind = 72758;

% set save root
save_root = './outputs/figures/penn_crop_sample/';
makedir(save_root);

% set opt and init dataset
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

sidx = 1:dataset.size();
sid = dataset.getSeqFrId(sidx);
run = sidx(ismember(sid,2178));

seq = dataset.getSeq(ind);

[sid, fid] = dataset.getSeqFrId(seq(1));

for i = 1:numel(seq)
    [sid_i, fid_i] = dataset.getSeqFrId(seq(i));
    save_file = [save_root sprintf('%04d-%03d-%02d.pdf',sid,fid,i)];
    if ~exist(save_file,'dir')
        im_file = ['data/Penn_Action_cropped/frames/' num2str(sid_i,'%04d') '/' num2str(fid_i,'%06d') '.jpg'];
        im = imread(im_file);
        clf;
        imshow(im);
        set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
        set(gca,'Position',[0.00 0.00 1.00 1.00]);
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
    end
end

close;