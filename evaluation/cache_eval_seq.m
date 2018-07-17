% cache eval sequences to the following files:
%   evaluation/seq_train.txt
%   evaluation/seq_val.txt
%   evaluation/seq_test.txt

fprintf('caching eval sequences ... \n');

opt.data = './data/penn-crop';
opt.nPhase = 16;
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

sp = {'train','val','test'};
for s = 1:numel(sp)
    seq_file = ['./evaluation/seq_' sp{s} '.txt'];
    if exist(seq_file,'file')
        continue
    end
    dataset = penn_crop(opt, sp{s});
    eval_seq = zeros(dataset.size(),opt.seqLength+1);
    for i = 1:dataset.size()
        sid = dataset.getSeqFrId(i);
        seq = dataset.getSeq(i);
        [~, fid] = dataset.getSeqFrId(seq);
        eval_seq(i,:) = [sid fid'];
    end
    dlmwrite(seq_file,eval_seq,'delimiter',' ');
end

fprintf('done.\n');
