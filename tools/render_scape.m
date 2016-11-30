
expID = 'seq16-hg-256-res-clstm-res-64-w1e-6';  mode = 1;

% split = 'val';
split = 'test';

% set parameters
interval = 1;
factor = 1.1;

% texture = 0;
% texture = 1;
% texture = 2;

% set vis root
vis_root = sprintf('./outputs/render_scape_penn/%s/texture_%1d/%s/', expID, texture, split);
makedir(vis_root);

% set opt and init dataset
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;
dataset = penn_crop(opt, split);

% get video ids for visualization
sidx = dataset.getSampledIdx();
sidx = sidx(1:interval:numel(sidx));

% visualize first min(K,len) videos for each action
K = 3;
extra = [787 1300 268 2027 295 2308 2247 1444];
list_seq = dir('./data/Penn_Action_cropped/labels/*.mat');
list_seq = {list_seq.name}';
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/Penn_Action_cropped/labels/' list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
sid = dataset.getSeqFrId(sidx);
seq = unique(sid);
keep = false(numel(seq),1);
for i = 1:numel(list_act)
    ii = find(ismember(seq,find(ia == i)));  
    keep(ii(1:min(numel(ii),K))) = true;
end
keep(ismember(seq,extra)) = true;
seq = seq(keep);
run = sidx(ismember(sid,seq));

% prepare for scape
addpath('./Deep3DPose/prepare/mesh/skel');
addpath('./Deep3DPose/prepare/mesh/quatern');
addpath('./Deep3DPose/2-model/io');
addpath(genpath('./Deep3DPose/prepare/scape/MATLAB_daz_m_srf'));
skel_scape = load('./Deep3DPose/prepare/mesh/data/tpose.txt');
load('./Deep3DPose/prepare/mesh3/data/arm_75_view_0');
Meta.instance.readA;
Meta.instance.readPCA;
weights = Meta.instance.weight;
shapepara = Meta.instance.sem_default;

% mtl file
if texture ~= 0
    switch texture
        case 1
            text_file_1 = '../../../../../../Deep3DPose/data/textures/2.png';
            text_file_2 = '../../../../../../Deep3DPose/data/textures2/1.png';
        case 2
            text_file_1 = '../../../../../../Deep3DPose/prepare/mesh3/data/1.png';
            text_file_2 = '../../../../../../Deep3DPose/prepare/mesh3/data/1.png';
    end
    mtl_file = [vis_root 'all.mtl'];
    if ~exist(mtl_file,'file')
        fid = fopen(mtl_file, 'w');
        fprintf(fid, 'newmtl Material\n');
        fprintf(fid, sprintf('map_Kd  %s\n',text_file_1));
        fprintf(fid, 'newmtl Material2\n');
        fprintf(fid, sprintf('map_Kd  %s\n',text_file_2));
    end
end

% prepare for rendering
blender_path = '/z/ywchao/tools/blender-2.78a-linux-glibc211-x86_64/blender';
blank_file = '/z/ywchao/codes/image-play/Deep3DPose/4-render/blank.blend';
render_file = '/z/ywchao/codes/image-play/Deep3DPose/4-render/render_model_views_ip.py';
view_file = [vis_root 'view.txt'];

% view file
a = 0;
b = 90;
c = -90;
d = 2.15;
if ~exist(view_file,'file')
    fid = fopen(view_file, 'w');
    fprintf(fid, '%f %f %f %f\n', a,b,c,d);
    fclose(fid);
end

fprintf('render scape ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    [sid, fid] = dataset.getSeqFrId(i);
    % make directories
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    makedir(vis_dir);
    % load 3d predictions
    pred_file = sprintf('./exp/penn-crop/%s/pred_%s/%05d.mat',expID,split,i);
    preds = load(pred_file);
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    repos = zeros(opt.seqLength,17,3);
    repos(:,joints,:) = preds.repos;
    repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
    repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
    repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
    repos(:,11,:) = preds.repos(:,1,:);
    % convert to deep 3d format
    joints = [10 9 12 13 14 15 16 17 1 5 6 7 2 3 4];
    skel = repos(:,joints,:);
    skel = skel / 56.444;
    skel(:,:,2) = -skel(:,:,2);
    skel(:,:,3) = -skel(:,:,3);
    % load 2d predictions for getting bbox
    eval_file = sprintf('./exp/penn-crop/%s/eval_%s/%05d.mat',expID,split,i);
    preds = load(eval_file);
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred2 = zeros(opt.seqLength,17,2);
    pred2(:,joints,:) = preds.eval;
    pred2(:,1,:) = (preds.eval(:,8,:) + preds.eval(:,9,:))/2;
    pred2(:,8,:) = (preds.eval(:,2,:) + preds.eval(:,3,:) + preds.eval(:,8,:) + preds.eval(:,9,:))/4;
    pred2(:,9,:) = (preds.eval(:,1,:) + preds.eval(:,2,:) + preds.eval(:,3,:))/3;
    pred2(:,11,:) = preds.eval(:,1,:);
    % get bbox
    x1_a = min(pred2(:,:,1),[],2);
    x2_a = max(pred2(:,:,1),[],2);
    y1_a = min(pred2(:,:,2),[],2);
    y2_a = max(pred2(:,:,2),[],2);
    h = (y2_a-y1_a) * (factor-1);
    w = (x2_a-x1_a) * (factor-1);
    x1_a = round(x1_a - w/2);
    y1_a = round(y1_a - h/2);
    x2_a = round(x2_a + w/2);
    y2_a = round(y2_a + h/2);
    for j = 1:opt.seqLength
        % skip if figure exists
        obj_file = [vis_dir sprintf('%03d-%02d.obj',fid,j)];
        vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];
        if exist(obj_file,'file') && exist(vis_file,'file')
            continue
        end
        % get RR
        jointsRR = zeros(3, 3, 16);
        [RR, R] = skel2RR(squeeze(skel(j,:,:)), skel_scape);
        jointsRR(:, :, 1:15) = RR;
        jointsRR(:, :, 16) = R;
        % move head
        head = jointsRR(:, :, 3);
        head = matrix2quaternion(head);
        matrix = q2matrix(head);
        matrix = matrix';
        jointsRR(:, :, 3) = matrix*jointsRR(:, :, 3);
        % generate points
        RR = jointsRR(:, :, 1:15);
        R = jointsRR(:, :, 16);
        body = Body(RR, shapepara);
        points = body.points;
        p = R'*points';
        % p = points';
        p = 0.5*p;
        points = p';
        points = moveToCenter(weights, points, 2);
        p = points';
        % obj
        file_id = fopen(obj_file, 'w');
        if texture == 0
            fprintf(file_id, 'mtllib dummy.mtl\n');
        else
            fprintf(file_id, 'mtllib ../all.mtl\n');
        end
        for k = 1:6449
            fprintf(file_id, 'v %f %f %f\n', p(:,k));
        end
        fprintf(file_id, '%s', restfiles);
        fclose(file_id);
        % render
        command = sprintf('%s %s --background --python %s -- %s %s %s',...
            blender_path, blank_file, render_file, obj_file, vis_dir, view_file);
        system(command);
        % get filenames
        raw_file = sprintf('%s/image_a%03d_e%03d_t%03d_d%03d.png',vis_dir,round(a),round(b),round(c),round(d));
        log_file = sprintf('%s/image_a%03d_e%03d_t%03d_d%03d_cam.txt',vis_dir,round(a),round(b),round(c),round(d));
        % load rendered image
        [im_fg, ~, ma_fg] = imread(raw_file);
        im_fg = double(im_fg);
        ma_fg = double(ma_fg);
        % crop rendered image
        bgColor = 0;
        [nr, nc] = size(ma_fg);
        colsum = sum(ma_fg == bgColor, 1) ~= nr;
        rowsum = sum(ma_fg == bgColor, 2) ~= nc;
        ll = find(colsum, 1, 'first');
        rr = find(colsum, 1, 'last');
        tt = find(rowsum, 1, 'first');
        bb = find(rowsum, 1, 'last');
        ma_fg = ma_fg(tt:bb, ll:rr);
        im_fg = im_fg(tt:bb, ll:rr, :);
        % load background image
        bgd_file = sprintf('./data/Penn_Action_cropped/frames/%04d/%06d.jpg',sid,fid);
        im_bg = imread(bgd_file);
        im_bg = double(im_bg);
        % get bbox
        x1 = max(x1_a(j),1);
        y1 = max(y1_a(j),1);
        x2 = min(x2_a(j),size(im_bg,2));
        y2 = min(y2_a(j),size(im_bg,1));
        % resize foreground image
        im_fg = imresize(im_fg,[y2-y1+1,x2-x1+1]);
        ma_fg = imresize(ma_fg,[y2-y1+1,x2-x1+1]);
        ma_fg = min(ma_fg,1);
        ma_bg = 1 - ma_fg;
        im_cb = uint8(im_bg);
        for k = 1:3
            fg = uint8(ma_fg .* im_fg(:,:,k));
            bg = uint8(ma_bg .* im_bg(y1:y2,x1:x2,k));
            im_cb(y1:y2,x1:x2,k) = fg + bg;
        end
        imwrite(im_cb,vis_file,'png');
        % clean up
        % movefile(raw_file,vis_file);
        delete(raw_file);
        delete(log_file);
    end
end
fprintf('done.\n');

close;