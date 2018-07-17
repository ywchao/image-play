
% set directories
exp_name = sprintf('nn-%s-th%02d',mode,thres_vis);
curr_dir = sprintf('exp/penn-crop/hg-256/eval_%s/',split);
pred_dir = sprintf('exp/penn-crop/%s/eval_%s/',exp_name,split);
png_root = sprintf('outputs/vis_%s/%s_png/',exp_name,split);
gif_root = sprintf('outputs/vis_%s/%s_gif/',exp_name,split);

% set body joint config
param.pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
param.p_no = numel(param.pa);

% set vis params
param.msize = 4;
param.partcolor = {'g','g','g','r','b','r','b','y','y','m','c','m','c'};

% set opt
opt.data = './data/penn-crop';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset = penn_crop(opt, split);

% get sampled indices
interval = 2;
sidx = dataset.getSampledIdx();
sidx = sidx(1:interval:numel(sidx));

% limit to first min(K,len) videos for each action
K = 3;
list_seq = dir('./data/penn-crop/labels/*.mat');
list_seq = {list_seq.name}';
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/penn-crop/labels/' list_seq{i}];
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
seq = seq(keep);
run = sidx(ismember(sid,seq));

% init figure
figure(1);
set(gcf,'Position',[0 0 1024 256]);
clear hi hg hn hp

fprintf('visualizing nn ... \n');
for i = run
    tic_print(sprintf('  %05d/%05d\n',find(i == run),numel(run)))
    [sid, fid] = dataset.getSeqFrId(i);
    
    % make directories
    png_dir = [png_root num2str(sid,'%04d') '/'];
    gif_dir = [gif_root num2str(sid,'%04d') '/'];
    makedir(png_dir);
    makedir(gif_dir);
    
    % skip if gif exist
    gif_file = fullfile(gif_dir,sprintf('%03d.gif',fid));
    if exist(gif_file,'file')
        continue
    end

    % load current frame estimation
    curr_file = sprintf('%s%05d.mat',curr_dir,i);
    curr = load(curr_file);
    curr = squeeze(curr.eval);
    
    % load prediction
    pred_file = sprintf('%s%05d.mat',pred_dir,i);
    pred = load(pred_file);
    nn_sid = pred.sid;
    nn_fid = pred.fid;
    nn_mse = pred.mse;
    pred = pred.eval;

    % load input im
    im_file = sprintf('./data/penn-crop/frames/%04d/%06d.jpg',sid,fid);
    im = imread(im_file);
    % plot predicted pose
    if exist('hi','var')
        delete(hi);
    end
    hi = subplot('Position',[0 0 1/4 1]); imshow(im); hold on;
    for child = 2:param.p_no
        x1 = curr(param.pa(child),1);
        y1 = curr(param.pa(child),2);
        x2 = curr(child,1);
        y2 = curr(child,2);

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
    text(1,10,'Pred','Color','k','FontSize',10,'BackgroundColor','w');

    % load gt pose
    lb_file = sprintf('./data/penn-crop/labels/%04d.mat',sid);
    anno = load(lb_file);
    % plot gt pose
    if exist('hg','var')
        delete(hg);
    end
    hg = subplot('Position',[1/4 0 1/4 1]); imshow(im); hold on;
    for child = 2:param.p_no
        x1 = anno.x(fid,param.pa(child));
        y1 = anno.y(fid,param.pa(child));
        x2 = anno.x(fid,child);
        y2 = anno.y(fid,child);
        % skip invisible joints
        if anno.visibility(fid,child)
            plot(x2, y2, 'o', ...
                'color', param.partcolor{child}, ...
                'MarkerSize', param.msize, ...
                'MarkerFaceColor', param.partcolor{child});
            if anno.visibility(fid,param.pa(child))
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
    % compute gt pose mse
    gt_part = permute([anno.x(fid,:)' anno.y(fid,:)'],[3 1 2]);
    gt_vis = anno.visibility(fid,:);
    [gt_part, gt_c] = normalize_part(gt_part, gt_vis);
    if gt_c >= thres_vis
        cu_part = permute(curr,[3 1 2]);
        cu_part = normalize_part(cu_part, gt_vis);
        mse_gt = sum((cu_part(:) - gt_part(:)).^2) / gt_c;
    else
        mse_gt = NaN;
    end
    % display gt pose mse
    text(1,10,['GT: MSE ' num2str(mse_gt,'%.4f')],'Color','k','FontSize',10,'BackgroundColor','w');

    % load nn training image
    nn_im_file = sprintf('./data/penn-crop/frames/%04d/%06d.jpg',nn_sid,nn_fid);
    nn_im = imread(nn_im_file);
    % load nn gt pose
    nn_lb_file = sprintf('./data/penn-crop/labels/%04d.mat',nn_sid);
    nn_anno = load(nn_lb_file);
    % plot nn pose on training image
    if exist('hn','var')
        delete(hn);
    end
    hn = subplot('Position',[2/4 0 1/4 1]); imshow(nn_im); hold on;
    for child = 2:param.p_no
        x1 = nn_anno.x(nn_fid,param.pa(child));
        y1 = nn_anno.y(nn_fid,param.pa(child));
        x2 = nn_anno.x(nn_fid,child);
        y2 = nn_anno.y(nn_fid,child);
        % skip invisible joints
        if nn_anno.visibility(nn_fid,child)
            plot(x2, y2, 'o', ...
                'color', param.partcolor{child}, ...
                'MarkerSize', param.msize, ...
                'MarkerFaceColor', param.partcolor{child});
            if nn_anno.visibility(nn_fid,param.pa(child))
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
    text(1,10,['NN: MSE ' num2str(nn_mse,'%.4f')],'Color','k','FontSize',10,'BackgroundColor','w');
    
    % display pred pose
    for j = 1:opt.seqLength
        % plot pred pose on input
        if exist('hp','var')
            delete(hp);
        end
        hp = subplot('Position',[3/4 0 1/4 1]); imshow(im); hold on;
        for child = 2:param.p_no
            x1 = pred(j,param.pa(child),1);
            y1 = pred(j,param.pa(child),2);
            x2 = pred(j,child,1);
            y2 = pred(j,child,2);
            % skip invisible joints
            if pred(j,child,1) ~= 0 && pred(j,child,2) ~= 0
                plot(x2, y2, 'o', ...
                    'color', param.partcolor{child}, ...
                    'MarkerSize', param.msize, ...
                    'MarkerFaceColor', param.partcolor{child});
                if pred(j,param.pa(child),1) ~= 0 && pred(j,param.pa(child),2) ~= 0
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
        set(gcf,'PaperPositionMode','auto');
        % save figure
        png_file = [png_dir sprintf('%03d-%02d.png',fid,j)];
        print(gcf,png_file,'-dpng','-r0');
    end
    
    % generate gif
    gif_file = fullfile(gif_dir,sprintf('%03d.gif',fid));
    for j = 1:opt.seqLength
        png_file = fullfile(png_dir,sprintf('%03d-%02d.png',fid,j));
        im = imread(png_file);
        [imind,cm] = rgb2ind(im,256);
        if j == 1
            imwrite(imind,cm,gif_file,'gif','Loopcount',inf,'DelayTime',0.25);
        else
            imwrite(imind,cm,gif_file,'gif','WriteMode','append','DelayTime',0.25);
        end
    end
end
fprintf('done.\n');

close;