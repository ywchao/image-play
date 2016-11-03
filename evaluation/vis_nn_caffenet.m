
config;

exp_name = 'hg-256';

% split = 'val';
% split = 'test';

% set visibility threshold for training data
% selected by maximizing validation accuracy
param.thres_vis = 4;

% set directories
curr_dir = sprintf('exp/penn-crop/pose-est-%s/eval_%s/',exp_name,split);
pred_dir = sprintf('exp/penn-crop/nn-caffenet-%s-th%02d/eval_%s/',exp_name,param.thres_vis,split);
vis_root = sprintf('evaluation/vis_pred/nn-caffenet-%s-thres%02d/%s/',exp_name,param.thres_vis,split);

% set body joint config
param.pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
param.p_no = numel(param.pa);

% set vis params
param.msize = 4;
param.partcolor = {'g','g','g','r','b','r','b','y','y','m','c','m','c'};

% set opt
opt.data = './data/Penn_Action_cropped';
opt.nPhase = 16;
opt.seqType = 'phase';
opt.seqLength = 16;
opt.inputRes = 256;
opt.outputRes = 64;

% init dataset
dataset = penn_crop(opt, split);

% get sampled indices
sidx = dataset.getSampledIdx();

fprintf('starting visualizing nn caffenet ... \n');
for i = 1:numel(sidx)
    tic_print(sprintf('  %04d/%04d\n',i,numel(sidx)));
    [sid, fid] = dataset.getSeqFrId(sidx(i));
    
    % set vis dir and file
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    makedir(vis_dir);
    
    % skip if all exist
    list_file = dir([vis_dir sprintf('%03d-',fid) '*']);
    list_file = {list_file.name}';
    if numel(list_file) == opt.seqLength
        continue
    end

    % load current frame estimation
    curr_file = sprintf('%s%05d.mat',curr_dir,sidx(i));
    curr = load(curr_file);
    curr = squeeze(curr.eval);
    
    % load prediction
    pred_file = sprintf('%s%05d.mat',pred_dir,sidx(i));
    pred = load(pred_file);
    nn_sid = pred.sid;
    nn_fid = pred.fid;
    nn_mse = pred.mse;
    pred = pred.eval;
    
    % clear figure
    clf; clear h;

    % load input im
    im_file = [frdata_root sprintf('%04d/%06d.jpg',sid,fid)];
    im = imread(im_file);
    % plot predicted pose
    subplot('Position',[0 0 1/4 1]); imshow(im); hold on;
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
    lb_file = [lbdata_root sprintf('%04d.mat',sid)];
    anno = load(lb_file);
    % plot gt pose
    subplot('Position',[1/4 0 1/4 1]); imshow(im); hold on;
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
    % % compute gt pose mse
    % gt_part = permute([anno.x(fid,:)' anno.y(fid,:)'],[3 1 2]);
    % gt_vis = anno.visibility(fid,:);
    % [gt_part, gt_c] = normalize_part(gt_part, gt_vis);
    % if gt_c >= param.thres_vis
    %     cu_part = permute(curr,[3 1 2]);
    %     cu_part = normalize_part(cu_part, gt_vis);
    %     mse_gt = sum((cu_part(:) - gt_part(:)).^2) / gt_c;
    % else
    %     mse_gt = NaN;
    % end
    % % display gt pose mse
    % text(1,10,['GT: MSE ' num2str(mse_gt,'%.4f')],'Color','k','FontSize',10,'BackgroundColor','w');

    % load nn training image
    nn_im_file = [frdata_root sprintf('%04d/%06d.jpg',nn_sid,nn_fid)];
    nn_im = imread(nn_im_file);
    % load nn gt pose
    nn_lb_file = [lbdata_root sprintf('%04d.mat',nn_sid)];
    nn_anno = load(nn_lb_file);
    % plot nn pose on training image
    subplot('Position',[2/4 0 1/4 1]); imshow(nn_im); hold on;
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
        % clear figure
        if exist('h','var')
            delete(h);
        end
        % plot pred pose on input
        h = subplot('Position',[3/4 0 1/4 1]); imshow(im); hold on;
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
        set(gcf,'Position',[0 0 1024 256]);
        set(gcf,'PaperPositionMode','auto');
        set(gcf,'color',[1 1 1]);
        set(gca,'color',[1 1 1]);
        % save figure
        vis_file = [vis_dir sprintf('%03d-%02d.png',fid,j)];
        print(gcf,vis_file,'-dpng','-r150');
    end
end
fprintf('done.\n');

close;