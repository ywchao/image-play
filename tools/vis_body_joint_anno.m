
config;

% head
% right-shoulder
% left-shoulder
% right-elbow
% left-elbow
% right-wrist
% left-wrist
% right-hip
% left-hip
% right-knee
% left-knee
% right-ankle
% left-ankle

% set body joint config
pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
p_no = numel(pa);

% set vis params
msize = 4;
partcolor = {'g','g','g','r','b','r','b','y','y','m','c','m','c'};

% choose source

% % 1. with image displayed
% vis_im = true;
% vis_root = './outputs/dataset_vis/body_joints/';

% 2. without image displayed
vis_im = false;
vis_root = './outputs/dataset_vis/body_joints_only/';

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

figure(1);

% reading annotations
fprintf('visualizing body joints ... \n');
for i = 1:num_seq
    fprintf('%04d/%04d\n',i,num_seq);
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    % set vis dir
    vis_dir = [vis_root name_seq '/'];
    makedir(vis_dir);
    for j = 1:anno.nframes
        tic_print(sprintf('  %03d/%03d\n',j,anno.nframes));
        % skip if vis file exists
        vis_file = [vis_dir list_fr{j}];
        if exist(vis_file,'file')
            continue
        end
        % read image
        file_im = [fr_dir list_fr{j}];
        im = imread(file_im);
        % plot body joints on the image
        if exist('h','var')
            delete(h);
        end
        if vis_im
            h = imshow(im); hold on;
        else
            im_bg = uint8(64*ones(size(im,1),size(im,2),1));
            h =  imshow(im_bg); hold on;
        end
        setup_im_gcf(size(im,1),size(im,2));
        for child = 2:p_no
            x1 = anno.x(j,pa(child));
            y1 = anno.y(j,pa(child));
            x2 = anno.x(j,child);
            y2 = anno.y(j,child);
            % skip invisible joints
            if anno.visibility(j,child)
                plot(x2, y2, 'o', ...
                    'color', partcolor{child}, ...
                    'MarkerSize', msize, ...
                    'MarkerFaceColor', partcolor{child});
                if anno.visibility(j,pa(child))
                    plot(x1, y1, 'o', ...
                        'color', partcolor{child}, ...
                        'MarkerSize', msize, ...
                        'MarkerFaceColor', partcolor{child});
                    line([x1 x2], [y1 y2], ...
                        'color', partcolor{child}, ...
                        'linewidth',round(msize/2));
                end
            end
        end
        drawnow;
        % pause(0);
        % save vis to file
        print(gcf,vis_file,'-djpeg','-r0');
    end
    clf;
end
fprintf('\n');

close;
