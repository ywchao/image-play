
config;

addpath('./ijcv_flow_code/utils');
addpath('./ijcv_flow_code/utils/flowColorCode');

% set parameters
samp = true;

n_phase = 16;

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

res_root = './flownet-release/models/flownet/res_penn-crop_samp/';
vis_root = './flownet-release/models/flownet/vis_penn-crop_samp/';

figure(1);

for i = 1:num_seq
    tic_print(sprintf('  %04d/%04d\n',i,num_seq));
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frdata_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    num_fr = numel(list_fr);
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    % get dir
    count = -1;
    for j = 1:anno.nframes
        % for now just consider one sequence
        if samp && j > 1
            continue
        end
        % set and make directories
        res_dir = [res_root name_seq '/' num2str(j,'%03d') '/'];
        vis_dir = [vis_root name_seq '/' num2str(j,'%03d') '/'];
        makedir(vis_dir);
        % get seq ind
        ind = linspace(j,j+num_fr-1,n_phase);
        ind = round(ind);
        ind(ind > num_fr) = num_fr;
        for k = 1:numel(ind)-1
            count = count + 1;
            vis_file = [vis_dir num2str(ind(k),'%04d') '_' num2str(ind(k+1),'%04d') '.png'];
            % skip if existed
            if exist(vis_file,'file')
                continue
            end
            im_file_1 = [fr_dir list_fr{ind(k)}];
            im_file_2 = [fr_dir list_fr{ind(k+1)}];
            im_file_f = [res_dir 'flownets-pred-' num2str(count,'%07d') '.flo'];
            im_1 = imread(im_file_1);
            im_2 = imread(im_file_2);
            im_f = readFlowFile(im_file_f);
            
            alpha = 1/2;
            im_c = uint8(double(im_1)*alpha+double(im_2)*(1-alpha));
           
            subplot('Position',[0 0 1/3 1]); imshow(im_c);
            subplot('Position',[1/3 0 1/3 1]); imshow(flowToColor(im_f));
            % subplot('Position',[2/3 0 1/3 1]); plotflow(im_v); axis off;
            subplot('Position',[2/3 0 1/3 1]);
            rfactor = 0.1;
            im_v = imresize(im_f, rfactor);
            [X, Y] = meshgrid(1:size(im_v,2), 1:size(im_v,1));
            quiver(X,Y,im_v(:,:,1),im_v(:,:,2));
            set(gca,'YDir','Reverse');
            
            axis image; axis off;

            set(gcf,'Position',[0 0 size(im_1,2)*3 size(im_1,1)]);
            set(gcf,'PaperPositionMode','auto');
            set(gcf,'color',[1 1 1]);
            set(gca,'color',[1 1 1]);
            
            print(gcf,vis_file,'-dpng','-r150');
        end
    end
end

close;
