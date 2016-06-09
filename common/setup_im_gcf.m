function [  ] = setup_im_gcf( im_row, im_col, gcf_pos, gca_pos, h_fig )

if nargin < 3
    gcf_pos = [0 0 im_col im_row];
end

if nargin < 4
    gca_pos = [0 0 1 1];
end

if nargin == 5
    figure(h_fig)
end

axis image; axis off;

set(gcf,'Position',gcf_pos);
set(gca,'Position',gca_pos);
set(gcf,'PaperPositionMode','auto');
set(gcf,'color',[1 1 1]);
set(gca,'color',[1 1 1]);

end