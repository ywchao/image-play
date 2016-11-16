split = 'test';

save_root = 'outputs/figures/pck_curves/';
makedir(save_root);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_name = 'pose-est-hg-256';
clr = 'c';
opt.seqLength = 1;
figure(100);
eval_pck;
for i = 1:opt.seqLength
    figure(i);
    set(gcf,'Position',[2 26 256 256]);
    set(gca,'Position',[0.07 0.07 0.91 0.91]);
    set(gca,'fontsize',8);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_name = 'nn-skel-hg-256-th09';
clr = 'r';
opt.seqLength = 16;
figure(100);
eval_pck;
for i = 1:opt.seqLength
    figure(i);
    set(gcf,'Position',[2 26 256 256]);
    set(gca,'Position',[0.07 0.07 0.91 0.91]);
    set(gca,'fontsize',8);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_name = 'nn-skel-caffenet-hg-256-th09-K22500';
clr = 'g';
opt.seqLength = 16;
figure(100);
eval_pck;
for i = 1:opt.seqLength
    figure(i);
    set(gcf,'Position',[2 26 256 256]);
    set(gca,'Position',[0.07 0.07 0.91 0.91]);
    set(gca,'fontsize',8);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_name = 'nn-skel-oracle-hg-256-th09';
clr = 'k';
opt.seqLength = 16;
figure(100);
eval_pck;
for i = 1:opt.seqLength
    figure(i);
    set(gcf,'Position',[2 26 256 256]);
    set(gca,'Position',[0.07 0.07 0.91 0.91]);
    set(gca,'fontsize',8);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exp_name = 'seq16-hg-256-res-clstm-res-64-w1e-6';
clr = 'b';
opt.seqLength = 16;
figure(100);
eval_pck;
for i = 1:opt.seqLength
    figure(i);
    set(gcf,'Position',[2 26 256 256]);
    set(gca,'Position',[0.07 0.07 0.91 0.91]);
    set(gca,'fontsize',8);
    plot(range,acc(i,:),'color',clr,'LineStyle','-','LineWidth',1);
    hold on;
    grid on;
    axis([0 0.1 0 1]);
    set(gca,'xtick',0:0.02:0.1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:16
    figure(i);
    if i == 1
        hl = legend('Hourglass [18]','NN-all','NN-Caffenet','NN-oracle','3D-PPNet', ...
            'Location','southeast');
        set(hl,'FontSize',10);
    end
    save_file = [save_root sprintf('%02d.pdf',i)];
    if ~exist(save_file,'file')
        set(gcf,'PaperPositionMode','auto');
        print(gcf,save_file,'-dpdf','-r0');
    end
end
close all;
