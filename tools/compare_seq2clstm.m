figure(1);

idx_epoch = 1;
idx_loss1 = 2;
idx_loss2 = 3;
idx_acc1 = 7;
idx_acc2 = 6;

% clstm-init0
log_file = './exp/penn-crop/seq2-hg-no-skip-clstm-init0/train.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s %s %s');
fclose(f);
epoch = cellfun(@(x)str2double(x),C{idx_epoch}(2:end));
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

% int = 1000;
ind = [];
for i = 1:100
    ii = find(epoch == i);
    % if isempty(ii)
    if numel(ii) < 19690
        continue
    end
    % ii = ii(1:int:numel(ii));
    % ind = [ind; ii];  %#ok
    % ind = [ind; ii(1)]; %#ok
    ind = [ind; ii(19690)]; %#ok
end
ind = [1; ind];

subplot(2,2,1);
plot(0:numel(ind)-1,loss1(ind),'--r'); hold on;
plot(0:numel(ind)-1,loss2(ind),'r');

subplot(2,2,2);
plot(0:numel(ind)-1,acc1(ind),'--r'); hold on;
plot(0:numel(ind)-1,acc2(ind),'r');

% res-clstm-init0
log_file = './exp/penn-crop/seq2-hg-no-skip-res-clstm-init0/train.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s %s %s');
fclose(f);
epoch = cellfun(@(x)str2double(x),C{idx_epoch}(2:end));
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

% int = 1000;
ind = [];
for i = 1:100
    ii = find(epoch == i);
    % if isempty(ii)
    if numel(ii) < 19690
        continue
    end
    % ii = ii(1:int:numel(ii));
    % ind = [ind; ii];  %#ok
    % ind = [ind; ii(1)]; %#ok
    ind = [ind; ii(19690)]; %#ok
end
ind = [1; ind];

subplot(2,2,1);
plot(0:numel(ind)-1,loss1(ind),'--b');
plot(0:numel(ind)-1,loss2(ind),'b');

subplot(2,2,2);
plot(0:numel(ind)-1,acc1(ind),'--b');
plot(0:numel(ind)-1,acc2(ind),'b');

% clstm-init1
log_file = './exp/penn-crop/seq2-hg-no-skip-clstm-init1/train.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s %s %s');
fclose(f);
epoch = cellfun(@(x)str2double(x),C{idx_epoch}(2:end));
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

% int = 1000;
ind = [];
for i = 1:100
    ii = find(epoch == i);
    % if isempty(ii)
    if numel(ii) < 19690
        continue
    end
    % ii = ii(1:int:numel(ii));
    % ind = [ind; ii];  %#ok
    % ind = [ind; ii(1)]; %#ok
    ind = [ind; ii(19690)]; %#ok
end
ind = [1; ind];

subplot(2,2,1);
plot(0:numel(ind)-1,loss1(ind),'--m');
plot(0:numel(ind)-1,loss2(ind),'m');

subplot(2,2,2);
plot(0:numel(ind)-1,acc1(ind),'--m');
plot(0:numel(ind)-1,acc2(ind),'m');

% res-clstm-init1
log_file = './exp/penn-crop/seq2-hg-no-skip-res-clstm-init1/train.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s %s %s');
fclose(f);
epoch = cellfun(@(x)str2double(x),C{idx_epoch}(2:end));
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

% int = 1000;
ind = [];
for i = 1:100
    ii = find(epoch == i);
    % if isempty(ii)
    if numel(ii) < 19690
        continue
    end
    % ii = ii(1:int:numel(ii));
    % ind = [ind; ii];  %#ok
    % ind = [ind; ii(1)]; %#ok
    ind = [ind; ii(19690)]; %#ok
end
ind = [1; ind];

subplot(2,2,1);
plot(0:numel(ind)-1,loss1(ind),'--g');
plot(0:numel(ind)-1,loss2(ind),'g');

subplot(2,2,2);
plot(0:numel(ind)-1,acc1(ind),'--g');
plot(0:numel(ind)-1,acc2(ind),'g');


% set legend
subplot(2,2,1);
h_leg = legend('hg-single-no-skip clstm init0 (loss1)', ...
    'hg-single-no-skip clstm init0 (loss2)', ...
    'hg-single-no-skip res-clstm init0 (loss1)', ...
    'hg-single-no-skip res-clstm init0 (loss2)', ...
    'hg-single-no-skip clstm init1 (loss1)', ...
    'hg-single-no-skip clstm init1 (loss2)', ...
    'hg-single-no-skip res-clstm init1 (loss1)', ...
    'hg-single-no-skip res-clstm init1 (loss2)', ...
    'Location', 'northeast');
set(h_leg,'FontSize',5);
subplot(2,2,2);
h_leg = legend('hg-single-no-skip clstm init0 (loss1)', ...
    'hg-single-no-skip clstm init0 (loss2)', ...
    'hg-single-no-skip res-clstm init0 (loss1)', ...
    'hg-single-no-skip res-clstm init0 (loss2)', ...
    'hg-single-no-skip clstm init1 (loss1)', ...
    'hg-single-no-skip clstm init1 (loss2)', ...
    'hg-single-no-skip res-clstm init1 (loss1)', ...
    'hg-single-no-skip res-clstm init1 (loss2)', ...
    'Location', 'southeast');
set(h_leg,'FontSize',5);

subplot(2,2,1);
lim = xlim;
set(gca,'XTick',0:lim(2));
grid on;
title('training loss');
xlabel('epoch');
subplot(2,2,2);
lim = xlim;
set(gca,'XTick',0:lim(2));
grid on;
title('training accuracy');
xlabel('epoch');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx_epoch = 1;
idx_loss1 = 6;
idx_loss2 = 2;
idx_acc1 = 5;
idx_acc2 = 4;

% clstm-init0
log_file = './exp/penn-crop/seq2-hg-no-skip-clstm-init0/val.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s');
fclose(f);
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

subplot(2,2,3);
plot(1:numel(loss1),loss1,'--r'); hold on
plot(1:numel(loss2),loss2,'r');

subplot(2,2,4);
plot(1:numel(acc1),acc1,'--r'); hold on
plot(1:numel(acc2),acc2,'r');

% res-clstm-init0
log_file = './exp/penn-crop/seq2-hg-no-skip-res-clstm-init0/val.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s');
fclose(f);
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

subplot(2,2,3);
plot(1:numel(loss1),loss1,'--b');
plot(1:numel(loss2),loss2,'b');

subplot(2,2,4);
plot(1:numel(acc1),acc1,'--b');
plot(1:numel(acc2),acc2,'b');

% clstm-init1
log_file = './exp/penn-crop/seq2-hg-no-skip-clstm-init1/val.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s');
fclose(f);
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

subplot(2,2,3);
plot(1:numel(loss1),loss1,'--m');
plot(1:numel(loss2),loss2,'m');

subplot(2,2,4);
plot(1:numel(acc1),acc1,'--m');
plot(1:numel(acc2),acc2,'m');

% res-clstm-init1
log_file = './exp/penn-crop/seq2-hg-no-skip-res-clstm-init1/val.log';
f = fopen(log_file);
C = textscan(f,'%s %s %s %s %s %s');
fclose(f);
loss1 = cellfun(@(x)str2double(x),C{idx_loss1}(2:end));
loss2 = cellfun(@(x)str2double(x),C{idx_loss2}(2:end));
acc1 = cellfun(@(x)str2double(x),C{idx_acc1}(2:end));
acc2 = cellfun(@(x)str2double(x),C{idx_acc2}(2:end));

subplot(2,2,3);
plot(1:numel(loss1),loss1,'--g');
plot(1:numel(loss2),loss2,'g');

subplot(2,2,4);
plot(1:numel(acc1),acc1,'--g');
plot(1:numel(acc2),acc2,'g');


subplot(2,2,3);
lim = xlim;
set(gca,'XTick',0:lim(2));
axis([0 lim(2) ylim]);
grid on;
title('validation loss');
xlabel('epoch');
subplot(2,2,4);
lim = xlim;
set(gca,'XTick',0:lim(2));
grid on;
axis([0 lim(2) ylim]);
title('validation accuracy');
xlabel('epoch');

% save to file
save_file = 'outputs/seq2clstm.pdf';
if ~exist(save_file,'file')
    set(gcf,'PaperPosition',[0 0 8 6]);
    subplot(2,2,1); set(gca,'fontsize',8);
    subplot(2,2,2); set(gca,'fontsize',8);
    subplot(2,2,3); set(gca,'fontsize',8);
    subplot(2,2,4); set(gca,'fontsize',8);
    print(gcf,save_file,'-dpdf');
end

close;
