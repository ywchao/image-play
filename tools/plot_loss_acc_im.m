figure(1);

% choose experiment

% exp_name = 'seq16-hg-pi-clstm';
% epoch_size = 39379;
% disp_int = 2000;

% exp_name = 'seq16-hg-pi-res-clstm';
% epoch_size = 39379;
% disp_int = 2000;

% set parameters
max_epoch = 4;

seq_length = 16;
format = ['%s %s %s %s' repmat(' %s %s %s',[1,seq_length])];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/penn-crop/' exp_name '/train.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss_p = zeros(numel(iter),seq_length);
acc_p = zeros(numel(iter),seq_length);
loss_i = zeros(numel(iter),seq_length);
for i = 1:seq_length
    loss_p(:,i) = cellfun(@(x)str2double(x),C{i+4}(2:end));
    acc_p(:,i) = cellfun(@(x)str2double(x),C{i+seq_length+4}(2:end));
    loss_i(:,i) = cellfun(@(x)str2double(x),C{i+2*seq_length+4}(2:end));
end

ind = [];
for i = 1:max_epoch
    ii = find(epoch == i);
    if isempty(ii)
        continue
    end
    % sample index uniformly
    ind = [ind; ii(1:disp_int:numel(ii))];  %#ok
    % add the last iter of each epoch
    if ismember(epoch_size*i, ii) && ~ismember(epoch_size*i, ind)
        ind = [ind; epoch_size*i];  %#ok
    end
end
it = (epoch(ind)-1)*epoch_size + iter(ind);

subplot('Position',[0.03+0/3 0.55 1/3-0.03 0.4]);
plot(it,loss_p(ind,1),'--r'); hold on;
plot(it,loss_p(ind,2),'r');
for i = 3:16
    it_t = it(loss_p(ind,i) ~= 0);
    ind_t = ind(loss_p(ind,i) ~= 0);
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,loss_p(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training loss pose');
xlabel('iteration');

subplot('Position',[0.03+1/3 0.55 1/3-0.03 0.4]);
plot(it,acc_p(ind,1),'--r'); hold on;
plot(it,acc_p(ind,2),'r');
for i = 3:16
    it_t = it(~isnan(acc_p(ind,i)));
    ind_t = ind(~isnan(acc_p(ind,i)));
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,acc_p(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training accuracy pose');
xlabel('iteration');

subplot('Position',[0.03+2/3 0.55 1/3-0.03 0.4]);
plot(it,loss_i(ind,1),'--r'); hold on;
plot(it,loss_i(ind,2),'r');
for i = 3:16
    it_t = it(loss_i(ind,i) ~= 0);
    ind_t = ind(loss_i(ind,i) ~= 0);
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,loss_i(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training loss image');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/penn-crop/' exp_name '/val.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss_p = zeros(numel(iter),seq_length);
acc_p = zeros(numel(iter),seq_length);
loss_i = zeros(numel(iter),seq_length);
for i = 1:seq_length
    loss_p(:,i) = cellfun(@(x)str2double(x),C{i+4}(2:end));
    acc_p(:,i) = cellfun(@(x)str2double(x),C{i+seq_length+4}(2:end));
    loss_i(:,i) = cellfun(@(x)str2double(x),C{i+2*seq_length+4}(2:end));
end

it = (epoch-1)*epoch_size + iter;

subplot('Position',[0.03+0/3 0.05 1/3-0.03 0.4]);
plot(it,loss_p(:,1),'--ro','MarkerSize',2); hold on;
plot(it,loss_p(:,2),'-ro','MarkerSize',2);
for i = 3:16
    it_t = it(loss_p(:,i) ~= 0);
    ind_t = loss_p(:,i) ~= 0;
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,loss_p(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',2);
end
grid on;
title('validation loss pose');
xlabel('iteration');

subplot('Position',[0.03+1/3 0.05 1/3-0.03 0.4]);
plot(it,acc_p(:,1),'--ro','MarkerSize',2); hold on;
plot(it,acc_p(:,2),'-ro','MarkerSize',2);
for i = 3:16
    it_t = it(~isnan(acc_p(:,i)));
    ind_t = ~isnan(acc_p(:,i));
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,acc_p(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',2);
end
grid on;
title('validation accuracy pose');
xlabel('iteration');

subplot('Position',[0.03+2/3 0.05 1/3-0.03 0.4]);
plot(it,loss_i(:,1),'--ro','MarkerSize',2); hold on;
plot(it,loss_i(:,2),'-ro','MarkerSize',2);
for i = 3:16
    it_t = it(loss_i(:,i) ~= 0);
    ind_t = loss_i(:,i) ~= 0;
    clr = (i-2)/(seq_length-2);
    % clr = (i-2)/(8-2);
    plot(it_t,loss_i(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',2);
end
grid on;
title('validation loss image');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot('Position',[0.03+0/3 0.55 1/3-0.03 0.4]); set(gca,'fontsize',5);
subplot('Position',[0.03+1/3 0.55 1/3-0.03 0.4]); set(gca,'fontsize',5);
subplot('Position',[0.03+2/3 0.55 1/3-0.03 0.4]); set(gca,'fontsize',5);
subplot('Position',[0.03+0/3 0.05 1/3-0.03 0.4]); set(gca,'fontsize',5);
subplot('Position',[0.03+1/3 0.05 1/3-0.03 0.4]); set(gca,'fontsize',5);
subplot('Position',[0.03+2/3 0.05 1/3-0.03 0.4]); set(gca,'fontsize',5);

subplot('Position',[0.03+0/3 0.55 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 0.01]);
subplot('Position',[0.03+0/3 0.05 1/3-0.03 0.4]);
axis([lim(1:2) 0.0015 0.0045]);

subplot('Position',[0.03+1/3 0.55 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 1]);
subplot('Position',[0.03+1/3 0.05 1/3-0.03 0.4]);
axis([lim(1:2) 0.35 0.90]);
set(gca,'YTick',0.35:0.05:0.90);

subplot('Position',[0.03+2/3 0.55 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 0.05]);
subplot('Position',[0.03+2/3 0.05 1/3-0.03 0.4]);
axis([lim(1:2) 0 0.025]);

% save to file
save_file = ['outputs/plot_' exp_name '.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPosition',[0 0 10.8 5]);
    set(gcf,'PaperOrientation','landscape');
    print(gcf,save_file,'-dpdf');
end

close;