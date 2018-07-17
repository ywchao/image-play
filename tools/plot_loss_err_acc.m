
% exp_name = 'hg-256-res-clstm';
% exp_name = 'hg-256-res-clstm-res-64';

seq_length = 16;
max_epoch = 4;
epoch_size = 26253;
disp_int = 1500;

format = ['%s %s %s %s' repmat(' %s %s %s',[1,seq_length])];

figure(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/penn-crop/' exp_name '/train.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss = zeros(numel(iter),seq_length);
err = zeros(numel(iter),seq_length);
acc = zeros(numel(iter),seq_length);
for i = 1:seq_length
    loss(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*0+4}(2:end));
    err(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*1+4}(2:end));
    acc(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*2+4}(2:end));
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

subplot('Position',[0.027+0/3 0.56 1/3-0.03 0.4]);
plot(it,loss(ind,1),'--r'); hold on;
plot(it,loss(ind,2),'r');
for i = 3:16
    it_t = it(loss(ind,i) ~= 0);
    ind_t = ind(loss(ind,i) ~= 0);
    clr = (i-2)/(seq_length-2);
    plot(it_t,loss(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training loss');
xlabel('iteration');

subplot('Position',[0.027+1/3 0.56 1/3-0.03 0.4]);
plot(it,err(ind,1),'--r'); hold on;
plot(it,err(ind,2),'r');
for i = 3:16
    it_t = it(err(ind,i) ~= 0);
    ind_t = ind(err(ind,i) ~= 0);
    clr = (i-2)/(seq_length-2);
    plot(it_t,err(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training error');
xlabel('iteration');

subplot('Position',[0.027+2/3 0.56 1/3-0.03 0.4]);
plot(it,acc(ind,1),'--r'); hold on;
plot(it,acc(ind,2),'r');
for i = 3:16
    it_t = it(~isnan(acc(ind,i)));
    ind_t = ind(~isnan(acc(ind,i)));
    clr = (i-2)/(seq_length-2);
    plot(it_t,acc(ind_t,i),'color',[1-clr,0,clr]);
end
grid on;
axis([0 it(end) ylim]);
title('training accuracy');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/penn-crop/' exp_name '/val.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss = zeros(numel(iter),seq_length);
err = zeros(numel(iter),seq_length);
acc = zeros(numel(iter),seq_length);
for i = 1:seq_length
    loss(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*0+4}(2:end));
    err(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*1+4}(2:end));
    acc(:,i) = cellfun(@(x)str2double(x),C{i+seq_length*2+4}(2:end));
end

it = (epoch-1)*epoch_size + iter;

subplot('Position',[0.027+0/3 0.06 1/3-0.03 0.4]);
plot(it,loss(:,1),'--ro','MarkerSize',3); hold on;
plot(it,loss(:,2),'-ro','MarkerSize',3);
for i = 3:16
    it_t = it(loss(:,i) ~= 0);
    ind_t = loss(:,i) ~= 0;
    clr = (i-2)/(seq_length-2);
    plot(it_t,loss(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',3);
end
grid on;
title('validation loss');
xlabel('iteration');

subplot('Position',[0.027+1/3 0.06 1/3-0.03 0.4]);
plot(it,err(:,1),'--ro','MarkerSize',3); hold on;
plot(it,err(:,2),'-ro','MarkerSize',3);
for i = 3:16
    it_t = it(err(:,i) ~= 0);
    ind_t = err(:,i) ~= 0;
    clr = (i-2)/(seq_length-2);
    plot(it_t,err(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',3);
end
grid on;
title('validation error');
xlabel('iteration');

subplot('Position',[0.027+2/3 0.06 1/3-0.03 0.4]);
plot(it,acc(:,1),'--ro','MarkerSize',3); hold on;
plot(it,acc(:,2),'-ro','MarkerSize',3);
for i = 3:16
    it_t = it(~isnan(acc(:,i)));
    ind_t = ~isnan(acc(:,i));
    clr = (i-2)/(seq_length-2);
    plot(it_t,acc(ind_t,i),'-o','color',[1-clr,0,clr],'MarkerSize',3);
end
grid on;
title('validation accuracy');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot('Position',[0.027+0/3 0.56 1/3-0.03 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.027+1/3 0.56 1/3-0.03 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.027+2/3 0.56 1/3-0.03 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.027+0/3 0.06 1/3-0.03 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.027+1/3 0.06 1/3-0.03 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.027+2/3 0.06 1/3-0.03 0.4]); set(gca,'fontsize',6);

subplot('Position',[0.027+0/3 0.56 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 0.01]);
subplot('Position',[0.027+0/3 0.06 1/3-0.03 0.4]);
axis([lim(1:2) 0 0.01]);

subplot('Position',[0.027+1/3 0.56 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 0.5]);
subplot('Position',[0.027+1/3 0.06 1/3-0.03 0.4]);
axis([lim(1:2) 0 0.5]);

subplot('Position',[0.027+2/3 0.56 1/3-0.03 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 1]);
set(gca,'YTick',0:0.05:1.00);
subplot('Position',[0.027+2/3 0.06 1/3-0.03 0.4]);
axis([lim(1:2) 0.30 1]);
set(gca,'YTick',0.30:0.05:1.00);

% save to file
save_file = ['output/plot_' exp_name '.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPosition',[0 0 11 6]);
    set(gcf,'PaperOrientation','landscape');
    print(gcf,save_file,'-dpdf');
end

close;