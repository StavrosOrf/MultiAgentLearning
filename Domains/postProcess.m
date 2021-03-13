clear variables
close all

n_AGVs = 120 ;
folder = sprintf('Small_%d_AGVs_SS',n_AGVs) ;

figure ;
c = get(gca,'colororder') ;
c_cell = cell(size(c,1),1) ;
for i = 1:size(c,1)
    c_cell{i} = c(i,:) ;
end
fa = 0.1 ;
fs = 12 ;
l_str = {'Link','Link, time','Intersection','Intersection, time','Centralized', 'Centralized, time'} ;

epochs = 500 ;
runs = 0:29 ;

n_fac = numel(l_str) ;
pasigG = zeros(n_fac,1) ;
pasigMove = zeros(n_fac,1) ;
pasigEnter = zeros(n_fac,1) ;
pasigWait = zeros(n_fac,1) ;
plMeanG = zeros(n_fac,1) ;
plMeanMove = zeros(n_fac,1) ;
plMeanEnter = zeros(n_fac,1) ;
plMeanWait = zeros(n_fac,1) ;
plMaxG = zeros(n_fac,1) ;

p_res = 1:6 ;
final_G = zeros(n_fac,numel(runs)) ;

for k = p_res
    if (k == 1)
        domainDir = [folder '_link_no_time'] ;
    elseif (k == 2)
        domainDir = [folder '_link_time'] ;
    elseif (k == 3)
        domainDir = [folder '_intersection_no_time'] ;
    elseif (k == 4)
        domainDir = [folder '_intersection_time'] ;
    elseif (k == 5)
        domainDir = [folder '_centralised_no_time'] ;
    elseif (k == 6)
        domainDir = [folder '_centralised_time'] ;
    end

    results = [domainDir '/Results'] ;

    % Plot evaluation results and statistics
    G = zeros(epochs,length(runs)) ;
    tMove = zeros(epochs,length(runs)) ;
    tEnter = zeros(epochs,length(runs)) ;
    tWait = zeros(epochs,length(runs)) ;
    
    for i = runs
        data = csvread(sprintf('%s/evaluation_%d.csv',results,i)) ;
        G(:,i+1) = data(:,2) ;
        tMove(:,i+1) = data(:,3) ;
        tEnter(:,i+1) = data(:,4) ;
        tWait(:,i+1) = data(:,5) ;
    end
    final_G(k,:) = G(end,:) ;

    figure(1)
    hold on
    [meanG,sigG,meanGci,sigGi] = normfit(G') ;
%     paMeanGci(k) = patch([1:epochs,epochs:-1:1],[meanGci(1,:),fliplr(meanGci(2,:))],c(k),...
%         'facealpha',fa, 'linestyle','none') ;
    pasigG(k) = patch([1:epochs,epochs:-1:1],[meanG+sigG,fliplr(meanG-sigG)],c(k,:),...
        'facealpha',fa, 'linestyle','none') ;
    plMeanG(k) = plot(1:epochs,meanG,'color',c(k,:)) ;
    title(sprintf('Learning Performance, %d AGVs $\\left(\\mu, \\sigma\\right)$',n_AGVs),'interpreter','latex','fontsize',fs)
    xlabel('Learning epoch','interpreter','latex','fontsize',fs)
    ylabel('Total number of deliveries','interpreter','latex','fontsize',fs)
    set(gca,'ticklabelinterpreter','latex','fontsize',fs)

%     figure(2)
%     hold on
%     [meanMove,sigMove,meanMoveci,sigMovei] = normfit(tMove') ;
% %     paMeanMoveci(k) = patch([1:epochs,epochs:-1:1],[meanMoveci(1,:),fliplr(meanMoveci(2,:))],c(k),...
% %         'facealpha',fa, 'linestyle','none') ;
%     pasigMove(k) = patch([1:epochs,epochs:-1:1],[meanMove+sigMove,fliplr(meanMove-sigMove)],c(k,:),...
%         'facealpha',fa, 'linestyle','none') ;
%     plMeanMove(k) = plot(1:epochs,meanMove,'color',c(k,:)) ;
%     title('Total AGV Time Spent Moving')
%     xlabel('Learning epoch')
%     ylabel('AGV motion times')
%     set(2,'position',[1000 700 560 420])
% 
%     figure(3)
%     hold on
%     [meanEnter,sigEnter,meanEnterci,sigEnterci] = normfit(tEnter') ;
% %     paMeanEnterci(k) = patch([1:epochs,epochs:-1:1],[meanEnterci(1,:),fliplr(meanEnterci(2,:))],c(k),...
% %         'facealpha',fa, 'linestyle','none') ;
%     pasigEnter(k) = patch([1:epochs,epochs:-1:1],[meanEnter+sigEnter,fliplr(meanEnter-sigEnter)],c(k,:),...
%         'facealpha',fa, 'linestyle','none') ;
%     plMeanEnter(k) = plot(1:epochs,meanEnter,'color',c(k,:)) ;
%     title('Total AGV Time Spent Waiting to Enter')
%     xlabel('Learning epoch')
%     ylabel('AGV wait to enter times')
%     set(3,'position',[350 100 560 420])
% 
%     figure(4)
%     hold on
%     [meanWait,sigWait,meanWaitci,sigWaitci] = normfit(tWait') ;
% %     paMeanWaitci(k) = patch([1:epochs,epochs:-1:1],[meanWaitci(1,:),fliplr(meanWaitci(2,:))],c(k),...
% %         'facealpha',fa, 'linestyle','none') ;
%     pasigWait(k) = patch([1:epochs,epochs:-1:1],[meanWait+sigWait,fliplr(meanWait-sigWait)],c(k,:),...
%         'facealpha',fa, 'linestyle','none') ;
%     plMeanWait(k) = plot(1:epochs,meanWait,'color',c(k,:)) ;
%     title('Total AGV Time Spent Waiting at Intersections')
%     xlabel('Learning epoch')
%     ylabel('AGV wait to cross times')
%     set(4,'position',[1000 100 560 420])
    
    figure(5)
    hold on
    maxG = max(G,[],2) ;
    plMaxG(k) = plot(1:epochs,maxG,'color',c(k,:)) ;
    title(sprintf('Learning Performance, %d AGVs (max)',n_AGVs),'interpreter','latex','fontsize',fs)
    xlabel('Learning epoch','interpreter','latex','fontsize',fs)
    ylabel('Total number of deliveries','interpreter','latex','fontsize',fs)
    set(gca,'ticklabelinterpreter','latex','fontsize',fs)
end


hl(1) = legend(plMeanG(p_res),l_str{p_res},'location','southeast') ;
set(1,'position',[350 700 560 420])
% legend(plMeanMove(2:6),l_str{2:6},'location','southeast') ;
% legend(plMeanEnter(2:6),l_str{2:6}) ;
% legend(plMeanWait(2:6),l_str{2:6}) ;
hl(2) = legend(plMaxG(p_res),l_str{p_res},'location','southeast') ;
set(5,'position',[1000 700 560 420])

for i = 1:2
    set(hl(i),'interpreter','latex','fontsize',fs)
end

figure(6)
addpath('~/Documents/MATLAB/distributionPlot') ;
addpath('~/Documents/MATLAB/rotateticklabel') ;
hd = distributionPlot(final_G(p_res,:)','xNames',l_str(p_res),'color',c_cell) ;
set(gca,'ylim',[0 1100])
ht = rotateticklabel(gca,45);
for i = 1:numel(ht)
    set(ht(i),'interpreter','latex','fontsize',fs)
end
dataObj = get(gca,'children') ;
set(dataObj(7),'color','k','marker','x');
set(dataObj(8),'color','k');
title(sprintf('Final Team Performance (%d AGVs)',n_AGVs),'interpreter','latex','fontsize',fs)
ylabel('Total number of deliveries','interpreter','latex','fontsize',fs)
set(gca,'ticklabelinterpreter','latex','fontsize',fs)
set(6,'position',[350 100 560 470])
set(gca,'position',[0.1300 0.1960 0.7750 0.730])
legend(dataObj(7:8),{'median','mean'},'location','southeast','interpreter','latex','fontsize',fs)