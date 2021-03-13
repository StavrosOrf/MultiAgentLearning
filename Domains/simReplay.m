clear variables
close all

domainDir = 'Small_120_AGVs_SS_link_no_time' ;

verticesFile = [domainDir '/vertices_XY.csv'] ;
edgesFile = [domainDir '/edges.csv'] ;
capsFile = [domainDir '/capacities.csv'] ;
distsFile = [domainDir '/distances.csv'] ;
origsFile = [domainDir '/origins.csv'] ;

results = [domainDir '/Results'] ;

epochs = 500 ;
runs = 0:9 ;

data = csvread(sprintf('%s/evaluation_%d.csv',results,runs(end))) ;

%% Replay first and last episodes
nodes = csvread(verticesFile) ;
edges = csvread(edgesFile) ;
caps = csvread(capsFile) ;
edgeLen = edges(:,3) ;
eInd = edges + 1 ;
xrange = max(nodes(:,1)) - min(nodes(:,1)) ;
yrange = max(nodes(:,2)) - min(nodes(:,2)) ;
d = 0.01 ; % separate links in plot
cbar = 0.5 ; % per uav on capacity bar
fs = 12 ; % font size

pCap = 1 ;
pCost = 1 ;

% Plot graph
figure
fh = gcf ;
hold on ;
axis equal ;
alpha = zeros(size(eInd,1),1) ;
hN = plot(nodes(:,1),nodes(:,2),'ko','markersize',15,'linewidth',3) ;
for i = 1:size(nodes, 1)
    text(nodes(i,1)-0.04, nodes(i,2), num2str(i-1),'fontsize',8,'color',[0.5 0.5 0.5]) ;
end

ePos = cell(size(eInd,1),1) ; % store possible (x,y) positions that AGVs can take on each edge

hE = zeros(size(eInd,1),1) ;    % plot graph edges
hBu = zeros(size(eInd,1),1) ;   % bar displaying total capacity
hB = zeros(size(eInd,1),1) ;    % bar displaying current number of AGVs
hT = zeros(size(eInd,1),1) ;    % print out edge costs as "a: ..."
for i = 1:size(eInd,1)
    x = [nodes(eInd(i,1),1),nodes(eInd(i,2),1)] ;
    y = [nodes(eInd(i,1),2),nodes(eInd(i,2),2)] ;
    diffx = x(2) - x(1) ;
    diffy = y(2) - y(1) ;
    % blue paths map traffic going to the right, ties broken by paths going
    % upwards. Red paths otherwise.
    alpha(i) = atan2(diffy,diffx) ;
    dx = -d*xrange*sin(alpha(i)) ;
    dy = d*yrange*cos(alpha(i)) ;
    if (diffx > 0 || (diffx == 0 && diffy > 0) )
        c = [0.5,0.5,1] ; % light blue
        cc = [0.75,0.75,1] ;
        cb = [0.5,0.5,1] ;
    else
        c = [1,0.5,0.5] ; % light red
        cc = [1,0.75,0.75] ;
        cb = [1,0.25,0.25] ;
    end
    hE(i) = plot(x+dx,y+dy,'-','color',c,'linewidth',2) ;
    ePos{i} = zeros(2,edgeLen(i)) ;
    temp_x = linspace(x(1)+dx,x(2)+dx,edgeLen(i)+1) ;
    ePos{i}(1,:) = temp_x(2:end) ;
    temp_y = linspace(y(1)+dy,y(2)+dy,edgeLen(i)+1) ;
    ePos{i}(2,:) = temp_y(2:end) ;
    
    % Plot capacity bars
    hBscale = 2 ;
    hBthick = 12 ;
    midx = diffx/2 + x(1) ;
    midy = diffy/2 + y(1) ;
    barx(1) = midx - cbar*caps(i)/2*cos(alpha(i)) ;
    barx(2) = midx + cbar*caps(i)/2*cos(alpha(i)) ;
    bary(1) = midy - cbar*caps(i)/2*sin(alpha(i)) ;
    bary(2) = midy + cbar*caps(i)/2*sin(alpha(i)) ;
    if pCap
        hBu(i) = plot(barx+hBscale*dx,bary+hBscale*dy,'-','color',cc,'linewidth',hBthick) ;
        hB(i) = plot(barx(1)+hBscale*dx,bary(1)+hBscale*dy,'-','color',cb,'linewidth',hBthick) ;
    end
    
    % Plot agent output costs
    hTscale = 2 ;
    if pCost
        hT(i) = text(barx(1)+hTscale*dx,bary(1)+hTscale*dy,'a:','rotation',alpha(i)*180/pi,'fontsize',fs) ;
    end
end
axis tight ;

team = data(end,1) ; % index of champion team
fl_epoch = [0,epochs-1] ;
tf = 200 ; % number of timesteps per epoch
j = (tf+1)*team + 1 ; % starting index for champion episode

fl = [0,0,0 ;       % en route
    0.8,0.8,0.8 ;   % waiting to enter graph
    1,0,0];         % waiting en route

agv_0 = csvread(origsFile)' ;
agv_0 = agv_0 + 1 ; % index for matlab
    
fc = rand(size(agv_0,2),3) ; % AGV colours
    
% set up handles
h_agvs = zeros(size(agv_0,2),1) ;
for k = 1:size(agv_0,2)
    h_agvs(k) = plot(nodes(agv_0(k),1),nodes(agv_0(k),2),'o',...
        'markersize',10,'linewidth',3,'color',fl(2,:),...
        'markerfacecolor',fc(k,:)) ;
end

for i = fl_epoch
    actions = csvread(sprintf('%s/Replay/agent_actions_%d.csv',results,i)) ;
    states = csvread(sprintf('%s/Replay/agent_states_%d.csv',results,i)) ;
    agv_s = csvread(sprintf('%s/Replay/AGV_states_%d.csv',results,i)) ;
    agv_e = csvread(sprintf('%s/Replay/AGV_edges_%d.csv',results,i)) ;
    
    % trim to champion episode
    actions = actions(j:(j+tf),1:(end-1)) ;
    states = states(j:(j+tf),1:(end-1)) ;
    agv_s = agv_s(j:(j+tf),1:(end-1)) ;
    agv_e = agv_e(j:(j+tf),1:(end-1)) ;
    
    agv_sc = zeros(1,size(agv_e,2)) ;
    agv_init = ones(1,size(agv_e,2)) ;
    agv_ce = zeros(1,size(agv_e,2)) ;
    
    for k = 1:size(agv_0,2)
        set(h_agvs(k),'xdata',nodes(agv_0(k),1),'ydata',nodes(agv_0(k),2),'color',fl(2,:)) ;
    end
    
    for t = 1:tf+1
        for n = 1:size(eInd,1)
            % Display edge states
        	xlim = get(hBu(n),'xdata') ;
        	ylim = get(hBu(n),'ydata') ;
            xs = (xlim(2)-xlim(1))*states(t,n)/caps(n) + xlim(1) ;
            ys = (ylim(2)-ylim(1))*states(t,n)/caps(n) + ylim(1) ;
            set(hB(n),'xdata',[xlim(1) xs],'ydata',[ylim(1) ys]) ;

            % Display edge costs
            set(hT(n),'string',sprintf('a: %.2f',actions(t,n))) ;
        end
        
        % Plot AGVs
        for k = 1:size(agv_0,2)
            if (agv_s(t,k) == 0) % AGV en route
                if (agv_init(k) == 1) % switch off initialisation flag
                    agv_init(k) = 0 ;
                end
                e = agv_e(t,k) + 1 ; % current edge indexed for matlab
                if (agv_ce(k) ~= e) % check if agv has changed edges
                    agv_sc(k) = 0 ; % reset counter
                end
                agv_ce(k) = e ;  % update current edge
                agv_sc(k) = agv_sc(k) + 1 ; % move along edge
                if (agv_sc(k) > size(ePos{e},2)) % check if agv is waiting at an intersection
                    lc = fl(3,:) ;
                else
                    lc = fl(1,:) ;
                end
                e_sc = min(agv_sc(k),size(ePos{e},2)) ;
                xpos = ePos{e}(1,e_sc) ;
                ypos = ePos{e}(2,e_sc) ;
                set(h_agvs(k),'xdata',xpos,'ydata',ypos,'color',lc) ;
            elseif (agv_s(t,k) == 1 && agv_init(k) == 0) % AGV waiting to enter
                agv_sc(k) = 0 ;
                lc = fl(2,:) ;
                e = agv_ce(k) ; % read previous edge
                xpos = ePos{e}(1,end) ;
                ypos = ePos{e}(2,end) ;
                set(h_agvs(k),'xdata',xpos,'ydata',ypos) ;
                set(h_agvs(k),'color',lc) ;
                agv_init(k) = 1 ;
            end
        end
        title(sprintf('Champion team replay: Episode #%d, %d s',i,t-1)) ;
        drawnow
        pause
    end
end
