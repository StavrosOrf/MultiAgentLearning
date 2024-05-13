#author: Kallinteris Andreas

#version history
#v3 added support for new definitons and algos
#v4 2x2 
#v5 both time -> both times, avg & max graphs are 4x1 now

import matplotlib.pyplot
import os
import sys
import statistics
from scipy.stats import skew


def read_runs(run_path):
    data_g = []
    files = os.listdir(run_path) 
    for file in files:
        #if not file.endswith('.csv') and not file.endswith('.yaml'):
        if not '.' in file:
            if os.path.isfile(os.path.join(run_path, file)):
                f = open(os.path.join(run_path, file),'r')
                #print(f)
                data_g.append([float(i) for i in f.read().splitlines()])
                f.close()
    return data_g

def average_runs(data_g):
    assert(len(data_g) > 0)
    avg = [0] * len(data_g[0])
    for i in data_g:
        for j in range(len(avg)):
            avg[j] += i[j]
    avg = [x / len(data_g) for x in avg]
    return avg

def min_of_runs(data_g):
    minimum = [float("inf")] * len(data_g[0])
    for i in data_g:
        for j in range(len(minimum)):
            if minimum[j] > i[j]:
                minimum[j] = i[j]
    return minimum

def max_of_runs(data_g):
    maximum = [0] * len(data_g[0])
    for i in data_g:
        for j in range(len(maximum)):
            if maximum[j] < i[j]:
                maximum[j] = i[j]
    return maximum



if sys.argv[1] == '--max':
    PLOT_TYPE = 'max'
if sys.argv[1] == '--ave':
    PLOT_TYPE = 'average'
if sys.argv[1] == '--vio':
    PLOT_TYPE = 'violin'
if sys.argv[1] == '--print':
    PLOT_TYPE = 'print'


AGV_COUNT = ['90', '120', '200', '400'] # number of Autonomous Ground Vehicles (AGVs)
#AGV_COUNT = ['90', '120', '200'] # number of Autonomous Ground Vehicles (AGVs)
if sys.argv[2] == '--es':
    ALGO = 'ES'
elif sys.argv[2] == '--es_cn':
    ALGO = 'ES_CN'
elif sys.argv[2] == '--ccea':
    ALGO = 'CCEA'

Y_LIM_AVE = [(350, 575), (400, 725), (400, 900), (400, 950)]
Y_LIM_MAX = [(450, 580), (450, 750), (450, 950), (500, 950)]
if PLOT_TYPE == 'violin':
    fig, axis = matplotlib.pyplot.subplots(2, 2)
    #fig, axis = matplotlib.pyplot.subplots(1,len(AGV_COUNT))
else:
    #fig, axis = matplotlib.pyplot.subplots(1,len(AGV_COUNT))
    #fig, axis = matplotlib.pyplot.subplots(2, 2)
    fig, axis = matplotlib.pyplot.subplots(4,1)

for i in range(len(AGV_COUNT)):
    data_c = read_runs(ALGO + '/C_' + AGV_COUNT[i])
    avg_c = average_runs(data_c)
    min_c = min_of_runs(data_c)
    max_c = max_of_runs(data_c)
    last_c = [data_c[j][-1] for j in range(len(data_c))]

    data_c_t = read_runs(ALGO + '/C_t_' + AGV_COUNT[i])
    avg_c_t = average_runs(data_c_t)
    min_c_t = min_of_runs(data_c_t)
    max_c_t = max_of_runs(data_c_t)
    last_c_t = [data_c_t[j][-1] for j in range(len(data_c_t))]

    data_c_avgt = read_runs(ALGO + '/C_avgt_' + AGV_COUNT[i])
    avg_c_avgt = average_runs(data_c_avgt)
    min_c_avgt = min_of_runs(data_c_avgt)
    max_c_avgt = max_of_runs(data_c_avgt)
    last_c_avgt = [data_c_avgt[j][-1] for j in range(len(data_c_avgt))]

    data_c_botht = read_runs(ALGO + '/C_botht_' + AGV_COUNT[i])
    avg_c_botht = average_runs(data_c_botht)
    min_c_botht = min_of_runs(data_c_botht)
    max_c_botht = max_of_runs(data_c_botht)
    last_c_botht = [data_c_botht[j][-1] for j in range(len(data_c_botht))]

    data_i = read_runs(ALGO + '/I_' + AGV_COUNT[i])
    avg_i = average_runs(data_i)
    min_i = min_of_runs(data_i)
    max_i = max_of_runs(data_i)
    last_i = [data_i[j][-1] for j in range(len(data_i))]

    data_i_t = read_runs(ALGO + '/I_t_' + AGV_COUNT[i])
    avg_i_t = average_runs(data_i_t)
    min_i_t = min_of_runs(data_i_t)
    max_i_t = max_of_runs(data_i_t)
    last_i_t = [data_i_t[j][-1] for j in range(len(data_i_t))]

    data_i_avgt = read_runs(ALGO + '/I_avgt_' + AGV_COUNT[i])
    avg_i_avgt = average_runs(data_i_avgt)
    min_i_avgt = min_of_runs(data_i_avgt)
    max_i_avgt = max_of_runs(data_i_avgt)
    last_i_avgt = [data_i_avgt[j][-1] for j in range(len(data_i_avgt))]

    data_i_botht = read_runs(ALGO + '/I_botht_' + AGV_COUNT[i])
    avg_i_botht = average_runs(data_i_botht)
    min_i_botht = min_of_runs(data_i_botht)
    max_i_botht = max_of_runs(data_i_botht)
    last_i_botht = [data_i_botht[j][-1] for j in range(len(data_i_botht))]

    data_l = read_runs(ALGO + '/L_' + AGV_COUNT[i])
    avg_l = average_runs(data_l)
    min_l = min_of_runs(data_l)
    max_l = max_of_runs(data_l)
    last_l = [data_l[j][-1] for j in range(len(data_l))]

    data_l_t = read_runs(ALGO + '/L_t_' + AGV_COUNT[i])
    avg_l_t = average_runs(data_l_t)
    min_l_t = min_of_runs(data_l_t)
    max_l_t = max_of_runs(data_l_t)
    last_l_t = [data_l_t[j][-1] for j in range(len(data_l_t))]

    data_l_avgt = read_runs(ALGO + '/L_avgt_' + AGV_COUNT[i])
    avg_l_avgt = average_runs(data_l_avgt)
    min_l_avgt = min_of_runs(data_l_avgt)
    max_l_avgt = max_of_runs(data_l_avgt)
    last_l_avgt = [data_l_avgt[j][-1] for j in range(len(data_l_avgt))]

    data_l_botht = read_runs(ALGO + '/L_botht_' + AGV_COUNT[i])
    avg_l_botht = average_runs(data_l_botht)
    min_l_botht = min_of_runs(data_l_botht)
    max_l_botht = max_of_runs(data_l_botht)
    last_l_botht = [data_l_botht[j][-1] for j in range(len(data_l_botht))]

    linewidth = 1.5
    markersize = 2
    
    if PLOT_TYPE == 'average':
        axis[i].set_ylim(Y_LIM_AVE[i][0], Y_LIM_AVE[i][1])
        axis[i].set_xlim(0,500)
        axis[i].set_xlabel('Epoch')
        axis[i].plot(avg_c, label='Centralized', color='green', linewidth=linewidth, marker=".", markersize=markersize)
        axis[i].plot(avg_c_t, label='Centralized, last time', color='cyan', linewidth=linewidth, marker='o', markersize=markersize)
        axis[i].plot(avg_c_avgt, label='Centralized, avg time', color='orange', linewidth=linewidth, marker='*', markersize=markersize)
        axis[i].plot(avg_c_botht, label='Centralized, both times', color='brown', linewidth=linewidth, marker='p', markersize=markersize)
        axis[i].plot(avg_i, ls='dotted', label='Intersection', color='black', linewidth=linewidth, marker='x', markersize=markersize)
        axis[i].plot(avg_i_t, ls='dotted', label='Intersection, last time', color='purple', linewidth=linewidth, marker='3', markersize=markersize)
        axis[i].plot(avg_i_avgt, ls='dotted', label='Intersection, avg time', color='pink', linewidth=linewidth, marker='v', markersize=markersize)
        axis[i].plot(avg_i_botht, ls='dotted', label='Intersection, both times', color='gray', linewidth=linewidth, marker='4', markersize=markersize)
        axis[i].plot(avg_l, ls='dashed', label='Link', color='blue', linewidth=linewidth, marker='h', markersize=markersize)
        axis[i].plot(avg_l_t, ls='dashed', label='Link, time', color='red', linewidth=linewidth, marker='+', markersize=markersize)
        axis[i].plot(avg_l_avgt, ls='dashed', label='Link, avg time', color='olive', linewidth=linewidth, marker='d', markersize=markersize)
        axis[i].plot(avg_l_botht, ls='dashed', label='Link, both times', color='black', linewidth=linewidth, marker='|', markersize=markersize)
        #axis[i].fill_between(x=x, y1=min_c, y2=max_c, color='#90ee90')
        #axis[i].fill_between(x=x, y1=min_c_t, y2=max_c_t, color='#e0ffff')
        #axis[i].fill_between(x=x, y1=min_i, y2=max_c, color='#ffffe0')
        #axis[i].fill_between(x=x, y1=min_i_t, y2=max_c_t, color='#b695c0')
        #axis[i].fill_between(x=x, y1=min_l, y2=max_c, color='#add8e6')
        #axis[i].fill_between(x=x, y1=min_l_t, y2=max_c_t, color='#ff7276')
        # change markers of plot lines
        
        
        axis[i].set_title(AGV_COUNT[i] + ' AGVs')
        axis[i].tick_params(which='major', direction='in')                
        axis[0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'max':
        axis[i].set_ylim(Y_LIM_MAX[i][0], Y_LIM_MAX[i][1])
        axis[i].set_xlim(0,500)
        axis[i].set_xlabel('Epoch')
        axis[i].plot(max_c, label='Centralized', color='green', linewidth=linewidth, marker=".", markersize=markersize)
        axis[i].plot(max_c_t, label='Centralized, last time', color='cyan', linewidth=linewidth, marker='o', markersize=markersize)
        axis[i].plot(max_c_avgt, label='Centralized, avg time', color='orange', linewidth=linewidth, marker='*', markersize=markersize)
        axis[i].plot(max_c_botht, label='Centralized, both times', color='brown', linewidth=linewidth, marker='p', markersize=markersize)
        axis[i].plot(max_i, ls='dotted', label='Intersection', color='black', linewidth=linewidth, marker='x', markersize=markersize)
        axis[i].plot(max_i_t, ls='dotted', label='Intersection, last time', color='purple', linewidth=linewidth, marker='3', markersize=markersize)
        axis[i].plot(max_i_avgt, ls='dotted', label='Intersection, avg time', color='pink', linewidth=linewidth, marker='v', markersize=markersize)
        axis[i].plot(max_i_botht, ls='dotted', label='Intersection, both times', color='gray', linewidth=linewidth, marker='4', markersize=markersize)
        axis[i].plot(max_l, ls='dashed', label='Link', color='blue', linewidth=linewidth, marker='h', markersize=markersize)
        axis[i].plot(max_l_t, ls='dashed', label='Link, time', color='red', linewidth=linewidth, marker='+', markersize=markersize)
        axis[i].plot(max_l_avgt, ls='dashed', label='Link, avg time', color='olive', linewidth=linewidth, marker='d', markersize=markersize)
        axis[i].plot(max_l_botht, ls='dashed', label='Link, both times', color='black', linewidth=linewidth, marker='|', markersize=markersize)
        axis[i].set_title(AGV_COUNT[i] + ' AGVs')
        axis[i].tick_params(which='major', direction='in')
        axis[0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'violin':
        axis[int(i/2)][i%2].set_ylim(Y_LIM_AVE[i][0], Y_LIM_MAX[i][1])
        axis[int(i/2)][i%2].violinplot([last_l, last_l_t, last_l_avgt, last_l_botht, last_i, last_i_t, last_i_avgt, last_i_botht, last_c, last_c_t, last_c_avgt, last_c_botht], showextrema = False)
        axis[int(i/2)][i%2].collections[0].set_facecolor('blue')
        axis[int(i/2)][i%2].collections[1].set_facecolor('red')
        axis[int(i/2)][i%2].collections[2].set_facecolor('olive')
        axis[int(i/2)][i%2].collections[3].set_facecolor('black')
        axis[int(i/2)][i%2].collections[4].set_facecolor('black')
        axis[int(i/2)][i%2].collections[5].set_facecolor('purple')
        axis[int(i/2)][i%2].collections[6].set_facecolor('pink')
        axis[int(i/2)][i%2].collections[7].set_facecolor('gray')
        axis[int(i/2)][i%2].collections[8].set_facecolor('green')
        axis[int(i/2)][i%2].collections[9].set_facecolor('cyan')
        axis[int(i/2)][i%2].collections[10].set_facecolor('orange')
        axis[int(i/2)][i%2].collections[11].set_facecolor('brown')

        # add markers
        axis[int(i/2)][i%2].scatter(1, statistics.median(last_l), marker='x', color='black', zorder=1,s=40, label='median')
        axis[int(i/2)][i%2].scatter(2, statistics.median(last_l_t), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(3, statistics.median(last_l_avgt), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(4, statistics.median(last_l_botht), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(5, statistics.median(last_i), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(6, statistics.median(last_i_t), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(7, statistics.median(last_i_avgt), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(8, statistics.median(last_i_botht), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(9, statistics.median(last_c), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(10, statistics.median(last_c_t), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(11, statistics.median(last_c_avgt), marker='x', color='black', zorder=1,s=40)
        axis[int(i/2)][i%2].scatter(12, statistics.median(last_c_botht), marker='x', color='black', zorder=1,s=40)

        axis[int(i/2)][i%2].scatter(1, statistics.mean(last_l), marker='+', color='black', zorder=1, s=80, label='mean')
        axis[int(i/2)][i%2].scatter(2, statistics.mean(last_l_t), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(3, statistics.mean(last_l_avgt), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(4, statistics.mean(last_l_botht), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(5, statistics.mean(last_i), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(6, statistics.mean(last_i_t), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(7, statistics.mean(last_i_avgt), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(8, statistics.mean(last_i_botht), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(9, statistics.mean(last_c), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(10, statistics.mean(last_c_t), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(11, statistics.mean(last_c_avgt), marker='+', color='black', zorder=1,s=80)
        axis[int(i/2)][i%2].scatter(12, statistics.mean(last_c_botht), marker='+', color='black', zorder=1,s=80)

        X_LABELS = ['','Link','Link last time','Link avg time','Link both times','Intersection','Intersection last time','Intersection avg time','Intersection both times','Centralized','centralized last time','centralized avg time','centralized both times']
        X_LABELS_EMPTY = ['','','','','','','','','','','','','']
        #axis[1][0].set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=X_LABELS, rotation=90, ha="right")
        axis[1][0].set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=X_LABELS, transform_rotates_text=True ,rotation=90, ma="right", ha="center")
        axis[1][1].set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=X_LABELS, rotation=90, ha="right")
        axis[0][i%2].set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12], labels=X_LABELS_EMPTY, rotation=70, ha="right")
        axis[int(i/2)][i%2].set_title(AGV_COUNT[i] + ' AGVs')
        axis[1][0].legend(loc="lower right")
        axis[int(i/2)][i%2].tick_params(which='major', direction='in')
        axis[int(i/2)][0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'print':
        print(ALGO + '_' + AGV_COUNT[i])
        print('max Centralized:' + str(max(max_c)))
        print('max Centralized last time:' + str(max(max_c_t)))
        print('max Centralized avg time:' + str(max(max_c_avgt)))
        print('max Centralized both time:' + str(max(max_c_botht)))
        print('max Intersecton:' + str(max(max_i)))
        print('max Intersecton last time:' + str(max(max_i_t)))
        print('max Intersecton avg time:' + str(max(max_i_avgt)))
        print('max Intersecton both time:' + str(max(max_i_botht)))
        print('max link:' + str(max(max_l)))
        print('max link last time:' + str(max(max_l_t)))
        print('max link avg time:' + str(max(max_l_avgt)))
        print('max link both time:' + str(max(max_l_botht)))

        print('average Centralized:' + str(statistics.mean(max_c)))
        print('average Centralized last time:' + str(statistics.mean(max_c_t)))
        print('average Centralized avg time:' + str(statistics.mean(max_c_avgt)))
        print('average Centralized both time:' + str(statistics.mean(max_c_botht)))
        print('average Intersecton:' + str(statistics.mean(max_i)))
        print('average Intersecton last time:' + str(statistics.mean(max_i_t)))
        print('average Intersecton avg time:' + str(statistics.mean(max_i_avgt)))
        print('average Intersecton both time:' + str(statistics.mean(max_i_botht)))
        print('average link:' + str(statistics.mean(max_l)))
        print('average link last time:' + str(statistics.mean(max_l_t)))
        print('average link avg time:' + str(statistics.mean(max_l_avgt)))
        print('average link both time:' + str(statistics.mean(max_l_botht)))

        print('Worse Centralized:' + str(min(max_c)))
        print('Worse Centralized last time:' + str(min(max_c_t)))
        print('Worse Centralized avg time:' + str(min(max_c_avgt)))
        print('Worse Centralized both time:' + str(min(max_c_botht)))
        print('Worse Intersecton:' + str(min(max_i)))
        print('Worse Intersecton last time:' + str(min(max_i_t)))
        print('Worse Intersecton avg time:' + str(min(max_i_avgt)))
        print('Worse Intersecton both time:' + str(min(max_i_botht)))
        print('Worse link:' + str(min(max_l)))
        print('Worse link last time:' + str(min(max_l_t)))
        print('Worse link avg time:' + str(min(max_l_avgt)))
        print('Worse link both time:' + str(min(max_l_botht)))

        print('standard deviation Centralized:' + str(statistics.stdev(last_c)))
        print('standard deviation Centralized last time:' + str(statistics.stdev(last_c_t)))
        print('standard deviation Centralized avg time:' + str(statistics.stdev(last_c_avgt)))
        print('standard deviation Centralized both time:' + str(statistics.stdev(last_c_botht)))
        print('standard deviation Intersecton:' + str(statistics.stdev(last_i)))
        print('standard deviation Intersecton last time:' + str(statistics.stdev(last_i_t)))
        print('standard deviation Intersecton avg time:' + str(statistics.stdev(last_i_avgt)))
        print('standard deviation Intersecton both time:' + str(statistics.stdev(last_i_botht)))
        print('standard deviation link:' + str(statistics.stdev(last_l)))
        print('standard deviation link last time:' + str(statistics.stdev(last_l_t)))
        print('standard deviation link avg time:' + str(statistics.stdev(last_l_avgt)))
        print('standard deviation link both time:' + str(statistics.stdev(last_l_botht)))

        print('dispersion Centralized:' + str(statistics.variance(last_c) / statistics.mean(last_c)))
        print('dispersion Centralized last time:' + str(statistics.variance(last_c_t) / statistics.mean(last_c_t)))
        print('dispersion Centralized avg time:' + str(statistics.variance(last_c_avgt) / statistics.mean(last_c_avgt)))
        print('dispersion Centralized both time:' + str(statistics.variance(last_c_botht) / statistics.mean(last_c_botht)))
        print('dispersion Intersecton:' + str(statistics.variance(last_i) / statistics.mean(last_i)))
        print('dispersion Intersecton last time:' + str(statistics.variance(last_i_t) / statistics.mean(last_i_t)))
        print('dispersion Intersecton avg time:' + str(statistics.variance(last_i_avgt) / statistics.mean(last_i_avgt)))
        print('dispersion Intersecton both time:' + str(statistics.variance(last_i_botht) / statistics.mean(last_i_botht)))
        print('dispersion link:' + str(statistics.variance(last_l) / statistics.mean(last_l)))
        print('dispersion link last time:' + str(statistics.variance(last_l_t) / statistics.mean(last_l_t)))
        print('dispersion link avg time:' + str(statistics.variance(last_l_avgt) / statistics.mean(last_l_avgt)))
        print('dispersion link both time:' + str(statistics.variance(last_l_botht) / statistics.mean(last_l_botht)))

        print('skew Centralized:' + str(skew(last_c)))
        print('skew Centralized last time:' + str(skew(last_c_t)))
        print('skew Centralized avg time:' + str(skew(last_c_avgt)))
        print('skew Centralized both time:' + str(skew(last_c_botht)))
        print('skew Intersecton:' + str(skew(last_i)))
        print('skew Intersecton last time:' + str(skew(last_i_t)))
        print('skew Intersecton avg time:' + str(skew(last_i_avgt)))
        print('skew Intersecton both time:' + str(skew(last_i_botht)))
        print('skew link:' + str(skew(last_l)))
        print('skew link last time:' + str(skew(last_l_t)))
        print('skew link avg time:' + str(skew(last_l_avgt)))
        print('skew link both time:' + str(skew(last_l_botht)))
    else:
        print('invalid PLOT_TYPE')
        exit()

if PLOT_TYPE != 'violin':
    #axis[1].legend(loc='upper center', bbox_to_anchor=(1, -0.175), fancybox=True, ncol=6)
    axis[3].legend(loc='upper center', bbox_to_anchor=(0.50, -0.20), fancybox=True, ncol=4, fontsize="x-large")
    fig.set_figwidth(10*1.3)
    fig.set_figheight(14.1*1.3)
else:
    
    fig.set_figwidth(11)
    fig.set_figheight(9)

if PLOT_TYPE == 'average':
    matplotlib.pyplot.savefig(ALGO + '_ave_perf_4x1.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_ave_perf_4x1.png',bbox_inches='tight')
elif PLOT_TYPE == 'max':
    matplotlib.pyplot.savefig(ALGO + '_max_perf_4x1.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_max_perf_4x1.png',bbox_inches='tight')
elif PLOT_TYPE == 'violin':
    matplotlib.pyplot.savefig(ALGO + '_violin.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_violin.png',bbox_inches='tight')

#if PLOT_TYPE != 'print':
    #matplotlib.pyplot.show()
