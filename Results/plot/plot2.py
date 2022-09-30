#author: Kallinteris Andreas

import matplotlib.pyplot
import os
import sys
import statistics
from scipy.stats import skew


def read_runs(run_path):
    data_g = []
    files = os.listdir(run_path) 
    for file in files:
        if not file.endswith('.csv') and not file.endswith('.yaml'):
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


#AGV_COUNT = ['90', '120', '200', '400'] # number of Autonomous Ground Vehicles (AGVs)
AGV_COUNT = ['90', '120'] # number of Autonomous Ground Vehicles (AGVs)
#AGV_COUNT = ['200', '400'] # number of Autonomous Ground Vehicles (AGVs)
ALGO = 'ES'
#ALGO = 'CCEA'

Y_LIM_AVE = [(400, 550), (400, 650), (400, 800), (450, 850)]
#Y_LIM_AVE = [(400, 800), (450, 850)]
Y_LIM_MAX = [(450, 580), (450, 750), (450, 950), (500, 950)]
#Y_LIM_MAX = [(450, 950), (500, 950)]
if PLOT_TYPE == 'violin':
    fig, axis = matplotlib.pyplot.subplots(2, 1)
    #fig, axis = matplotlib.pyplot.subplots(1,len(AGV_COUNT))
else:
    fig, axis = matplotlib.pyplot.subplots(1,len(AGV_COUNT))

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


    if PLOT_TYPE == 'average':
        axis[i].set_ylim(Y_LIM_AVE[i][0], Y_LIM_AVE[i][1])
        axis[i].set_xlim(0,500)
        axis[i].set_xlabel('Epoch')
        axis[i].plot(avg_c, label='Centralized', color='green', linewidth=0.5)
        axis[i].plot(avg_c_t, label='Centralized, time', color='cyan', linewidth=0.5)
        axis[i].plot(avg_i, label='Intersection', color='yellow', linewidth=0.5)
        axis[i].plot(avg_i_t, label='Intersection, time', color='purple', linewidth=0.5)
        axis[i].plot(avg_l, label='Link', color='blue', linewidth=0.5)
        axis[i].plot(avg_l_t, label='Link, time', color='red', linewidth=0.5)
        #axis[i].fill_between(x=x, y1=min_c, y2=max_c, color='#90ee90')
        #axis[i].fill_between(x=x, y1=min_c_t, y2=max_c_t, color='#e0ffff')
        #axis[i].fill_between(x=x, y1=min_i, y2=max_c, color='#ffffe0')
        #axis[i].fill_between(x=x, y1=min_i_t, y2=max_c_t, color='#b695c0')
        #axis[i].fill_between(x=x, y1=min_l, y2=max_c, color='#add8e6')
        #axis[i].fill_between(x=x, y1=min_l_t, y2=max_c_t, color='#ff7276')
        axis[i].set_title(AGV_COUNT[i] + ' AGVs')
        axis[i].tick_params(which='major', direction='in')
        axis[0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'max':
        axis[i].set_ylim(Y_LIM_MAX[i][0], Y_LIM_MAX[i][1])
        axis[i].set_xlim(0,500)
        axis[i].set_xlabel('Epoch')
        axis[i].plot(max_c, label='Centralized', color='green', linewidth=0.5)
        axis[i].plot(max_c_t, label='Centralized, time', color='cyan', linewidth=0.5)
        axis[i].plot(max_i, label='Intersection', color='yellow', linewidth=0.5)
        axis[i].plot(max_i_t, label='Intersection, time', color='purple', linewidth=0.5)
        axis[i].plot(max_l, label='Link', color='blue', linewidth=0.5)
        axis[i].plot(max_l_t, label='Link, time', color='red', linewidth=0.5)
        axis[i].set_title(AGV_COUNT[i] + ' AGVs')
        axis[i].tick_params(which='major', direction='in')
        axis[0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'violin':
        axis[i%2].set_ylim(Y_LIM_AVE[i][0], Y_LIM_MAX[i][1])
        axis[i%2].violinplot([last_l, last_l_t, last_i, last_i_t, last_c, last_c_t], showextrema = False)
        axis[i%2].collections[0].set_facecolor('blue')
        axis[i%2].collections[1].set_facecolor('red')
        axis[i%2].collections[2].set_facecolor('orange')
        axis[i%2].collections[3].set_facecolor('purple')
        axis[i%2].collections[4].set_facecolor('green')
        axis[i%2].collections[5].set_facecolor('cyan')

        axis[i%2].scatter(1, statistics.median(last_l), marker='x', color='black', zorder=1,s=40, label='median')
        axis[i%2].scatter(2, statistics.median(last_l_t), marker='x', color='black', zorder=1,s=40)
        axis[i%2].scatter(3, statistics.median(last_i), marker='x', color='black', zorder=1,s=40)
        axis[i%2].scatter(4, statistics.median(last_i_t), marker='x', color='black', zorder=1,s=40)
        axis[i%2].scatter(5, statistics.median(last_c), marker='x', color='black', zorder=1,s=40)
        axis[i%2].scatter(6, statistics.median(last_c_t), marker='x', color='black', zorder=1,s=40)

        axis[i%2].scatter(1, statistics.mean(last_l), marker='+', color='black', zorder=1, s=80, label='mean')
        axis[i%2].scatter(2, statistics.mean(last_l_t), marker='+', color='black', zorder=1,s=80)
        axis[i%2].scatter(3, statistics.mean(last_i), marker='+', color='black', zorder=1,s=80)
        axis[i%2].scatter(4, statistics.mean(last_i_t), marker='+', color='black', zorder=1,s=80)
        axis[i%2].scatter(5, statistics.mean(last_c), marker='+', color='black', zorder=1,s=80)
        axis[i%2].scatter(6, statistics.mean(last_c_t), marker='+', color='black', zorder=1,s=80)

        X_LABELS = ['','Link','Link time','Intersection','Intersection Time','Centralized','centralized Time']
        X_LABELS_EMPTY = ['','','','','','','']
        #axis[0].set_xticklabels(X_LABELS, rotation = 45, ha="right")
        axis[1].set_xticklabels(X_LABELS, rotation = 45, ha="right")
        axis[0].set_xticklabels(X_LABELS_EMPTY, rotation = 45, ha="right")
        axis[i].set_title(ALGO + ' ' + AGV_COUNT[i] + ' AGVs')
        axis[0].legend()
        axis[i].tick_params(which='major', direction='in')
        axis[0].set_ylabel('Total Deliveries')
    elif PLOT_TYPE == 'print':
        print(ALGO + '_' + AGV_COUNT[i])
        print('max Centralized:' + str(max(max_c)))
        print('max Centralized time:' + str(max(max_c_t)))
        print('max Intersecton:' + str(max(max_i)))
        print('max Intersecton time:' + str(max(max_i_t)))
        print('max link:' + str(max(max_l)))
        print('max link time:' + str(max(max_l_t)))

        print('average Centralized:' + str(statistics.mean(max_c)))
        print('average Centralized time:' + str(statistics.mean(max_c_t)))
        print('average Intersecton:' + str(statistics.mean(max_i)))
        print('average Intersecton time:' + str(statistics.mean(max_i_t)))
        print('average link:' + str(statistics.mean(max_l)))
        print('average link time:' + str(statistics.mean(max_l_t)))

        print('standard deviation Centralized:' + str(statistics.stdev(last_c)))
        print('standard deviation Centralized time:' + str(statistics.stdev(last_c_t)))
        print('standard deviation Intersecton:' + str(statistics.stdev(last_i)))
        print('standard deviation Intersecton time:' + str(statistics.stdev(last_i_t)))
        print('standard deviation link:' + str(statistics.stdev(last_l)))
        print('standard deviation link time:' + str(statistics.stdev(last_l_t)))

        print('dispersion Centralized:' + str(statistics.variance(last_c) / statistics.mean(last_c)))
        print('dispersion Centralized time:' + str(statistics.variance(last_c_t) / statistics.mean(last_c_t)))
        print('dispersion Intersecton:' + str(statistics.variance(last_i) / statistics.mean(last_i)))
        print('dispersion Intersecton time:' + str(statistics.variance(last_i_t) / statistics.mean(last_i_t)))
        print('dispersion link:' + str(statistics.variance(last_l) / statistics.mean(last_l)))
        print('dispersion link time:' + str(statistics.variance(last_l_t) / statistics.mean(last_l_t)))

        print('skew Centralized:' + str(skew(last_c)))
        print('skew Centralized time:' + str(skew(last_c_t)))
        print('skew Intersecton:' + str(skew(last_i)))
        print('skew Intersecton time:' + str(skew(last_i_t)))
        print('skew link:' + str(skew(last_l)))
        print('skew link time:' + str(skew(last_l_t)))
    else:
        print('invalid PLOT_TYPE')
        exit()

if PLOT_TYPE != 'violin':
    #axis[1].legend(loc='upper center', bbox_to_anchor=(1, -0.175), fancybox=True, ncol=6)
    #axis[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=6)
    fig.set_figwidth(1)
    fig.set_figheight(5)
else:
    fig.set_figwidth(4.5)
    fig.set_figheight(9)

if PLOT_TYPE == 'average':
    matplotlib.pyplot.savefig(ALGO + '_ave_perf.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_ave_perf.png',bbox_inches='tight')
elif PLOT_TYPE == 'max':
    matplotlib.pyplot.savefig(ALGO + '_max_perf.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_max_perf.png',bbox_inches='tight')
elif PLOT_TYPE == 'violin':
    matplotlib.pyplot.savefig(ALGO + '_' + AGV_COUNT[0] + '_' + AGV_COUNT[1] + '_violin.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig(ALGO + '_' + AGV_COUNT[0] + '_' + AGV_COUNT[1] + '_violin.png',bbox_inches='tight')

#if PLOT_TYPE != 'print':
    #matplotlib.pyplot.show()
