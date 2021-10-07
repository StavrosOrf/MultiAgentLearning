import matplotlib.pyplot
import os
import sys

#Author: Kallinteris Andreas
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
    #for i in range(len(maximum)):
        #if i != 0:
            #if maximum[int(i-1)] > maximum[int(i)]:
                #maximum[int(i)]  = maximum[int(i-1)]
    return maximum


if sys.argv[1] == '--max':
    PLOT_TYPE = 'max' #possible values 'max', 'average'
if sys.argv[1] == '--ave':
    PLOT_TYPE = 'average' #possible values 'max', 'average'
AGV_COUNT = ['90', '120', '200'] # number of Autonomous Ground Vehicles (AGVs)

Y_LIM_AVE = [(350, 525), (400, 650), (400, 800)]
Y_LIM_MAX = [(450, 560), (425, 725), (450, 950)]

fig, axis = matplotlib.pyplot.subplots(1,len(AGV_COUNT))

for i in range(len(AGV_COUNT)):
    data_c_es = read_runs('ES/C_' + AGV_COUNT[i])
    avg_c_es = average_runs(data_c_es)
    x = [*range(0,len(avg_c_es),1)]
    min_c_es = min_of_runs(data_c_es)
    max_c_es = max_of_runs(data_c_es)

    data_c_t_es = read_runs('ES/C_t_' + AGV_COUNT[i])
    avg_c_t_es = average_runs(data_c_t_es)
    min_c_t_es = min_of_runs(data_c_t_es)
    max_c_t_es = max_of_runs(data_c_t_es)

    data_i_es = read_runs('ES/I_' + AGV_COUNT[i])
    avg_i_es = average_runs(data_i_es)
    min_i_es = min_of_runs(data_i_es)
    max_i_es = max_of_runs(data_i_es)

    data_i_t_es = read_runs('ES/I_t_' + AGV_COUNT[i])
    avg_i_t_es = average_runs(data_i_t_es)
    min_i_t_es = min_of_runs(data_i_t_es)
    max_i_t_es = max_of_runs(data_i_t_es)

    data_l_es = read_runs('ES/L_' + AGV_COUNT[i])
    avg_l_es = average_runs(data_l_es)
    min_l_es = min_of_runs(data_l_es)
    max_l_es = max_of_runs(data_l_es)

    data_l_t_es = read_runs('ES/L_t_' + AGV_COUNT[i])
    avg_l_t_es = average_runs(data_l_t_es)
    min_l_t_es = min_of_runs(data_l_t_es)
    max_l_t_es = max_of_runs(data_l_t_es)

    data_c_ga = read_runs('CCEA/C_' + AGV_COUNT[i])
    avg_c_ga = average_runs(data_c_ga)
    min_c_ga = min_of_runs(data_c_ga)
    max_c_ga = max_of_runs(data_c_ga)

    data_c_t_ga = read_runs('CCEA/C_t_' + AGV_COUNT[i])
    avg_c_t_ga = average_runs(data_c_t_ga)
    min_c_t_ga = min_of_runs(data_c_t_ga)
    max_c_t_ga = max_of_runs(data_c_t_ga)

    data_i_ga = read_runs('CCEA/I_' + AGV_COUNT[i])
    avg_i_ga = average_runs(data_i_ga)
    min_i_ga = min_of_runs(data_i_ga)
    max_i_ga = max_of_runs(data_i_ga)

    data_i_t_ga = read_runs('CCEA/I_t_' + AGV_COUNT[i])
    avg_i_t_ga = average_runs(data_i_t_ga)
    min_i_t_ga = min_of_runs(data_i_t_ga)
    max_i_t_ga = max_of_runs(data_i_t_ga)

    data_l_ga = read_runs('CCEA/L_' + AGV_COUNT[i])
    avg_l_ga = average_runs(data_l_ga)
    min_l_ga = min_of_runs(data_l_ga)
    max_l_ga = max_of_runs(data_l_ga)

    data_l_t_ga = read_runs('CCEA/L_t_' + AGV_COUNT[i])
    avg_l_t_ga = average_runs(data_l_t_ga)
    min_l_t_ga = min_of_runs(data_l_t_ga)
    max_l_t_ga = max_of_runs(data_l_t_ga)

    if PLOT_TYPE == 'average':
        #centralized
        axis[i].plot(avg_c_es, ls='dotted', label='ES Centralized', linewidth=0.5)
        axis[i].plot(avg_c_t_es, ls='dotted', label='ES Centralized, time', linewidth=0.5)
        axis[i].plot(avg_c_ga, label='CCEA Centralized', linewidth=0.5)
        axis[i].plot(avg_c_t_ga, label='CCEA Centralized, time', linewidth=0.5)
        #Intersection
        axis[i].plot(avg_i_es, ls='dotted', label='MA-ES Intersection', linewidth=0.5)
        axis[i].plot(avg_i_t_es, ls='dotted', label='MA-ES Intersection, time', linewidth=0.5)
        axis[i].plot(avg_i_ga, label='CCEA Intersection', linewidth=0.5)
        axis[i].plot(avg_i_t_ga, label='CCEA Intersection, time', linewidth=0.5)
        #Link
        axis[i].plot(avg_l_es, ls='dotted', label='MA-ES Link', linewidth=0.5)
        axis[i].plot(avg_l_t_es, ls='dotted', label='MA-ES Link, time', linewidth=0.5)
        axis[i].plot(avg_l_ga, label='CCEA Link', linewidth=0.5)
        axis[i].plot(avg_l_t_ga, label='CCEA Link, time', linewidth=0.5)
        axis[i].set_ylim(Y_LIM_AVE[i][0], Y_LIM_AVE[i][1])
    elif PLOT_TYPE == 'max':
        #centralized
        axis[i].plot(max_c_es, ls='dotted', label='ES Centralized', linewidth=0.5)
        axis[i].plot(max_c_t_es, ls='dotted', label='ES Centralized, time', linewidth=0.5)
        axis[i].plot(max_c_ga, label='CCEA Centralized', linewidth=0.5)
        axis[i].plot(max_c_t_ga, label='CCEA Centralized, time', linewidth=0.5)
        #Intersection
        axis[i].plot(max_i_es, ls='dotted', label='MA-ES Intersection', linewidth=0.5)
        axis[i].plot(max_i_t_es, ls='dotted', label='MA-ES Intersection, time', linewidth=0.5)
        axis[i].plot(max_i_ga, label='CCEA Intersection', linewidth=0.5)
        axis[i].plot(max_i_t_ga, label='CCEA Intersection, time', linewidth=0.5)
        #Link
        axis[i].plot(max_l_es, ls='dotted', label='MA-ES Link', linewidth=0.5)
        axis[i].plot(max_l_t_es, ls='dotted', label='MA-ES Link, time', linewidth=0.5)
        axis[i].plot(max_l_ga, label='CCEA Link', linewidth=0.5)
        axis[i].plot(max_l_t_ga, label='CCEA Link, time', linewidth=0.5)
        axis[i].set_ylim(Y_LIM_MAX[i][0], Y_LIM_MAX[i][1])
    else:
        print('invalid PLOT_TYPE')
        exit()

    axis[i].set_xlim(0,500)
    axis[i].set_title(AGV_COUNT[i] + ' AGVs')
    axis[i].set_xlabel('Epoch')
    axis[i].tick_params(which='major', direction='in')


fig.set_figwidth(11)
fig.set_figheight(2.25)
axis[0].set_ylabel('Total Deliveries')
if PLOT_TYPE == 'average':
    axis[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.185), fancybox=True, ncol=4)

if PLOT_TYPE == 'average':
    matplotlib.pyplot.savefig('ESvsGA_average.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig('ESvsGA_average.png',bbox_inches='tight')
elif PLOT_TYPE == 'max':
    matplotlib.pyplot.savefig('ESvsGA_max.eps',bbox_inches='tight')
    matplotlib.pyplot.savefig('ESvsGA_max.png',bbox_inches='tight')
#matplotlib.pyplot.show()
