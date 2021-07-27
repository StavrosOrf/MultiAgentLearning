import matplotlib.pyplot
import os

#author: Kallinteris Andreas

def read_runs(run_path):
    data_g = []
    files = os.listdir(run_path)
    for file in files:
        if not file.endswith('.csv'):
            if os.path.isfile(os.path.join(run_path, file)):
                f = open(os.path.join(run_path, file),'r')
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
    return maximum



PLOT_TYPE = 'average' #possible values 'max', 'average'

fig, axis = matplotlib.pyplot.subplots(1,4)

agv_count = ['90', '120', '200', '400'] # number of Autonomous Ground Vehicles (AGVs)

for i in range(len(agv_count)):
    data_c = read_runs('C_' + agv_count[i])
    avg_c = average_runs(data_c)
    x = [*range(0,len(avg_c),1)]
    min_c = min_of_runs(data_c)
    max_c = max_of_runs(data_c)

    data_c_t = read_runs('C_t_' + agv_count[i])
    avg_c_t = average_runs(data_c_t)
    min_c_t = min_of_runs(data_c_t)
    max_c_t = max_of_runs(data_c_t)

    data_i = read_runs('I_' + agv_count[i])
    avg_i = average_runs(data_i)
    min_i = min_of_runs(data_i)
    max_i = max_of_runs(data_i)

    data_i_t = read_runs('I_t_' + agv_count[i])
    avg_i_t = average_runs(data_i_t)
    min_i_t = min_of_runs(data_i_t)
    max_i_t = max_of_runs(data_i_t)

    data_l = read_runs('L_' + agv_count[i])
    avg_l = average_runs(data_l)
    min_l = min_of_runs(data_l)
    max_l = max_of_runs(data_l)

    data_l_t = read_runs('L_t_' + agv_count[i])
    avg_l_t = average_runs(data_l_t)
    min_l_t = min_of_runs(data_l_t)
    max_l_t = max_of_runs(data_l_t)

    if PLOT_TYPE == 'average':
        axis[i].plot(avg_c, label='Centralized', color='green')
        axis[i].plot(avg_c_t, label='Centralized, time', color='cyan')
        axis[i].plot(avg_i, label='Intersection', color='yellow')
        axis[i].plot(avg_i_t, label='Intersection, time', color='purple')
        axis[i].plot(avg_l, label='Link', color='blue')
        axis[i].plot(avg_l_t, label='Link, time', color='red')
        axis[i].set_ylim(200,900)
        #axis[i].fill_between(x=x, y1=min_c, y2=max_c, color='#90ee90')
        #axis[i].fill_between(x=x, y1=min_c_t, y2=max_c_t, color='#e0ffff')
        #axis[i].fill_between(x=x, y1=min_i, y2=max_c, color='#ffffe0')
        #axis[i].fill_between(x=x, y1=min_i_t, y2=max_c_t, color='#b695c0')
        #axis[i].fill_between(x=x, y1=min_l, y2=max_c, color='#add8e6')
        #axis[i].fill_between(x=x, y1=min_l_t, y2=max_c_t, color='#ff7276')
    elif PLOT_TYPE == 'max':
        axis[i].plot(max_c, label='Centralized', color='green')
        axis[i].plot(max_c_t, label='Centralized, time', color='cyan')
        axis[i].plot(max_i, label='Intersection', color='yellow')
        axis[i].plot(max_i_t, label='Intersection, time', color='purple')
        axis[i].plot(max_l, label='Link', color='blue')
        axis[i].plot(max_l_t, label='Link, time', color='red')
        axis[i].set_ylim(200,1000)
    else:
        print('invalid PLOT_TYPE')
        exit()
    axis[i].set_xlim(0,500)
    axis[i].set_title(agv_count[i] + ' AGVs')
    axis[i].set_ylabel('Total Deliveries')
    axis[i].set_xlabel('Epoch')
    axis[i].tick_params(which='major', direction='in')

axis[0].legend()
matplotlib.pyplot.show()
