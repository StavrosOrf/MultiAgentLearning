import matplotlib.pyplot
import os

#Author: Kallinteris Andreas

def read_runs(run_path):
    data_g = []
    files = os.listdir(run_path)
    for file in files:
        #if not file.endswith('.csv'):
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



PLOT_TYPE = 'max' #possible values 'max', 'average'

fig, axis = matplotlib.pyplot.subplots(1,3)

agv_count = '120' # number of Autonomous Ground Vehicles (AGVs)

data_c_es = read_runs('ES_C_' + agv_count)
avg_c_es = average_runs(data_c_es)
x = [*range(0,len(avg_c_es),1)]
min_c_es = min_of_runs(data_c_es)
max_c_es = max_of_runs(data_c_es)

data_c_t_es = read_runs('ES_C_t_' + agv_count)
avg_c_t_es = average_runs(data_c_t_es)
min_c_t_es = min_of_runs(data_c_t_es)
max_c_t_es = max_of_runs(data_c_t_es)

data_i_es = read_runs('ES_I_' + agv_count)
avg_i_es = average_runs(data_i_es)
min_i_es = min_of_runs(data_i_es)
max_i_es = max_of_runs(data_i_es)

data_i_t_es = read_runs('ES_I_t_' + agv_count)
avg_i_t_es = average_runs(data_i_t_es)
min_i_t_es = min_of_runs(data_i_t_es)
max_i_t_es = max_of_runs(data_i_t_es)

data_l_es = read_runs('ES_L_' + agv_count)
avg_l_es = average_runs(data_l_es)
min_l_es = min_of_runs(data_l_es)
max_l_es = max_of_runs(data_l_es)

data_l_t_es = read_runs('ES_L_t_' + agv_count)
avg_l_t_es = average_runs(data_l_t_es)
min_l_t_es = min_of_runs(data_l_t_es)
max_l_t_es = max_of_runs(data_l_t_es)

data_c_ga = read_runs('GA_C_' + agv_count)
avg_c_ga = average_runs(data_c_ga)
min_c_ga = min_of_runs(data_c_ga)
max_c_ga = max_of_runs(data_c_ga)

data_c_t_ga = read_runs('GA_C_t_' + agv_count)
avg_c_t_ga = average_runs(data_c_t_ga)
min_c_t_ga = min_of_runs(data_c_t_ga)
max_c_t_ga = max_of_runs(data_c_t_ga)

data_i_ga = read_runs('GA_I_' + agv_count)
avg_i_ga = average_runs(data_i_ga)
min_i_ga = min_of_runs(data_i_ga)
max_i_ga = max_of_runs(data_i_ga)

data_i_t_ga = read_runs('GA_I_t_' + agv_count)
avg_i_t_ga = average_runs(data_i_t_ga)
min_i_t_ga = min_of_runs(data_i_t_ga)
max_i_t_ga = max_of_runs(data_i_t_ga)

data_l_ga = read_runs('GA_L_' + agv_count)
avg_l_ga = average_runs(data_l_ga)
min_l_ga = min_of_runs(data_l_ga)
max_l_ga = max_of_runs(data_l_ga)

data_l_t_ga = read_runs('GA_L_t_' + agv_count)
avg_l_t_ga = average_runs(data_l_t_ga)
min_l_t_ga = min_of_runs(data_l_t_ga)
max_l_t_ga = max_of_runs(data_l_t_ga)

if PLOT_TYPE == 'average':
    #centralized
    axis[0].plot(avg_c_es, label='ES Centralized', color='green')
    axis[0].plot(avg_c_t_es, label='ES Centralized, time', color='cyan')
    axis[0].plot(avg_c_ga, label='GA Centralized', color='yellow')
    axis[0].plot(avg_c_t_ga, label='GA Centralized, time', color='purple')
    axis[0].set_ylim(200,900)
    #Intersection
    axis[1].plot(avg_i_es, label='ES Intersection', color='green')
    axis[1].plot(avg_i_t_es, label='ES Intersection, time', color='cyan')
    axis[1].plot(avg_i_ga, label='GA Intersection', color='yellow')
    axis[1].plot(avg_i_t_ga, label='GA Intersection, time', color='purple')
    axis[1].set_ylim(200,900)
    #Link
    axis[2].plot(avg_l_es, label='ES Link', color='green')
    axis[2].plot(avg_l_t_es, label='ES Link, time', color='cyan')
    axis[2].plot(avg_l_ga, label='GA Link', color='yellow')
    axis[2].plot(avg_l_t_ga, label='GA Link, time', color='purple')
    axis[2].set_ylim(200,900)
elif PLOT_TYPE == 'max':
    #centralized
    axis[0].plot(max_c_es, label='ES Centralized', color='green')
    axis[0].plot(max_c_t_es, label='ES Centralized, time', color='cyan')
    axis[0].plot(max_c_ga, label='GA Centralized', color='yellow')
    axis[0].plot(max_c_t_ga, label='GA Centralized, time', color='purple')
    axis[0].set_ylim(200,1000)
    #Intersection
    axis[1].plot(max_i_es, label='ES Intersection', color='green')
    axis[1].plot(max_i_t_es, label='ES Intersection, time', color='cyan')
    axis[1].plot(max_i_ga, label='GA Intersection', color='yellow')
    axis[1].plot(max_i_t_ga, label='GA Intersection, time', color='purple')
    axis[1].set_ylim(200,1000)
    #Link
    axis[2].plot(max_l_es, label='ES Link', color='green')
    axis[2].plot(max_l_t_es, label='ES Link, time', color='cyan')
    axis[2].plot(max_l_ga, label='GA Link', color='yellow')
    axis[2].plot(max_l_t_ga, label='GA Link, time', color='purple')
    axis[2].set_ylim(200,1000)
else:
    print('invalid PLOT_TYPE')
    exit()
axis[0].set_xlim(0,500)
axis[1].set_xlim(0,500)
axis[2].set_xlim(0,500)
#axis[i].set_title(agv_count[i] + ' AGVs')
axis[0].set_ylabel('Total Deliveries')
axis[1].set_ylabel('Total Deliveries')
axis[2].set_ylabel('Total Deliveries')
axis[0].set_xlabel('Epoch')
axis[1].set_xlabel('Epoch')
axis[2].set_xlabel('Epoch')
axis[0].tick_params(which='major', direction='in')
axis[1].tick_params(which='major', direction='in')
axis[2].tick_params(which='major', direction='in')

axis[0].legend()
axis[1].legend()
axis[2].legend()
matplotlib.pyplot.show()
