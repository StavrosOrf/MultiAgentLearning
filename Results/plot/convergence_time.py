import os
import math
#author: Stavros Orfanoudakis

root_folder = "./CCEA/" # ES, CN_ES
convergence_threshold = 0.9
agv_number_list = [90,120,200,400]

for agv_number in agv_number_list:

	for folder in os.listdir(root_folder):
		# print(folder,"--------------")

		if int(folder.split("_")[-1]) != agv_number:
			continue

		number_of_runs = 0
		convergence_time_sum = 0
		
		for file_name in os.listdir(root_folder+folder):
			
			if len(file_name.split(".")) > 1:
				continue

			number_of_runs += 1		
			data_g = []

			f = open(os.path.join(root_folder+folder, file_name),'r')
			#print(f)
			data_g = [float(i) for i in f.read().splitlines()]
			f.close()

			max_value = max(data_g)
			# print(max_value)
			for i,value in enumerate(data_g):

				if value >= max_value*convergence_threshold:
					# print(i)
					convergence_time_sum += i
					break

			# print(data_g)

		print(folder," \n",round(convergence_time_sum/number_of_runs,1))
		print("----------------------------")
		# print(folder," \t\t-------------- Convergence time: ",round(convergence_time_sum/number_of_runs,0), "Number of runs: ", number_of_runs)

		# exit()
