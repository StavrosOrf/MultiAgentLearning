#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include <iostream>
#include "Warehouse.hpp"
#include "Warehouse_ES.hpp"
#include "Agents/ESAgent.hpp" 
#include "boost/asio.hpp"
#include <time.h>
#include <chrono>
#include <ctime>
#include <math.h>


class Warehouse_ES_container {
	public:
		Warehouse_ES_container(YAML::Node configs);
		~Warehouse_ES_container(void){
			for (Warehouse_ES* w : population){
				delete w;
				w = 0;
			}
		}

		uint evolution_strategy(const size_t n_threads, bool verbose, size_t run, std::ofstream &file);
		uint evolution_strategy_ADAM(const size_t n_threads, bool verbose, size_t run, std::ofstream &file);

		void copy_best_team_policy(std::vector<esNN*> sourceNNs,std::vector<esNN*> targetNNs);
		void save_best_team_policy(std::vector<esNN*> teamNNs,int epoch,int G);
		void load_best_team_policy(std::vector<esNN*> teamNNs);
	protected:
		std::vector<Warehouse_ES*> population; 
		int epoch;
		float learning_rate;
		float N_proc_std_dev;
		float b1;
		float b2;
		//std::ofstream* file;

};