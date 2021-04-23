#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Warehouse_ES.hpp"
#include "Agents/ESAgent.hpp" 
#include "boost/asio.hpp"
#include <time.h>
#include <ctime>

class Warehouse_ES_container {
	public:
		Warehouse_ES_container(YAML::Node configs,std::ofstream*  eval_file);
		~Warehouse_ES_container(void){}

		uint evolution_strategy(size_t n_threads=1, bool verbose=false, size_t run=0);
	protected:
		std::vector<Warehouse_ES*> population; 
		int epoch;
		float learning_rate;
		float N_proc_std_dev;
		std::ofstream* file;

};