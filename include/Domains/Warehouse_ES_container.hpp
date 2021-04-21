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

#define LEARNING_RATE 0.01

class Warehouse_ES_container {
	public:
		Warehouse_ES_container(YAML::Node configs);
		~Warehouse_ES_container(void){}

		uint evolution_strategy(size_t n_threads=1, bool verbose=false);
	protected:
		std::vector<Warehouse_ES*> population; 
		int epoch;
		float learning_rate;
};