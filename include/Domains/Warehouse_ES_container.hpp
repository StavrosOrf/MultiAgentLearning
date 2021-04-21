#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Warehouse_ES.hpp"
#include "Agents/ESAgent.hpp" 

#define POP_SIZE 100

// #define 

class Warehouse_ES_container {
	public:
		Warehouse_ES_container(YAML::Node configs);
		~Warehouse_ES_container(void){}

		void evolution_strategy(bool verbose=false);

	protected:

		std::vector<Warehouse_ES*> population; 
		int epoch = 10;



};