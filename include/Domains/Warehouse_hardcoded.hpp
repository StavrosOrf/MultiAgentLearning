#pragma once

#include <vector>
#include <list>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"

#define STD_DEV 1

class Warehouse_hardcoded: public Warehouse {
	public:
	Warehouse_hardcoded(YAML::Node configs) : Warehouse(configs){
			}
		~Warehouse_hardcoded(void);

		epoch_results simulate_epoch (bool verbose = false);
                void InitialiseMATeam(){}
	protected:
};