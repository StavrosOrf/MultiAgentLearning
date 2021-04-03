#ifndef WAREHOUSE_CENTRALISED_H_
#define WAREHOUSE_CENTRALISED_H_

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.h"
#include "Planning/Search.h"


class WarehouseCentralised : public Warehouse {
	public:
		WarehouseCentralised(YAML::Node configs) : Warehouse(configs){}
		~WarehouseCentralised(void);

		epoch_results SimulateEpochDDPG(bool verbose = false);

		void InitialiseMATeam(); // create agents for each vertex in graph
		std::vector<float> QueryActorMATeam(std::vector<float> states);
		std::vector<float> QueryTargetActorMATeam(std::vector<float> states);
	private:

};

#endif // WAREHOUSE_CENTRALISED_H_
