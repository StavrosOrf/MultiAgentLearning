#ifndef WAREHOUSE_CENTRALISED_H_
#define WAREHOUSE_CENTRALISED_H_

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.h"
#include "Agents/DDPGAgent.h"
#include "Planning/Search.h"


class WarehouseCentralised : public Warehouse {
	public:
		WarehouseCentralised(YAML::Node configs) : Warehouse(configs){}
		~WarehouseCentralised(void);

		epoch_results SimulateEpochDDPG(bool verbose = false);

		void InitialiseMATeam(); // create agents for each vertex in graph
	private:
};

#endif // WAREHOUSE_CENTRALISED_H_
