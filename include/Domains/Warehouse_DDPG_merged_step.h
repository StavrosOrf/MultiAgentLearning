#ifndef WAREHOUSE_DDPG_MERGED_H
#define WAREHOUSE_DDPG_MERGED_H

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.h"
#include "Warehouse_DDPG.h"
#include "Planning/Search.h"
#include "Agents/DDPGAgent.h"


class Warehouse_DDPG_merged_step : public Warehouse_DDPG {
	public:
		Warehouse_DDPG_merged_step(YAML::Node configs) : Warehouse_DDPG(configs){}
		~Warehouse_DDPG_merged_step(void);

		epoch_results SimulateEpoch (bool verbose = false,int epoch = 1);
	private:
};

#endif // WAREHOUSE_DDPG_MERGED_H
