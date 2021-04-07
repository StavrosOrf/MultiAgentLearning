#ifndef WAREHOUSE_DDPG_H
#define WAREHOUSE_DDPG_H

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.h"
#include "Planning/Search.h"
#include "Agents/DDPGAgent.h"


class Warehouse_DDPG : public Warehouse {
	public:
		Warehouse_DDPG(YAML::Node configs) : Warehouse(configs){
			N_proc_std_dev = configs["DDPG"]["rand_proc_std_dev"].as<float>();
			DDPGAgent::set_batch_size(configs["DDPG"]["batch_size"].as<uint>());
		}
		~Warehouse_DDPG(void);

		virtual epoch_results SimulateEpoch (bool verbose = false, int epoch = 1);

		void InitialiseMATeam(); // create agents for each vertex in graph
	protected:
		std::vector<float> QueryActorMATeam(std::vector<float> states) __attribute__ ((pure));
		std::vector<float> QueryTargetActorMATeam(std::vector<float> states) __attribute__ ((pure));
		float N_proc_std_dev;
		std::vector<DDPGAgent *> ddpg_maTeam;
};

#endif // WAREHOUSE_DDPG_H
