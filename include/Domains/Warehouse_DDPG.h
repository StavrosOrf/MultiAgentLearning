#ifndef WAREHOUSE_CENTRALISED_H_
#define WAREHOUSE_CENTRALISED_H_

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

		epoch_results SimulateEpoch(bool verbose = false);

		void InitialiseMATeam(); // create agents for each vertex in graph
	private:
		std::vector<float> QueryActorMATeam(std::vector<float> states);
		std::vector<float> QueryTargetActorMATeam(std::vector<float> states);
		float N_proc_std_dev;
		vector<DDPGAgent *> ddpg_maTeam;
};

#endif // WAREHOUSE_CENTRALISED_H_
