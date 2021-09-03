#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Planning/Search.h"
#include "Agents/experience_replay.hpp"
#include "Agents/COMAAgent.hpp"

class Warehouse_COMA : public Warehouse {
	public:
		Warehouse_COMA(YAML::Node configs) : Warehouse(configs), N_proc_std_dev(0){
			N_proc_std_dev = configs["COMA"]["rand_proc_std_dev"].as<float>();
			//DDPGAgent::set_batch_size(configs["DDPG"]["batch_size"].as<uint>());
		}
		~Warehouse_COMA(void);

		virtual epoch_results simulate_epoch_COMA (bool verbose, int epoch = 1);

		void InitialiseMATeam(); // create agents for each vertex in graph
	protected:
		std::vector<float> query_actor_MATeam(std::vector<float> states) __attribute__ ((pure));

		float N_proc_std_dev;
		std::vector<COMAAgent*> maTeam;
};