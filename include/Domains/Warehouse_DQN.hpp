#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Planning/Search.h"
#include "Agents/experience_replay.hpp"
#include "Agents/DQNAgent.hpp"
#include <time.h>
#include <iostream>

class Warehouse_DQN : public Warehouse {
	public:
		Warehouse_DQN(YAML::Node configs) : Warehouse(configs), N_proc_std_dev(0){
			N_proc_std_dev = configs["DQN"]["rand_proc_std_dev"].as<float>();
			//DQNAgent::set_batch_size(configs["COMA"]["batch_size"].as<uint>());
		}
		~Warehouse_DQN(void);

		virtual epoch_results simulate_epoch_DQN(bool verbose);


		void InitialiseMATeam(); // create agents for each vertex in graph
	protected:
		std::vector<float> query_actor_MATeam(std::vector<float> &states, bool training);
		epoch_results evaluateEpoch();
		float N_proc_std_dev;
		std::vector<DQNAgent*> maTeam;
};