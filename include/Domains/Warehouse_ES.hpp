#pragma once

#include <vector>
#include <list>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Agents/ESAgent.hpp" 

#define STD_DEV 1

class Warehouse_ES : public Warehouse {
	public:
		Warehouse_ES(YAML::Node configs) : Warehouse(configs){
			N_proc_std_dev = configs["DDPG"]["rand_proc_std_dev"].as<float>();
		}
		~Warehouse_ES(void);

		epoch_results SimulateEpoch(bool verbose, int epoch){}
		epoch_resultsES SimulateEpochES (const int epoch = 1, bool verbose = false);

		void InitialiseMATeam(); // create agents for each vertex in graph
		std::vector<float> QueryActorMATeam(std::vector<float> states);
		void initialiseNNWeights(std::vector<esNN> v);

		void set_team_NNs(std::vector<esNN*> teamNNs);

		std::vector<esNN*> produce_random_team_NNs();
	protected:
		float N_proc_std_dev;
		std::vector<ESAgent *> maTeam;		
		std::vector<esNN*> produce_random_team_NNs(agent_def agent_type);
};