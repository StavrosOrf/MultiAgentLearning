#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
#include "Warehouse.hpp"
#include "Agents/ESAgent.hpp" 

#define LEARNING_RATE 0.01
#define STD_DEV 1

class Warehouse_ES : public Warehouse {
	public:
		Warehouse_ES(YAML::Node configs) : Warehouse(configs){}
		~Warehouse_ES(void);

		epoch_results SimulateEpoch(bool verbose,int epoch){;}
		epoch_resultsES SimulateEpochES (bool verbose = false, int epoch = 1);

		void InitialiseMATeam(); // create agents for each vertex in graph
		std::vector<float> QueryActorMATeam(std::vector<float> states);
		void initialiseNNWeights(std::vector<esNN> v);

		void setTeamNNs(std::vector<esNN*> teamNNs);

		std::vector<esNN*> produce_random_team_NNs();
	protected:
		float N_proc_std_dev;
		std::vector<ESAgent *> maTeam;		
};