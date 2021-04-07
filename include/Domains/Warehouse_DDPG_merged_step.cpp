#include "Warehouse_DDPG_merged_step.h"

Warehouse_DDPG_merged_step::~Warehouse_DDPG_merged_step(void){
	delete whGraph;
	whGraph = 0;
	for (size_t i = 0; i < whAGVs.size(); i++){
		delete whAGVs[i];
		whAGVs[i] = 0;
	}
	for (size_t i = 0; i < whAgents.size(); i++){
		delete whAgents[i];
		whAgents[i] = 0;
	}
	for (size_t i = 0; i != ddpg_maTeam.size(); i++){
		delete ddpg_maTeam[i];
		ddpg_maTeam[i] = NULL;
	}
}

epoch_results Warehouse_DDPG_merged_step::SimulateEpoch(bool verbose){
	InitialiseNewEpoch();
	std::normal_distribution<float> n_process(0, N_proc_std_dev);
	std::default_random_engine n_generator(time(NULL));
	const float maxBaseCost=*std::max_element(baseCosts.begin(), baseCosts.end());

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"==============================================================Step - "<<t<<std::endl;
		//Select action
		const std::vector<float> cur_state = get_edge_utilization();
		const std::vector<float> actions = QueryActorMATeam(cur_state);
		assert(actions.size() == N_EDGES);

		if(verbose){
			print_warehouse_state();
			ddpg_maTeam[0]->printAboutNN();
			printf("Actions:		 ");
			for (size_t n = 0; n < actions.size(); n++)
				printf(" %4.2f",actions[n]);
			printf("\n");
		}

		std::vector<float> final_costs = baseCosts;
		// Add Random Noise from process N
		for (size_t n = 0; n < N_EDGES; n++)
			final_costs[n] += (actions[n]+n_process(n_generator)) * maxBaseCost;
		assert(final_costs.size() == N_EDGES);

		if(verbose){
			printf("CostsAdd: ");
			for (size_t n = 0; n < final_costs.size(); n++)
				printf(" %4.1f",final_costs[n]-baseCosts[n]);
			printf("\n");
		}

		traverse_one_step(final_costs);
	}

	// Log Perfomance Counters
	epoch_results results = {0,0,0,0};
	for (size_t k = 0; k < whAGVs.size(); k++){
		results.totalMove += whAGVs[k]->GetMoveTime();
		results.totalEnter += whAGVs[k]->GetEnterTime();
		results.totalWait += whAGVs[k]->GetWaitTime();
		results.totalDeliveries+= whAGVs[k]->GetNumCompleted();
		//totalCommand += whAGVs[k]->GetNumCommanded();
	}

	//TODO LEARN
	return results;
}
