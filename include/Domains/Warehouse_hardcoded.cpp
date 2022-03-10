#include "Warehouse_hardcoded.hpp"

Warehouse_hardcoded::~Warehouse_hardcoded(void){
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
}

epoch_results Warehouse_hardcoded::simulate_epoch(bool verbose){
	InitialiseNewEpoch();

	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"=== time step: "<<t<<" ==============================================================Step - "<<t<<std::endl;

		const std::vector<float> current_state = get_edge_utilization();
		//Select action
		std::vector<float> actions(38, 0);

		{
			if (current_state[0] != 0)
				actions[0] = 1024;
			if (current_state[4] != 0)
				actions[4] = 1024;
			if (current_state[1] != 0)
				actions[1] = 512;
			if (current_state[5] != 0)
				actions[5] = 512;
			if (current_state[8] != 0)
				actions[8] = 512;
			if (current_state[9] != 0)
				actions[9] = 512;
		}

		std::vector<float> final_costs = baseCosts;
		for (size_t n = 0; n < N_EDGES; n++)
			final_costs[n] += actions[n] * max_base_travel_cost();

		if (verbose)
			print_warehouse_state();
		traverse_one_step(final_costs);
	}

	epoch_results results;
	// Log Perfomance Counters
	size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0;
	for (size_t k = 0; k < whAGVs.size(); k++){
		totalMove += whAGVs[k]->GetMoveTime();
		totalEnter += whAGVs[k]->GetEnterTime();
		totalWait += whAGVs[k]->GetWaitTime();
		totalSuccess += whAGVs[k]->GetNumCompleted();
	}
	if(verbose)
		std::cout<<"Stats:\n Total Move: "<<totalMove<<"\n Total Enter: "<<totalEnter<<
			"\n Total wait: "<<totalWait<< "\n Total Success: "<<totalSuccess<<std::endl;
	results.update((float) totalSuccess, (float) totalMove, (float) totalEnter, (float) totalWait);
	assert(totalMove+totalEnter+totalWait == whAGVs.size()*nSteps);

	return results;
}