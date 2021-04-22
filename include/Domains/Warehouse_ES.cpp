#include "Warehouse_ES.hpp"

Warehouse_ES::~Warehouse_ES(void){
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
	for (size_t i = 0; i != maTeam.size(); i++){
		delete maTeam[i];
		maTeam[i] = NULL;
	}
}

epoch_resultsES Warehouse_ES::SimulateEpochES(const int epoch, bool verbose){
	InitialiseNewEpoch();
	// const int64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();		
	//for each agent add noise sample * std_dev
	std::vector<std::vector<torch::Tensor>> samples(maTeam.size());
	for (size_t i = 0; i < maTeam.size(); i++){
		for (size_t j = 0; j < maTeam[i]->NN->parameters().size(); j++ ){
			torch::Tensor random_sample = torch::randn(maTeam[i]->NN->parameters()[j].sizes());			
			torch::Tensor t = (maTeam[i]->NN->parameters()[j]).detach().clone();
			maTeam[i]->NN->parameters()[j].set_data(t + N_proc_std_dev * random_sample);
			samples[i].push_back(random_sample);		
		}		
		// maTeam[i]->updateNNWeights(STD_DEV * random_sample);
	}

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"=== Epoch: "<<epoch<<" ==============================================================Step - "<<t<<std::endl;

		//Select action	
		const std::vector<float> actions = QueryActorMATeam(get_edge_utilization());

		std::vector<float> final_costs = baseCosts;
		for (size_t n = 0; n < N_EDGES; n++)
			final_costs[n] += actions[n] * max_base_travel_cost();

		if (verbose){
			printf("\nFinalCosts: ");
			for (size_t n = 0; n < final_costs.size(); n++)
				printf(" %.1f",final_costs[n]);
			printf("\n");
		}
	
		if (verbose)
			print_warehouse_state();
		traverse_one_step(final_costs);
	}

	epoch_resultsES results;
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
	results.update(totalSuccess, totalMove, totalEnter, totalWait, samples);
	assert(totalMove+totalEnter+totalWait == whAGVs.size()*nSteps);

	return results;
}

void Warehouse_ES::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	assert(maTeam.empty());

	if(agent_type == agent_def::centralized)
		maTeam.push_back(new ESAgent(N_EDGES*(1+incorporates_time), N_EDGES));
	else if(agent_type == agent_def::link){			
		for (size_t i = 0; i < whGraph->GetEdges().size(); i++){				
			maTeam.push_back(new ESAgent((1+incorporates_time), 1));
		}
	}
	else if (agent_type == agent_def::intersection){
		for (int v : whGraph->GetVertices())
			maTeam.push_back(new ESAgent((1+incorporates_time)*whAgents[v]->eIDs.size(), whAgents[v]->eIDs.size()));
	}

	assert(!maTeam.empty());
}

// Template this
std::vector<float> Warehouse_ES::QueryActorMATeam(std::vector<float> states){
 	assert(states.size() == N_EDGES*(1 + incorporates_time));
 	if(agent_type == agent_def::centralized)
 		return maTeam[0]->evaluateNN(states);
 	else if (agent_type == agent_def::link){
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

 		for (size_t i = 0; i < maTeam.size(); i++)
 			if(incorporates_time)
 				actions.push_back(maTeam[i]->evaluateNN({states[i], states[i+N_EDGES]})[0]);
 			else{
 				float t = (maTeam[i]->evaluateNN({states[i]}))[0];
 				
 				actions.push_back(t);
 			}
		assert(actions.size() == N_EDGES);

 		return actions;
 	}else if (agent_type == agent_def::intersection){
 		std::vector<float> actions(N_EDGES);

		for (size_t i = 0; i < maTeam.size(); i++)
			for (int j = 0; whAgents[i]->eIDs.size(); j++)
				actions[whAgents[i]->eIDs[j]] =
					maTeam[i]->evaluateNN(states)[j];

		return actions;
	}
	else{
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
		return {0};
	} 
}

void Warehouse_ES::set_team_NNs(std::vector<esNN*> teamNNs){
	for (size_t i = 0; i < maTeam.size(); i++)	
		maTeam[i]->setNN(teamNNs[i]);
}

/************************************************************************************************
**Note	:This is very Junky solution								*
**Output:A random std::vector<esNN*> which contains the NNs of a multiagent team		*
************************************************************************************************/
std::vector<esNN*> Warehouse_ES::produce_random_team_NNs(){
	std::vector<esNN*> team;

	if (agent_type == agent_def::centralized)
		team.push_back((new ESAgent(N_EDGES*(1+incorporates_time), N_EDGES))->NN);
	else if (agent_type == agent_def::link)
		for (size_t i = 0; i < whGraph->GetEdges().size(); i++)
			team.push_back((new ESAgent((1+incorporates_time), 1))->NN);
	else if (agent_type == agent_def::intersection)
		for (int v : whGraph->GetVertices())
			team.push_back((new ESAgent((1+incorporates_time)*whAgents[v]->eIDs.size(), whAgents[v]->eIDs.size()))->NN);

	return team;
}

// void Warehouse_ES::initialiseNNWeights(std::vector<esNN> v){

// 	for (int i = 0; i < v.size(); i++){
// 		maTeam[i].updateNNWeights()
// 	}
// }
