#include "Warehouse_COMA.hpp"

Warehouse_COMA::~Warehouse_COMA(void){
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
	// DDPGAgent::clear_replar_buffer();
}


void Warehouse_COMA::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	if (algo != algo_type::coma){
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
	}

	// COMAAgent::init_critic_NNs();
	assert(maTeam.empty());
	if(agent_type == agent_def::centralized)
		maTeam.push_back(new COMAAgent(N_EDGES*(1+incorporates_time), N_EDGES,N_EDGES*(1+incorporates_time), N_EDGES));
	else if(agent_type == agent_def::link){			
		for (size_t i = 0; i < whGraph->GetEdges().size(); i++){				
			maTeam.push_back(new COMAAgent((1+incorporates_time), 1,(1+incorporates_time)*N_EDGES, N_EDGES));
		}
	}
		else if (agent_type == agent_def::intersection)
		for (int v : whGraph->GetVertices())
			maTeam.push_back(new COMAAgent((1+incorporates_time)*whAgents[v]->eIDs.size(), whAgents[v]->eIDs.size(), (1+incorporates_time)*N_EDGES, N_EDGES));

	assert(!maTeam.empty());
}