#include "Warehouse_DDPG.h"

#define REWARD_METHOD 1
//0 Random
//1 Value of state
//2 Routability
Warehouse_DDPG::~Warehouse_DDPG(void){
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

epoch_results Warehouse_DDPG::SimulateEpoch(bool verbose, int epoch){
	assert(0);
	InitialiseNewEpoch();
	std::normal_distribution<float> n_process(0, N_proc_std_dev);
	std::default_random_engine n_generator(time(NULL));
	epoch_results results = {0,0,0,0};
	const float maxBaseCost=*std::max_element(baseCosts.begin(), baseCosts.end());

	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0);

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"==============================================================Step - "<<t<<std::endl;
		//Select action
#if REWARD_METHOD == 1
		float value_of_state = 0, value_of_prev_state = 0;
#endif
		float reward = 0;
		std::vector<float> actions = QueryActorMATeam(cur_state);

		if(verbose){
			//std::cout<<"State: " << cur_state << std::endl;
			ddpg_maTeam[0]->printAboutNN();

			printf("Actor: ");
			for (size_t n = 0; n < actions.size(); n++)
				printf(" %4.6f", actions[n]);
			printf("\n");

			printf("TActor: ");
			for (size_t n = 0; n < actions.size(); n++)
				printf(" %4.6f", QueryTargetActorMATeam(cur_state)[n]);
			printf("\n");
		}

		std::vector<float> final_costs = baseCosts;
		for (size_t n = 0; n < N_EDGES; n++){ // Add Random Noise from process N
			//actions[n] = QueryActorMATeam(cur_state)[n]*maxBaseCost + n_process(n_generator)*baseCosts[n];
			actions[n] = QueryActorMATeam(cur_state)[n]*maxBaseCost*n_process(n_generator);
			final_costs[n] += actions[n];
		}

		if (verbose){
			printf("CostsAdd: ");
			for (size_t n = 0; n < final_costs.size(); n++)
				printf(" %4.6f",final_costs[n]-baseCosts[n]);
			printf("\n");
		}

		float routable_agvs = 0;//total number of agvs that could be routed in the graph
		for (int v = 0; v != whGraph->GetNumVertices(); v++)
			routable_agvs += std::min<float>(get_vertex_reamaining_outgoing_capacity(v), get_vertex_utilization()[v]);

		if (verbose)
			print_warehouse_state();
		traverse_one_step(final_costs);

		//update current state
		const std::vector<float> temp_state = cur_state;
		cur_state = get_edge_utilization();

		// Log Perfomance Counters
		size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0, totalCommand = 0;
		for (size_t k = 0; k < whAGVs.size(); k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			totalCommand += whAGVs[k]->GetNumCommanded();
		}
		if(verbose)
			std::cout<<"Stats: Total Move: "<<totalMove<<"\tTotal Enter: "<<totalEnter<<
					"\tTotal wait: "<<totalWait<< "\tTotal Success: "<<totalSuccess<<
					"\tTotal Command: "<<totalCommand<<std::endl;
		assert(totalMove+totalEnter+totalWait == whAGVs.size());
		
		results.update(totalSuccess, totalMove, totalEnter, totalWait);

		for (AGV* a : whAGVs) //Reset AGVs counters
			a->ResetPerformanceCounters();

#if REWARD_METHOD == 1
		//Create and Save replay to buffer
		{
			whGraph->reset_edge_costs();
			//TODO optimize Perfomance (with LUTs?)

			float total = 0;

			for (AGV* a : whAGVs)
				if (a->is_on_edge()){
					Search s0(whGraph, a->get_start_vertex(),
							a->GetNextVertex());
					Search s1(whGraph, a->GetNextVertex(),
							a->GetDestinationVertex());
					float totalInverse = 0;
					totalInverse += s0.PathSearchLenght();
					totalInverse += s1.PathSearchLenght();
					total += 1/totalInverse;
				} else if (a->is_on_graph()){
					Search s0(whGraph, a->get_start_vertex(),
							a->get_cur_vertex());
					Search s1(whGraph, a->get_cur_vertex(),
							a->GetDestinationVertex());
					float totalInverse = 0;
					totalInverse += s0.PathSearchLenght();
					totalInverse += s1.PathSearchLenght();
					total += 1/totalInverse;
				}
			value_of_state = total;
			reward = (value_of_state - value_of_prev_state)*32;

			value_of_prev_state = value_of_state;
			assert(!std::isnan(reward) && !std::isinf(reward));
		}
#endif

#if REWARD_METHOD == 2
		float total_AGVs_that_entered_an_edge_this_step = 0;
		for(AGV* a : whAGVs)
			total_AGVs_that_entered_an_edge_this_step += a->entered_edge_this_step();
		reward = total_AGVs_that_entered_an_edge_this_step - routable_agvs;
#endif
#if REWARD_METHOD == 0
		reward = rand()%100 - 50;
#endif
		if(verbose)
			std::cout<<"Reward: "<<reward<<std::endl;

		replay r = {temp_state,cur_state,actions,reward};
		if (routable_agvs != 0)
			DDPGAgent::addToReplayBuffer(r);
		else
			if(verbose)
				std::cout << "Skipping adding it to the replay buffer" << std::endl;

		if(DDPGAgent::get_replay_buffer_size() > DDPGAgent::get_batch_size() * 2 && t%TRAINING_STEP == 0){
			for (size_t n = 0; n < ddpg_maTeam.size(); n++){
				std::vector<replay> miniBatch = DDPGAgent::getReplayBufferBatch();
				std::vector<float> Qvals;//Qvals
				std::vector<float> Qprime;//Qprime ( the Y )
				std::vector<std::vector<float>> states; //all states from the batch
				std::vector<std::vector<float>> all_actions; //all actions from the batch
				Qvals.reserve(DDPGAgent::get_batch_size());
				Qprime.reserve(DDPGAgent::get_batch_size());
				states.reserve(DDPGAgent::get_batch_size());
				all_actions.reserve(DDPGAgent::get_batch_size());

				for (size_t i = 0; i < DDPGAgent::get_batch_size(); i++){
					replay b = miniBatch[i];
					std::vector<float> nta = QueryTargetActorMATeam(b.next_state);

					assert(N_EDGES && nta.size() == N_EDGES);
					float y = b.reward + GAMMA *
						ddpg_maTeam[n]->EvaluateTargetCriticNN_DDPG(b.next_state,nta)[0];
					//Generate Qvals and Qprime for Q backprop
					float q = ddpg_maTeam[n]->EvaluateCriticNN_DDPG(b.current_state,b.action)[0];
					Qvals.push_back(q);
					Qprime.push_back(y);

					if (agent_type == agent_def::centralized)
						states.push_back(b.current_state);						
					else if (agent_type == agent_def::link){
							states.push_back(b.current_state);
							all_actions.push_back(b.action);																		
					}					
				}
				
				//Update all the NNs
				ddpg_maTeam[n]->updateQCritic(Qvals, Qprime);

				if (agent_type == agent_def::centralized)
					ddpg_maTeam[n]->updateMuActor(states);	
				else if (agent_type == agent_def::link)
					ddpg_maTeam[n]->updateMuActorLink(states,all_actions,n,incorporates_time);					
				else if (agent_type == agent_def::intersection){/*TODO*/}
			}

			for (size_t n = 0; n < ddpg_maTeam.size(); n++)
				ddpg_maTeam[n]->updateTargetWeights();

		}else
			if(verbose)
				std::cout << "Not enough Replays yet for updating NNs!"<<std::endl;
	}	
	return results;
}

void Warehouse_DDPG::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	if (algo == algo_type::ddpg)
		assert(ddpg_maTeam.empty());
		if(agent_type == agent_def::centralized)
			ddpg_maTeam.push_back(new DDPGAgent(N_EDGES*(1+incorporates_time), N_EDGES,N_EDGES*(1+incorporates_time), N_EDGES));
		else if(agent_type == agent_def::link)
			for (size_t i = 0; whGraph->GetEdges().size(); i++)
				ddpg_maTeam.push_back(new DDPGAgent((1+incorporates_time), 1,(1+incorporates_time)*N_EDGES, N_EDGES));
 		else if (agent_type == agent_def::intersection)
			for (int v : whGraph->GetVertices())
				ddpg_maTeam.push_back(new DDPGAgent((1+incorporates_time)*whAgents[v]->eIDs.size(), whAgents[v]->eIDs.size(), (1+incorporates_time)*N_EDGES, N_EDGES));
	else{
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
	}
	assert(!ddpg_maTeam.empty());
}

std::vector<float> Warehouse_DDPG::QueryActorMATeam(std::vector<float> states){
 	assert(states.size() == N_EDGES*(1 + incorporates_time));
 	if(agent_type == agent_def::centralized)
 		return ddpg_maTeam[0]->EvaluateActorNN_DDPG(states);
 	else if (agent_type == agent_def::link){
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

 		for (size_t i = 0; i < ddpg_maTeam.size(); i++)
 			if(incorporates_time)
 				actions.push_back(ddpg_maTeam[i]->EvaluateActorNN_DDPG({states[i], states[i+N_EDGES]})[0]);
 			else{
 				float t = (ddpg_maTeam[i]->EvaluateActorNN_DDPG({states[i]}))[0];
 				actions.push_back(t);
 			}
		assert(actions.size() == N_EDGES);
 		return actions;
 	}else if (agent_type == agent_def::intersection){
 		std::vector<float> actions(N_EDGES);

		for (size_t i = 0; i < ddpg_maTeam.size(); i++)
			for (int j = 0; whAgents[i]->eIDs.size(); j++)
				actions[whAgents[i]->eIDs[j]] =
					ddpg_maTeam[i]->EvaluateActorNN_DDPG(states)[j];

		return actions;
	}
	else{
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
		return {0};
	} 
}


std::vector<float> Warehouse_DDPG::QueryTargetActorMATeam(std::vector<float> states){
 	assert(states.size() == N_EDGES*(1 + incorporates_time));
 	if(agent_type == agent_def::centralized)
 		return ddpg_maTeam[0]->EvaluateActorNN_DDPG(states);
 	else if (agent_type == agent_def::link){
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

 		for (size_t i = 0; i < ddpg_maTeam.size(); i++)
 			if(incorporates_time)
 				actions.push_back(ddpg_maTeam[i]->EvaluateTargetActorNN_DDPG({states[i], states[i+N_EDGES]})[0]);
 			else{
 				float t = (ddpg_maTeam[i]->EvaluateTargetActorNN_DDPG({states[i]}))[0];
 				actions.push_back(t);
 			}
 		return actions;
 	}else if (agent_type == agent_def::intersection){
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

		for (size_t i = 0; i < ddpg_maTeam.size(); i++)
			for (int j = 0; whAgents[i]->eIDs.size(); j++)
				actions[whAgents[i]->eIDs[j]] =
					ddpg_maTeam[i]->EvaluateTargetActorNN_DDPG(states)[j];
		return actions;
	}
	else{
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
		return {0};
	} 
}
