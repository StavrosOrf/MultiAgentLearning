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
	COMAAgent::ERB.clear();
}


epoch_results Warehouse_COMA::simulate_epoch_COMA (bool verbose, int epoch){
	InitialiseNewEpoch();
	std::normal_distribution<float> n_process(1, N_proc_std_dev);
	std::default_random_engine n_generator(time(NULL));

	COMAAgent::ERB.clear();

	assert(COMAAgent::get_batch_size());
	for (size_t e = 0; e != COMAAgent::get_batch_size(); e++){
		std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0);
		for (size_t t = 0; t < nSteps; t++){
			std::vector<float> actions = query_actor_MATeam(cur_state);
			
			traverse_one_step(actions);
			 
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
				std::cout<<"Stats:\n Total Move: "<<totalMove<<"\n Total Enter: "<<totalEnter<<
					"\n Total wait: "<<totalWait<< "\n Total Success: "<<totalSuccess<<std::endl;

			float reward = totalMove + totalEnter;

			const std::vector<float> old_state = cur_state;
			const std::vector<float> next_state = get_edge_utilization();
			
			COMAAgent::ERB.add_random({old_state, next_state, actions, reward});
		}

	}



	
	/*
	epoch_results results;
	float reward;

	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0);

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"=== Epoch: "<<epoch<<" ==============================================================Step - "<<t<<std::endl;

		//Select action
		std::vector<float> actions; //= QueryActorMATeam(cur_state); //TODO

		if(verbose){
			maTeam[0]->printAboutNN();

			std::cout<<"-\nState: " << cur_state <<"\n-"<< std::endl;			

			printf("Actor: ");
			for (size_t n = 0; n < actions.size(); n++)
				printf(" %.2f", actions[n]);
			printf("\n");

			// printf("TActor: ");
			// for (size_t n = 0; n < actions.size(); n++)
			// 	printf(" %4.6f", QueryTargetActorMATeam(cur_state)[n]);
			// printf("\n");
		}

		std::vector<float> final_costs = baseCosts;
		for (size_t n = 0; n < N_EDGES; n++){ // Add Random Noise from process N
			// actions[n] = QueryActorMATeam(cur_state)[n]*max_base_travel_cost() + n_process(n_generator)*max_base_travel_cost();			
			final_costs[n] += actions[n];
		}

		if (verbose){
			// printf("CostsAdd: ");
			// for (size_t n = 0; n < final_costs.size(); n++)
			// 	printf(" %.1f",final_costs[n]-baseCosts[n] + min);
			printf("\nFinalCosts: ");
			for (size_t n = 0; n < final_costs.size(); n++)
				printf(" %.1f",final_costs[n]);
			printf("\n");
		}

		float routable_agvs = 0;//total number of agvs that could be routed in the graph, Used for reward 2
		for (size_t v = 0; v != whGraph->GetNumVertices(); v++) // 
			routable_agvs += std::min<float>(get_vertex_remaining_outgoing_capacity(v), get_vertex_utilization()[v]);

		if (verbose)
		 	print_warehouse_state();

		traverse_one_step(final_costs);

		// printAgvPaths();

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
			std::cout<<"Stats:\n Total Move: "<<totalMove<<"\n Total Enter: "<<totalEnter<<
					"\n Total wait: "<<totalWait<< "\n Total Success: "<<totalSuccess<<std::endl;
		results.update(totalSuccess, totalMove, totalEnter, totalWait);
		assert(totalMove+totalEnter+totalWait == whAGVs.size());

		for (AGV* a : whAGVs) //Reset AGVs counters
			a->ResetPerformanceCounters();



// REWARD_METHOD == 2
		float total_AGVs_that_entered_an_edge_this_step = 0;
		for(AGV* a : whAGVs)
			total_AGVs_that_entered_an_edge_this_step += (float) (a->entered_edge_this_step());
		reward = total_AGVs_that_entered_an_edge_this_step - routable_agvs;



		// reward = rand()%100 - 50;


		if(verbose)
			std::cout<<"Reward: "<<reward<<std::endl;
		assert(!std::isnan(reward) && !std::isinf(reward));
		// if (routable_agvs != 0)
		COMAAgent::ERB.add_random({temp_state,cur_state,actions,reward});
		*/
		// else if(verbose)
	
		// if(DDPGAgent::get_replay_buffer_size() > DDPGAgent::get_batch_size() * 2 && t%TRAINING_STEP == 0){
		// 	for (size_t n = 0; n < ddpg_maTeam.size(); n++){
		// 		std::vector<experience_replay> miniBatch = DDPGAgent::getReplayBufferBatch();
		// 		std::vector<float> Qvals;//Qvals
		// 		std::vector<float> Qprime;//Qprime ( the Y )
		// 		std::vector<std::vector<float>> states; //all states from the batch
		// 		std::vector<std::vector<float>> all_actions; //all actions from the batch
		// 		Qvals.reserve(DDPGAgent::get_batch_size());
		// 		Qprime.reserve(DDPGAgent::get_batch_size());
		// 		states.reserve(DDPGAgent::get_batch_size());
		// 		all_actions.reserve(DDPGAgent::get_batch_size());

		// 		for (size_t i = 0; i < DDPGAgent::get_batch_size(); i++){
		// 			experience_replay b = miniBatch[i];
		// 			std::vector<float> n_target_actor = QueryTargetActorMATeam(b.next_state);
		// 			// std::cout <<"nstate: "<<b.next_state.size()<<std::endl;
		// 			// std::cout <<"n_target_actor: "<<n_target_actor.size()<<std::endl;
		// 			assert(N_EDGES && n_target_actor.size() == N_EDGES);
		// 			float y = b.reward + GAMMA *
		// 				ddpg_maTeam[n]->EvaluateTargetCriticNN_DDPG(b.next_state,n_target_actor)[0];
		// 			//Generate Qvals and Qprime for Q backprop
		// 			float q = ddpg_maTeam[n]->EvaluateCriticNN_DDPG(b.current_state,b.action)[0];
		// 			Qvals.push_back(q);
		// 			Qprime.push_back(y);

		// 			if (agent_type == agent_def::centralized)
		// 				states.push_back(b.current_state);						
		// 			else if (agent_type == agent_def::link){
		// 					states.push_back(b.current_state);
		// 					all_actions.push_back(b.action);																		
		// 			}					
		// 		}
				
		// 		//Update all the NNs
		// 		ddpg_maTeam[n]->updateQCritic(Qvals, Qprime,verbose);

		// 		if (agent_type == agent_def::centralized)
		// 			ddpg_maTeam[n]->updateMuActor(states,verbose);	
		// 		else if (agent_type == agent_def::link)
		// 			ddpg_maTeam[n]->updateMuActorLink(states,all_actions,n,incorporates_time);					
		// 		else if (agent_type == agent_def::intersection){/*TODO*/}
		// 	}

		// 	for (size_t n = 0; n < ddpg_maTeam.size(); n++)
		// 		ddpg_maTeam[n]->updateTargetWeights();

		// }else
		// 	if(verbose)
		// 		std::cout << "Not enough Replays yet for updating NNs!"<<std::endl;
	//}	
	//return results;
}



void Warehouse_COMA::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	if (algo != algo_type::coma){
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
	}

	COMAAgent::init_critic_NNs(N_EDGES*(1+incorporates_time), N_EDGES);
	assert(maTeam.empty());
	if (agent_type == agent_def::centralized)
		maTeam.push_back(new COMAAgent(N_EDGES*(1+incorporates_time), N_EDGES));
	else if (agent_type == agent_def::link)
		for (size_t i = 0; i < whGraph->GetEdges().size(); i++)
			maTeam.push_back(new COMAAgent((1+incorporates_time), 1));
	else if (agent_type == agent_def::intersection)
		for (int v : whGraph->GetVertices())
			maTeam.push_back(new COMAAgent((1+incorporates_time)*whAgents[v]->eIDs.size(), whAgents[v]->eIDs.size()));

	assert(!maTeam.empty());
}

std::vector<float> Warehouse_COMA::query_actor_MATeam(std::vector<float> states){
 	assert(states.size() == N_EDGES*(1 + incorporates_time));
 	if(agent_type == agent_def::centralized){
		assert(0);
 		return maTeam[0]->evaluate_actorNN(states);
	}
 	else if (agent_type == agent_def::link){
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

 		for (size_t i = 0; i < maTeam.size(); i++)
 			if(incorporates_time)
 				actions.push_back(maTeam[i]->evaluate_actorNN({states[i], states[i+N_EDGES]})[0]);
 			else{
 				float t = (maTeam[i]->evaluate_actorNN({states[i]}))[0];
 				actions.push_back(t);
 			}
 		return actions;
 	}
	 else if (agent_type == agent_def::intersection){//TODO WITH TIME
 		std::vector<float> actions;
 		actions.reserve(N_EDGES);

		for (size_t i = 0; i < maTeam.size(); i++)
			for (int j = 0; whAgents[i]->eIDs.size(); j++)
				actions[whAgents[i]->eIDs[j]] =
					maTeam[i]->evaluate_actorNN(states)[j];
		return actions;
	}
	else{
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
		return {0};
	} 
}