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
}

epoch_results Warehouse_COMA::simulate_epoch_COMA (bool verbose){

	epoch_results results; // TODO fix
	//std::normal_distribution<float> n_process(1, N_proc_std_dev);
	//std::default_random_engine n_generator(time(NULL));
	
	///SIMULATE step
	std::vector<experience_replay> replay,samples; //empty buffer
	samples.reserve(DQN_consts::batch_size);
	replay.reserve(DQN_consts::simulation_steps);

	InitialiseNewEpoch();
	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0),next_state;


	for (int t = 0; t < DQN_consts::simulation_steps ; t++){
		// std::cout<<"State: "<<cur_state<<std::endl;
		std::vector<float> actions = query_actor_MATeam(cur_state,true);
		// std::cout<<"Actions: "<<actions<<std::endl;
		traverse_one_step(actions);
		 
		// Log Performance Counters
		size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0;//, totalCommand = 0;
		for (size_t k = 0; k < whAGVs.size(); k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			//totalCommand += whAGVs[k]->GetNumCommanded();
		}
		
		for (size_t i = 0; i < whAGVs.size(); i++)// Reset all AGVs
			whAGVs[i]->ResetPerformanceCounters();
		
		/* Get reward for each agent explicitly*/
		float reward = totalMove;// + totalEnter; 
		// std::cout<<"Reward: "<<reward<<std::endl;
		next_state = get_edge_utilization();		
		replay.push_back({cur_state, next_state, actions, reward});	
		cur_state = next_state;

		// TRAINING

		if (replay.size() < DQN_consts::batch_size )
			continue;
		
		/*Get samples(batch)*/
		samples.clear();
		std::ranges::sample(replay, std::back_inserter(samples), DQN_consts::batch_size, std::mt19937{std::random_device{}()});
		// std::cout<samples[1].reward<<std::endl;
		// std::cout<samples[49].reward<<std::endl;

		/* Reset Target Critic in EVERY C steps*/
		if(! t % DQN_consts::reset_step )
			COMAAgent::reset_target_critic(); 
	}
	return evaluateEpoch();

	// std::vector<float> q_input_states, q_input_actions, rewardsV;
	// q_input_states.reserve(COMAAgent::get_batch_size()*nSteps);
	// q_input_actions.reserve(COMAAgent::get_batch_size()*nSteps);
	// rewardsV.reserve(COMAAgent::get_batch_size()*nSteps);

	// for (size_t i = 0; i < maTeam.size(); ++i){
	// 	q_input_states.clear();
	// 	q_input_actions.clear();
	// 	rewardsV.clear();

	// 	for (size_t b = 0; b < COMAAgent::get_batch_size(); ++b){
	// 		for (size_t t = 0; t < nSteps; ++t){				
	// 			q_input_actions.push_back(replay[b*nSteps + t].action[i]);				
	// 			//q_input_states.push_back(replay[b*nSteps + t].current_state[i]);
	// 			q_input_states.insert(q_input_states.end(), replay[b*nSteps + t].current_state.begin(), replay[b*nSteps + t].current_state.end());
	// 			rewardsV.push_back(replay[b*nSteps + t].reward);
	// 		}
	// 	}

	// 	// //Train Critic 
	// 	// // torch::Tensor Q_targets = COMAAgent::evaluate_target_critic_NN(q_input_states,q_input_actions).squeeze(1);
	// 	// // torch::Tensor Q = COMAAgent::evaluate_critic_NN(q_input_states,q_input_actions).squeeze(1);

	// 	// torch::Tensor rewards = torch::tensor(rewardsV);//.unsqueeze(0);
	// 	// //std::cout<<rewards<<std::endl;
	// 	// //std::cout << Q_targets << std::endl;
	// 	// Q_targets = Q_targets+rewards;

	// 	// torch::Tensor dQ = Q_targets - Q;
	// 	// torch::Tensor critic_loss = torch::mean(torch::pow(dQ,2));
	// 	// std::cout<<critic_loss<<std::endl;

	// 	// COMAAgent::optimizerQNN->zero_grad();
	// 	// critic_loss.backward();
	// 	// COMAAgent::optimizerQNN->step();


	// }


	
	//TODO try sampling from history
	// std::vector<float> sample_index;
	// for (size_t i = 0; i < replay.size(); ++i)
	// 	sample_index.push_back(i);

	// std::vector<float> state;
	// std::vector<float> s,a;
	// std::vector<int> action_samples;
	// s.reserve(COMA_consts::actor_samples);
	// a.reserve(COMA_consts::actor_samples);
	// action_samples.reserve(COMA_consts::actor_samples);

	// for (size_t i = 0; i < maTeam.size(); ++i) {
		//Train Actor
		// const torch::Tensor monte_carlo_samples = torch::rand(COMA_consts::actor_samples);
		// const torch::Tensor monte_carlo_samples = torch::rand(COMA_consts::actor_samples);

		// q_input_actions.clear();			
		// q_input_states.clear();
		// a.clear();
		// s.clear();
		// for (size_t b = 0; b < COMAAgent::get_batch_size(); ++b){
		// 	for (size_t t = 0; t < nSteps; ++t){				
		// 		q_input_actions.push_back(replay[b*nSteps + t].action[i]);
		// 		q_input_states.push_back(replay[b*nSteps + t].current_state[i]);				
		// 	}
		// }

		// std::sample(sample_index.begin(), sample_index.end(), std::back_inserter(action_samples),COMA_consts::actor_samples
		// 			, std::mt19937{std::random_device{}()});

		// for (size_t j = 0; j < action_samples.size(); j++){
		// 	a.push_back(q_input_actions[action_samples[i]]);
		// 	s.push_back(q_input_states[action_samples[i]]);
		// }

		// torch::Tensor Q = COMAAgent::evaluate_critic_NN(s,a).squeeze(1);

		// std::cout << Q << std::endl;

		// torch::Tensor Q = COMAAgent::evaluate_critic_NN(s,a).squeeze(1);
		
		//torch::Tensor Baseline = ;

		// for (int b = 0; b < COMAAgent::get_batch_size(); ++b){
		// 	for (int t = 0; t < nSteps; ++t){
		// 		targets[b][t][i] = temp[b*nSteps + t]; //Targets y_t for each agent
		// 	}
	// 	}
	// }




	
}

epoch_results Warehouse_COMA::evaluateEpoch(){
	epoch_results results;
	InitialiseNewEpoch();
	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0),next_state;

	for (size_t t = 0; t < 200; t++){
		
		// std::cout<<"State: "<<cur_state<<std::endl;
		std::vector<float> actions = query_actor_MATeam(cur_state,false);
		// std::cout<<"Actions: "<<actions<<std::endl;
		traverse_one_step(actions);
		 
		// Log Performance Counters
		size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0;//, totalCommand = 0;
		for (size_t k = 0; k < whAGVs.size(); k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			//totalCommand += whAGVs[k]->GetNumCommanded();
		}

		for (size_t i = 0; i < whAGVs.size(); i++)// Reset all AGVs
			whAGVs[i]->ResetPerformanceCounters();
				
		results.update((float) totalSuccess, (float) totalMove, (float) totalEnter, (float) totalWait);
		
		next_state = get_edge_utilization();
		cur_state = next_state;				
	}
	return results;
}

void Warehouse_COMA::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	if (algo != algo_type::coma){
		std::cout << "ERROR: Invalid agent_defintion" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	assert(maTeam.empty());
	if (agent_type == agent_def::centralized){
		std::cout << "Centralized Agent is not supported with DQN yet" << std::endl;
		exit(EXIT_FAILURE);
	}
	else if (agent_type == agent_def::link)
		for (size_t i = 0; i < whGraph->GetEdges().size(); i++)
			maTeam.push_back(new DQNAgent((1+incorporates_time),DQN_consts::actions_size));
	else if (agent_type == agent_def::intersection){
		std::cout << "Intersection Agent Does not work with DQN yet" << std::endl;
		exit(EXIT_FAILURE);
	}

	assert(!maTeam.empty());
}

std::vector<float> Warehouse_COMA::query_actor_MATeam(std::vector<float> &states,bool training){
	srand((unsigned)time(NULL));

	assert(states.size() == N_EDGES*(1 + incorporates_time));
	assert(agent_type == agent_def::link);
	std::vector<float> actions;
	actions.reserve(N_EDGES);

	for (size_t i = 0; i < maTeam.size(); i++)
		if(incorporates_time){
			// actions.push_back(maTeam[i]->evaluate_critic_NN({states[i], states[i+N_EDGES]})[0]);
			continue;
		}else{
			std::vector<float> v;
			if(training){
				float r = (float) rand()/RAND_MAX;
				// std::cout<<r<<std::endl;
				if(r > 0.5){
					actions.push_back(DQN_consts::actions[rand() % DQN_consts::actions_size]);
				}else{
					v = (maTeam[i]->evaluate_critic_NN({states[i]},{states[i]}));			
					actions.push_back(DQN_consts::actions[std::max_element(v.begin(),v.end()) - v.begin()]);	
				}
			}else{
				v = (maTeam[i]->evaluate_critic_NN({states[i]},{states[i]}));			
				actions.push_back(DQN_consts::actions[std::max_element(v.begin(),v.end()) - v.begin()]);
			}
			
		}
	return actions;
}