#include "Warehouse_DQN.hpp"
#define REWARD_METHOD_3_BUFFER_SIZE 30

Warehouse_DQN::~Warehouse_DQN(void){
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

epoch_results Warehouse_DQN::simulate_epoch_DQN([[maybe_unused]] bool verbose){

	maTeam[0]->printAboutNN();
	epoch_results results;

	std::vector<experience_replayDQN> replay,samples; //empty buffer
	samples.reserve(DQN_consts::batch_size);
	replay.reserve(DQN_consts::simulation_steps);

	InitialiseNewEpoch();
	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0),next_state;
	
	int index = 0;
	std::vector<float> bufferCurrStates[REWARD_METHOD_3_BUFFER_SIZE];
	std::vector<float> bufferNextStates[REWARD_METHOD_3_BUFFER_SIZE];
	std::vector<float> bufferActions[REWARD_METHOD_3_BUFFER_SIZE];
	float bufferRewards[REWARD_METHOD_3_BUFFER_SIZE];
	for (int i = 0; i < REWARD_METHOD_3_BUFFER_SIZE; i++)
		bufferRewards[i] = -1;

	for (size_t t = 0; t != DQN_consts::simulation_steps; t++){
		// std::cout<<"State: "<<cur_state<<std::endl;
		std::vector<float> actions = query_actor_MATeam(cur_state,true);
		
		traverse_one_step(actions);
		next_state = get_edge_utilization();
		std::cout<<t <<". |State: "<<next_state<<std::endl;
		// std::cout<<"Actions: "<<actions<<std::endl;
		// Log Performance Counters
		size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0,totalCommand = 0;
		// reward.clear();
		std::vector<float> reward(maTeam.size());
		for (size_t k = 0; k < whAGVs.size(); k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			totalCommand += whAGVs[k]->GetNumCommanded();
			// std::cout<<"Move: "<<whAGVs[k]->GetMoveTime()<<std::endl;
			// std::cout<<"Command:  "<<whAGVs[k]->GetNumCommanded()<<std::endl;
			
			// if(whAGVs[k]->GetMoveTime() > 0 && whAGVs[k]->GetNumCompleted() < 1){
			// 	std::cout<<"Enter:  "<<whAGVs[k]->GetEnterTime()<<std::endl;
			// 	std::cout<<"Completed:  "<<whAGVs[k]->GetNumCompleted()<<std::endl;	
			// 	std::cout<<"Edge: "<<whAGVs[k]->GetCurEdge()->GetVertex1()<<std::endl;	
			// }
		}
		// std::cout<<"Total Move: "<<totalMove<<std::endl;
		// std::cout<<"Total Succesfull: "<<totalSuccess<<std::endl;
		// if(t == 50)
		// 	exit(0);

		for (size_t i = 0; i < whAGVs.size(); i++)// Reset all AGVs
			whAGVs[i]->ResetPerformanceCounters();
		
		/* Get reward for each agent explicitly*/
		//Reward thought: |#AGVs that could move - #AGVs that moved|
		// float reward = totalMove;// + totalEnter;

		//Update current values for each state
		for (int i = 0; i< REWARD_METHOD_3_BUFFER_SIZE; i++){
			if (bufferRewards[i] != -1)
				bufferRewards[i] += totalSuccess;
		}

		//Time to write replay
		if (bufferRewards[index] != -1 ){
			// experience_replay r = {bufferCurrStates[index],cur_state,bufferActions[index],bufferRewards[index]/100};
			for (size_t i = 0; i < maTeam.size(); ++i)
			{
				reward[i] = totalSuccess; // totalMove ;//+ totalEnter;
				reward[i] = bufferRewards[index]; // totalMove ;//+ totalEnter;
			}						
			replay.push_back({bufferCurrStates[index], bufferNextStates[index], bufferActions[index],reward});		
			std::cout<<"Reward: "<<bufferRewards[index]<<std::endl;
		}

		bufferActions[index] = actions;
		bufferCurrStates[index] = cur_state;
		bufferNextStates[index] = next_state;
		bufferRewards[index] = 0;
		//Move buffer pointer
		index ++;

		if(index >= REWARD_METHOD_3_BUFFER_SIZE)
			index = 0;

		for (size_t i = 0; i < maTeam.size(); ++i)
		{
			reward[i] = totalSuccess; // totalMove ;//+ totalEnter;
		}
		// std::cout<<"---Reward: "<<reward[0]<<std::endl;
				
		
		cur_state = next_state;

		// TRAINING

		if (replay.size() < DQN_consts::batch_size )
			continue;
		
		/*Get samples(batch)*/
		
		for (size_t i = 0; i < maTeam.size(); ++i){
			samples.clear();
			std::ranges::sample(replay, std::back_inserter(samples), DQN_consts::batch_size, std::mt19937{std::random_device{}()});
			// std::cout<"TRAIN"<<std::endl;
			maTeam[i]->trainCritic({samples},i);
		}


		/* Reset Target Critic in EVERY C steps*/
		if(! t % DQN_consts::reset_step )
			for (auto agent : maTeam)
				agent->reset_target_critic();
	}
	return evaluateEpoch();

}

epoch_results Warehouse_DQN::evaluateEpoch(){
	epoch_results results;
	InitialiseNewEpoch();
	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0),next_state;
	std::cout<<"=================================== Evaluation ========================="<<std::endl;
	for (size_t t = 0; t < 200; t++){
		
		std::vector<float> actions = query_actor_MATeam(cur_state,false);
		std::cout<<t <<". |Actions: "<<actions<<std::endl;
		std::cout<<"    State: "<<next_state<<"\n"<<std::endl;
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

void Warehouse_DQN::InitialiseMATeam(){
	assert(whAgents.size());//this must be called after whAgents have been initialized
	if (algo != algo_type::dqn){
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
			// maTeam.push_back(new DQNAgent((1+incorporates_time),DQN_consts::actions_size));
			maTeam.push_back(new DQNAgent(1 + whGraph->GetEdges().size()*(1+incorporates_time),DQN_consts::actions_size));
	else if (agent_type == agent_def::intersection){//IMPLEMENT
		std::cout << "Intersection Agent Does not work with DQN yet" << std::endl;
		exit(EXIT_FAILURE);
	}

	assert(!maTeam.empty());
}

std::vector<float> Warehouse_DQN::query_actor_MATeam(std::vector<float> &states,bool training){
	srand((unsigned)time(NULL));

	assert(states.size() == N_EDGES*(1 + incorporates_time));
	assert(agent_type == agent_def::link); //TODO REMOVE
	std::vector<float> actions;
	std::vector<float> state_temp;
	actions.reserve(N_EDGES);

	for (size_t i = 0; i < maTeam.size(); i++)
		if(incorporates_time){
			// actions.push_back(maTeam[i]->evaluate_critic_NN({states[i], states[i+N_EDGES]})[0]);
			continue;
		}else{
			std::vector<float> v;
			if(training){
				float r = (float) rand()/RAND_MAX;
				
				if(r > 0.5){
					actions.push_back(DQN_consts::actions[rand() % DQN_consts::actions_size]);
				}else{
					// std::cout<<states<<std::endl;
					// assert(states.size());
					state_temp = states;
					state_temp.push_back(i+1);
					v = (maTeam[i]->evaluate_critic_NN({state_temp},{states[i]}));			
					actions.push_back(DQN_consts::actions[std::max_element(v.begin(),v.end()) - v.begin()]);	
				}
			}else{
				state_temp = states;
				state_temp.push_back(i+1);
				v = (maTeam[i]->evaluate_critic_NN({state_temp},{states[i]}));			
				actions.push_back(DQN_consts::actions[std::max_element(v.begin(),v.end()) - v.begin()]);
			}
			
		}
	return actions;
}