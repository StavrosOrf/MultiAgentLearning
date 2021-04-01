#include "WarehouseCentralised.h"

WarehouseCentralised::~WarehouseCentralised(void){
	delete whGraph;
	whGraph = 0;
	for (size_t i = 0; i < whAGVs.size(); i++){
		delete whAGVs[i];
		whAGVs[i] = 0;
	}
	for (size_t i = 0; i < whAgents.size(); i++){
		delete whAgents[i];
		whAgents[i] = 0;
		//TODO delete AGENTS
	}
	if (outputEvals)
		evalFile.close();
	if (outputEpReplay){
		agvStateFile.close();
		agvEdgeFile.close();
		agentStateFile.close();
		agentActionFile.close();
	}
}

epoch_results WarehouseCentralised::SimulateEpochDDPG(bool verbose){
	InitialiseNewEpoch();
	float totalDeliveries = 0;
	std::normal_distribution<float> n_process(0.0, 0.15);
	std::default_random_engine n_generator(time(NULL));
	epoch_results results = {0,0,0,0};
	const float maxBaseCost=*std::max_element(baseCosts.begin(), baseCosts.end());

	std::vector<float> cur_state(N_EDGES*(incorporates_time+1),0);
	std::vector<float> temp_state;

	if(verbose)
		print_warehouse_state();

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		if(verbose)
			std::cout <<"==============================================================Step - "<<t<<std::endl;
		//Select action
		float reward = 0;
		const vector<float> actions = ddpg_maTeam[0]->EvaluateActorNN_DDPG(cur_state);
		vector<float> final_costs = baseCosts;
		// Add Random Noise from process N
		for (size_t n = 0; n < N_EDGES; n++){
			final_costs[n] += std::clamp<float>(actions[n]+n_process(n_generator), -1, 1) * maxBaseCost;
			assert(final_costs[n] <= maxBaseCost + baseCosts[n]);
		}
		assert(final_costs.size() == N_EDGES);

		traverse_one_step(final_costs);

		//update current state
		temp_state = cur_state;
		cur_state = get_edge_utilization();
		if(verbose)
			print_warehouse_state();

		// Log Perfomance Counters
		size_t totalMove = 0, totalEnter = 0, totalWait = 0, totalSuccess = 0, totalCommand = 0;
		for (size_t k = 0; k < nAGVs; k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			totalCommand += whAGVs[k]->GetNumCommanded();
		}
		if(verbose)
			std::cout<<"Stats: \nTotal Move:\t"<<totalMove<<" \nTotal Enter:\t"<<totalEnter<<
					"\nTotal wait:\t"<<totalWait<< "\nTotal Success:\t"<<totalSuccess<<
					"\nTotal Command:\t"<<totalCommand<<std::endl;
		assert(totalMove+totalEnter+totalWait == whAGVs.size());
		
		results.totalDeliveries += totalSuccess;
		results.totalMove += totalMove;
		results.totalEnter += totalEnter;
		results.totalWait += totalWait;

		totalDeliveries += totalSuccess;
		for (AGV* a : whAGVs) //Reset AGVs counters
			a->ResetPerformanceCounters();

		//Create and Save replay to buffer
		//reward = (float)totalMove/whAGVs.size();
		{
			float totalInverse = 0;
			float AGVs_on_edges = 0;
			for (AGV* a : whAGVs)
				if (a->GetT2V() != 0){//Make Sure the AGV is on an Edge
					AGVs_on_edges++;
					Search s0(whGraph, a->GetOriginVertex(),
							a->GetNextVertex());
					Search s1(whGraph, a->GetNextVertex(),
							a->GetDestinationVertex());
					totalInverse += s0.PathSearchLenght();
					totalInverse += s1.PathSearchLenght();
				}
			if (AGVs_on_edges)
				reward = 1.5*whAGVs.size()*AGVs_on_edges / totalInverse - totalWait;
			else
				reward = 0;
			assert(!std::isnan(reward));
		}

		//TODO
		//Reward idea: (different weight for each type of edge)
		//           * (Number of AGVs moving on that edge)
		// 					 - w1*(Number of AGVs waiting)
		if(verbose)
			std::cout<<"Reward: "<<reward<<std::endl;

		replay r = {temp_state,cur_state,actions,reward};
		ddpg_maTeam[0]->addToReplayBuffer(r);


		if(ddpg_maTeam[0]->get_replay_buffer_size() > BATCH_SIZE * 2){
			std::vector<replay> miniBatch = ddpg_maTeam[0]->getReplayBufferBatch();

			std::vector<float> Qvals;//Qvals
			std::vector<float> Qprime;//Qprime ( the Y )
			std::vector<std::vector<float>> states; //all states from the batch

			for (size_t i = 0; i < BATCH_SIZE; i++){
				replay b = miniBatch[i];
				std::vector<float> nta = ddpg_maTeam[0]->EvaluateTargetActorNN_DDPG(b.next_state);
				assert(N_EDGES && nta.size() == N_EDGES);
				float y = b.reward + GAMMA *
					ddpg_maTeam[0]->EvaluateTargetCriticNN_DDPG(b.next_state,nta)[0];
				//Generate Qvals and Qprime for Q backprop
				float q = ddpg_maTeam[0]->EvaluateCriticNN_DDPG(b.current_state,b.action)[0];
				Qvals.push_back(q);
				Qprime.push_back(y);
				states.push_back(b.current_state);
			}

			//Update all the NNs
			ddpg_maTeam[0]->updateQCritic(Qvals, Qprime);
			ddpg_maTeam[0]->updateMuActor(states);
			ddpg_maTeam[0]->updateTargetWeights();

		}else
			if(verbose)
				std::cout << "Not enough Replays yet for updating NNs!"<<std::endl;
	}	
	return results;
}

void WarehouseCentralised::InitialiseMATeam(){
	// Initialise NE component and domain housekeeping components of the centralised agent
	vector<size_t> eIDs;
	for (size_t j = 0; j < N_EDGES; j++)
		eIDs.push_back(j);
	iAgent * agent = new iAgent{0, eIDs}; // only one centralised agent controlling all traffic
	whAgents.push_back(agent);
	if (algo == algo_type::ddpg){
		assert(ddpg_maTeam.empty());
		ddpg_maTeam.push_back(new DDPGAgent(N_EDGES*(1+incorporates_time), N_EDGES));
		assert(!ddpg_maTeam.empty());
	}else{
		std::cout << "ERROR: InitialiseMATeam invalid also";
		exit(1);
	}

	nAgents = whAgents.size();
}

