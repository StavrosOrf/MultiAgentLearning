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

void WarehouseCentralised::SimulateEpochDDPG(){
	InitialiseNewEpoch();
	double totalDeliveries = 0;
	std::normal_distribution<double> n_process(0.0, 0.1);
	std::default_random_engine n_generator;

	VectorXd cur_state(N_EDGES * 1);
	VectorXd temp_state(N_EDGES * 1);
	double reward;

	for (int i = 0; i != N_EDGES; i++)
		cur_state[i] = get_edge_utilization()[i];
	print_warehouse_state();

	// each timestep
	for (size_t t = 0; t < nSteps; t++){
		std::cout <<"==============================================================Step - "<<t<<std::endl;
		//Select action
		VectorXd actions = ddpg_maTeam[0]->EvaluateActorNN_DDPG(cur_state);
		// Add Random Noise from process N
		for (size_t n = 0; n < whGraph->GetEdges().size(); n++){
			actions[n] = std::max(0.0,actions[n]+n_process(n_generator));
			actions[n] = std::min(1.0,actions[n]);
			assert(0 <= actions[n] && actions[n] <= 1);
		}

		vector<double> final_costs = baseCosts;
		double maxBaseCost=*std::max_element(baseCosts.begin(), baseCosts.end());		

		for (size_t i = 0; i < nAgents; i++)
			for (size_t j = 0; j < whAgents[i]->eIDs.size(); j++){ // output [0,1] scaled to max base cost
				final_costs[whAgents[i]->eIDs[j]] += actions[j]*maxBaseCost;

				assert(0 <= final_costs[whAgents[i]->eIDs[j]]);
				assert(final_costs[whAgents[i]->eIDs[j]] <= maxBaseCost + baseCosts[whAgents[i]->eIDs[j]]);
			}
		assert(final_costs.size() == N_EDGES);

		UpdateGraphCosts(final_costs);
		replan_AGVs(final_costs);
		transition_AGVs();

		// Traverse
		for (size_t k = 0; k < nAGVs; k++)
			whAGVs[k]->Traverse();

		//for (size_t k = 0; k < nAGVs; k++) whAGVs[k]->DisplayPath();

		//update current state
		for(int i = 0; i < cur_state.size(); i++){
			temp_state[i] = cur_state[i];
			cur_state[i] = get_edge_utilization()[i];
		}
		print_warehouse_state();

		double maxEval = -1;
		vector<size_t> travelStats;
		// Log data
		size_t totalMove = 0;
		size_t totalEnter = 0;
		size_t totalWait = 0;
		size_t totalSuccess = 0;
		size_t totalCommand = 0;
		for (size_t k = 0; k < nAGVs; k++){
			totalMove += whAGVs[k]->GetMoveTime();
			totalEnter += whAGVs[k]->GetEnterTime();
			totalWait += whAGVs[k]->GetWaitTime();
			totalSuccess += whAGVs[k]->GetNumCompleted();
			totalCommand += whAGVs[k]->GetNumCommanded();
		}

		std::cout<<"Stats: \nTotal Move:"<<totalMove<<" \nTotal Enter:"<<totalEnter<<
				"\nTotal wait:"<<totalWait<< "\nTotal Success:"<<totalSuccess<<
				"\nTotal Command:"<<totalCommand<<std::endl;
		assert(totalMove+totalEnter+totalWait == whAGVs.size());

		totalDeliveries += totalSuccess;
		//Reset AGVs counters
		for (size_t k = 0; k < nAGVs; k++)
			whAGVs[k]->ResetPerformanceCounters();
		//Create and Save replay to buffer

		assert(temp_state.size() == N_EDGES && cur_state.size() == N_EDGES);
		reward = (double)totalMove/whAGVs.size();
		std::cout<<"Reward: "<<reward<<std::endl;
		replay r = {temp_state,cur_state,actions,reward};
		//TODO delete oldest replay when full
		ddpg_maTeam[0]->addToReplayBuffer(r);
		//TODO
		//Reward idea: (different weight for each type of edge)
		//           * (Number of AGVs moving on that edge)
		// 					 - w1*(Number of AGVs waiting)

		// We need to give incentive to AGVs to prefer moving
	  // than waiting in heavy traffic nodes
	  // ( maybe target Actor update will fix this)
	  // if not we need to think about that


		//TODO
		if(ddpg_maTeam[0]->replay_buffer.size() > BATCH_SIZE * 2){
			std::vector<replay> miniBatch = ddpg_maTeam[0]->getReplayBufferBatch();

			std::vector<Eigen::VectorXd> trainInputs;
			std::vector<Eigen::VectorXd> trainTargets;

			std::cout << "Y = {";
			for (size_t i = 0; i < BATCH_SIZE; i++){
				replay b = miniBatch[i];

				VectorXd na = ddpg_maTeam[0]->EvaluateTargetActorNN_DDPG(b.next_state);
				assert(b.next_state.size() == N_EDGES && na.size() == N_EDGES);
				assert(ddpg_maTeam[0]->EvaluateTargetCriticNN_DDPG(b.next_state,na).size() ==1);
				double y = b.reward + GAMMA *
					ddpg_maTeam[0]->EvaluateTargetCriticNN_DDPG(b.next_state,na)[0];
				std::cout << y << ", ";

				//Generate trainInputs and trainTargets for Q backprop
				VectorXd input(miniBatch[i].action.size() + miniBatch[i].current_state.size());
				input << miniBatch[i].action , miniBatch[i].current_state;
				trainInputs.push_back(input);
				Eigen::VectorXd t(1);
				t[0] = y;
				assert(t[0] == y && t.size() == 1);
				trainTargets.push_back(t);
			}
			std::cout << '}' << std::endl;

			//Update Q critic
			ddpg_maTeam[0]->updateQCritic(trainInputs, trainTargets);

			//TODO
			//Update actor critic


			//Update target Q critic and Mu target actor critic
			ddpg_maTeam[0]->updateTargetWeights();

		}else
			std::cout << "Not enough Replays yet for updating NN!"<<std::endl;
	}
	std::cout << "End of Simulation with G: "<<totalDeliveries<<std::endl;
	return;
}

void WarehouseCentralised::InitialiseMATeam(){
	// Initialise NE component and domain housekeeping components of the centralised agent
	vector<size_t> eIDs;
	for (size_t j = 0; j < N_EDGES; j++)
		eIDs.push_back(j);
	iAgent * agent = new iAgent{0, eIDs}; // only one centralised agent controlling all traffic
	whAgents.push_back(agent);
	if (algo == algo_type::ddpg){
		ddpg_maTeam.push_back(new DDPGAgent(N_EDGES * 1, N_EDGES));
		assert(!ddpg_maTeam.empty());
	}else{
		std::cout << "ERROR: InitialiseMATeam invalid also";
		exit(1);
	}


	nAgents = whAgents.size();
}

