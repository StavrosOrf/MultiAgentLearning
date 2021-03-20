#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <Eigen/Eigen>
#include "Learning/NeuralNet.h"

using std::vector ;
using std::string ;

#define REPLAY_BUFFER_SIZE 5000
#define GAMMA 0.95
#define TAU 0.01
#define BATCH_SIZE 50


class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space);
		~DDPGAgent();
		VectorXd EvaluateCriticNN_DDPG(VectorXd s);
		VectorXd EvaluateActorNN_DDPG(VectorXd s);
		VectorXd EvaluateTargetActorNN_DDPG(VectorXd s);
		VectorXd EvaluateTargetCriticNN_DDPG(VectorXd s);		
		void ResetEpochEvals() ;
	protected:
		struct replay
		{
			vector<double> current_state;
			vector<double> next_state;
			vector<double> action;
			vector<double> rewards;
	
		};

		NeuralNet * q_criticNN;
		NeuralNet * q_target_criticNN;
		NeuralNet * mu_actorNN;
		NeuralNet * mu_target_actorNN;

		vector<replay> replay_buffer;

};

#endif // DDPG_AGENT_H_
