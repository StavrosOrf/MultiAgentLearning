#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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
	protected:
		struct replay
		{
			vector<double> current_state;
			vector<double> next_state;
			vector<double> action;
			vector<double> rewards;
	
		};

		NeuralNet q_criticNN;
		NeuralNet q_target_criticNN;
		NeuralNet mu_actorNN;
		NeuralNet mu_target_actorNN;

		vector<replay> replay_buffer; //TODO kallinteris se t alocator
		//TODO add "INITIALIZE random process N"

};

#endif // DDPG_AGENT_H_
