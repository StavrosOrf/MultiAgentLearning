#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <Eigen/Eigen>
#include <cassert>
#include <torch/torch.h>
#include "Learning/NeuralNet.h"

using std::vector;
using std::string;

#define REPLAY_BUFFER_SIZE 5000
#define GAMMA 0.95
#define TAU 0.01
#define BATCH_SIZE 5

struct replay{
	//TODO make int later
	std::vector<double> current_state;
	std::vector<double> next_state;
	std::vector<double> action;
	double reward;
};

struct Net : torch::nn::Module {
	Net(int32_t numIn, int32_t numOut, int32_t numHid) {
		weightsA = register_parameter("input", torch::randn({numIn, numHid}));
		weightsB = register_parameter("output", torch::randn({numHid, numOut}));
	}
	torch::Tensor forward(torch::Tensor input) {
		torch::Tensor hidden_layer, output_layer;
		hidden_layer = torch::relu(torch::mm(input, weightsA));
		output_layer = torch::relu(torch::mm(hidden_layer, weightsB));		
		return output_layer;
	}
	torch::Tensor weightsA, weightsB;
};


class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space);
		~DDPGAgent();
		std::vector<double> EvaluateCriticNN_DDPG(std::vector<double> s,std::vector<double> a);
		std::vector<double> EvaluateActorNN_DDPG(std::vector<double> s);
		std::vector<double> EvaluateTargetActorNN_DDPG(std::vector<double> s);
		std::vector<double> EvaluateTargetCriticNN_DDPG(std::vector<double> s,std::vector<double> a);
		void ResetEpochEvals();

		vector<replay> getReplayBufferBatch(size_t size = BATCH_SIZE);
		void addToReplayBuffer(replay r);
		vector<replay> replay_buffer;

		void updateTargetWeights();
		void updateQCritic(std::vector<double> Qvals, std::vector<double> Qprime,std::vector<std::vector<double>> states);
	protected:

		NeuralNet * q_criticNN __attribute__ ((deprecated));
		NeuralNet * q_target_criticNN __attribute__ ((deprecated));;
		NeuralNet * mu_actorNN __attribute__ ((deprecated));;
		NeuralNet * mu_target_actorNN __attribute__ ((deprecated));

		Net* qNN;
		Net* qtNN;
		Net* muNN;
		Net* mutNN;


		// Net* qtNN;
		// Net* muNN;
		// Net* mutNN;
};

#endif // DDPG_AGENT_H_
