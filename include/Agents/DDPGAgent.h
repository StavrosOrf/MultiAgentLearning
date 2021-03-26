#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <Eigen/Eigen>
#include <torch/torch.h>
#include "Learning/NeuralNet.h"

using std::vector;
using std::string;

#define REPLAY_BUFFER_SIZE 5000
#define GAMMA 0.95
#define TAU 0.01
#define BATCH_SIZE 5

struct replay{
	VectorXd current_state;
	VectorXd next_state;
	VectorXd action;
	double reward;
};

struct Net : torch::nn::Module {
  Net(int32_t numIn, int32_t numOut, int32_t numHid) {
    input_N = register_parameter("input", torch::randn({numIn, numHid}));
    output_N = register_parameter("output", torch::randn({numHid, numOut}));
  }
  torch::Tensor forward(torch::Tensor input) {
    return torch::addmm(input_N, input, output_N);
  }
  torch::Tensor input_N, output_N;
};


class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space);
		~DDPGAgent();
		VectorXd EvaluateCriticNN_DDPG(VectorXd s,VectorXd a);
		VectorXd EvaluateActorNN_DDPG(VectorXd s);
		VectorXd EvaluateTargetActorNN_DDPG(VectorXd s);
		VectorXd EvaluateTargetCriticNN_DDPG(VectorXd s,VectorXd a);
		void ResetEpochEvals();

		vector<replay> getReplayBufferBatch(size_t size = BATCH_SIZE);
		void addToReplayBuffer(replay r);
		vector<replay> replay_buffer;

		void updateTargetWeights();
		void updateQCritic(vector<VectorXd> trainInputs, vector<VectorXd> trainTargets);
	protected:

		NeuralNet * q_criticNN __attribute__ ((deprecated));
		NeuralNet * q_target_criticNN __attribute__ ((deprecated));;
		NeuralNet * mu_actorNN __attribute__ ((deprecated));;
		NeuralNet * mu_target_actorNN __attribute__ ((deprecated));

		Net* qNN;
		Net* qtNN;
		Net* muNN;
		Net* mutNN;
};

#endif // DDPG_AGENT_H_
