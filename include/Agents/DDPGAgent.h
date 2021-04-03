#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <torch/torch.h>

#define REPLAY_BUFFER_SIZE 100000000
#define GAMMA 0.99
#define TAU 0.005
#define BATCH_SIZE 5

struct replay{
	std::vector<float> current_state;
	std::vector<float> next_state;
	std::vector<float> action;
	float reward;
};

struct Net : torch::nn::Module {
	Net(int32_t numIn, int32_t numOut, int32_t numHid) {
		weightsA = register_parameter("input", torch::rand({numIn, numHid}))*0.013;
		// weightsBa = register_parameter("hidden", torch::rand({numHid, numHid}))*0.013;
		// weightsBb = register_parameter("hidden", torch::rand({numHid, numHid}))*0.013;
		weightsB = register_parameter("output", torch::rand({numHid, numOut}))*0.013;
	}
	torch::Tensor forward(torch::Tensor input) {
		torch::Tensor hidden_layer, output_layer;
		//hidden_layer = torch::relu(torch::mm(input, weightsA));
		//output_layer = torch::relu(torch::mm(hidden_layer, weightsB));
		//hidden_layer = torch::tanh(torch::mm(input, weightsA));
		//output_layer = torch::tanh(torch::mm(hidden_layer, weightsB));
		hidden_layer = (torch::mm(input, weightsA));
		output_layer = (torch::mm(hidden_layer, weightsB));
		return output_layer;
	}
	torch::Tensor weightsA, weightsB;
};


class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space);
		~DDPGAgent();
		std::vector<float> EvaluateCriticNN_DDPG(const std::vector<float> s, const std::vector<float> a);
		std::vector<float> EvaluateActorNN_DDPG(const std::vector<float> s);
		std::vector<float> EvaluateTargetActorNN_DDPG(const std::vector<float> s);
		std::vector<float> EvaluateTargetCriticNN_DDPG(const std::vector<float> s, const std::vector<float> a);

		std::vector<replay> getReplayBufferBatch(size_t size = BATCH_SIZE);
		void addToReplayBuffer(replay r);
		size_t get_replay_buffer_size(){return replay_buffer.size();}

		void updateTargetWeights();
		void updateQCritic(const std::vector<float> Qvals, const std::vector<float> Qprime);
		void updateMuActor(const std::vector<std::vector<float>> states);
		void printAboutNN();
	protected:
		Net* qNN;
		Net* qtNN;
		Net* muNN;
		Net* mutNN;
		std::vector<replay> replay_buffer;
};

#endif // DDPG_AGENT_H_
