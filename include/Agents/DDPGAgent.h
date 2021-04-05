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
#define GAMMA 0.90
#define TAU 0.02
#define TRAINING_STEP 1

struct replay{
	std::vector<float> current_state;
	std::vector<float> next_state;
	std::vector<float> action;
	float reward;
};

struct Net : torch::nn::Module {
	Net(int32_t numIn, int32_t numOut, int32_t numHid) {
		weightsA = register_parameter("input", torch::rand({numIn, numHid}));
		weightsB = register_parameter("output", torch::rand({numHid, numOut}));		
	}
	torch::Tensor forward(torch::Tensor input) {
		torch::Tensor hidden_layer, output_layer;
		//hidden_layer = (torch::mm(input, weightsA));
		//output_layer = (torch::mm(hidden_layer, weightsB));
		hidden_layer = torch::tanh(torch::mm(input, weightsA));
		output_layer = torch::tanh(torch::mm(hidden_layer, weightsB));
		return output_layer;
	}
	torch::Tensor weightsA, weightsB;
	// torch::nn::Linear weightsA,weightsB;
};

class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space,size_t global_state_space,size_t global_action_space );
		~DDPGAgent();

		std::vector<float> EvaluateCriticNN_DDPG(const std::vector<float> s, const std::vector<float> a);
		std::vector<float> EvaluateActorNN_DDPG(const std::vector<float> s);
		std::vector<float> EvaluateTargetActorNN_DDPG(const std::vector<float> s);
		std::vector<float> EvaluateTargetCriticNN_DDPG(const std::vector<float> s, const std::vector<float> a);

		static std::vector<replay> getReplayBufferBatch(size_t size = batch_size);
		static void addToReplayBuffer(replay r);
		static size_t get_replay_buffer_size(){return replay_buffer.size();}

		void updateTargetWeights();
		void updateQCritic(const std::vector<float> Qvals, const std::vector<float> Qprime);
		void updateMuActor(const std::vector<std::vector<float>> states);
		void updateMuActorLink(std::vector<std::vector<float>> states,std::vector<std::vector<float>> all_actions,int agentNumber,bool withTime);

		void printAboutNN();

		static void set_batch_size(int i){batch_size = i;}
		static size_t get_batch_size(){return batch_size;}
	protected:
		Net* qNN, *qtNN;
		Net* muNN, *mutNN;
		inline static std::vector<replay> replay_buffer;
		inline static size_t batch_size;
};

#endif // DDPG_AGENT_H_
