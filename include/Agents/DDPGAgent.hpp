#ifndef DDPG_AGENT_H_
#define DDPG_AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/serialize/input-archive.h>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <torch/torch.h>
#include <cmath>
#include <cstdio>
#include "nn_modules.hpp"

#define REPLAY_BUFFER_SIZE 1024*1024
#define GAMMA 0.99
#define TAU 0.01
#define TRAINING_STEP 1

struct experience_replay{
	std::vector<float> current_state;
	std::vector<float> next_state;
	std::vector<float> action;
	float reward;
};

class DDPGAgent{
	public:
		DDPGAgent(size_t state_space, size_t action_space,size_t global_state_space,size_t global_action_space );
		~DDPGAgent();

		std::vector<float> EvaluateCriticNN_DDPG(const std::vector<float>& s, const std::vector<float>& a);
		std::vector<float> EvaluateActorNN_DDPG(const std::vector<float>& s);
		std::vector<float> EvaluateTargetActorNN_DDPG(const std::vector<float>& s);
		std::vector<float> EvaluateTargetCriticNN_DDPG(const std::vector<float>& s, const std::vector<float>& a);

		static std::vector<experience_replay> getReplayBufferBatch(size_t size = batch_size);
		static void addToReplayBuffer(experience_replay r);
		static size_t get_replay_buffer_size(){return replay_buffer.size();}

		void updateTargetWeights();
		void updateQCritic(const std::vector<float> Qvals, const std::vector<float> Qprime,bool verbose);
		void updateMuActor(const std::vector<std::vector<float>> states, bool verbose);
		void updateMuActorLink(std::vector<std::vector<float>> states,std::vector<std::vector<float>> all_actions,int agentNumber,bool withTime);

		void printAboutNN();

		static void set_batch_size(int i){batch_size = i;}
		const static size_t get_batch_size() {return batch_size;}
		static void clear_replar_buffer(){replay_buffer.clear();}
	protected:
		CriticNN* qNN, *qtNN;
		ActorNN* muNN, *mutNN;
		inline static std::vector<experience_replay> replay_buffer;
		inline static size_t batch_size;

		Net* temp = new Net(1, 1, 1*2);

		// We need a global optimizer, not a new one in each step!!!!!!!!
		torch::optim::Adam optimizerMuNN = torch::optim::Adam(temp->parameters(),0.01);
		torch::optim::Adam optimizerQNN = torch::optim::Adam(temp->parameters(),0.01);
};

#endif // DDPG_AGENT_H_
