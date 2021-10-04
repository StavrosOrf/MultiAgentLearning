#pragma once

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
#include "Agents/experience_replay.hpp"
#include "Agents/experience_replay_buffer.hpp"

namespace DQN_consts{ 
	const static size_t replay_buffer_size = 200*100; //1024*1024
	const float gamma = 0.99;
	const float a = 0.001; // Critic Learning Rate
	const size_t reset_step = 10; // reset q_t every C steps
	const size_t actor_samples = 10;
	const size_t simulation_steps = 200;
	const int actions_size = 7;
	// const int actions[29] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28};
	const int actions[7] = {0,1,2,4,8,16,32};
	const int batch_size = 50;
}

class DQNAgent{
	public:
		DQNAgent(size_t state_space, size_t action_space);
		~DQNAgent() = default;

		[[nodiscard, gnu::pure]] std::vector<float> evaluate_critic_NN(const std::vector<float>& s, const std::vector<float>& a);
		[[nodiscard, gnu::pure]] std::vector<float> evaluate_target_critic_NN(const std::vector<float>& s, const std::vector<float>& a);

		void reset_target_critic();
		void trainCritic(const std::vector<experience_replayDQN>& samples,const int agentNumber);
		// void updateTargetWeights();
		// void updateQCritic(const std::vector<float> Qvals, const std::vector<float> Qprime,bool verbose);
		// void updateMuActor(const std::vector<std::vector<float>> states, bool verbose);
		// void updateMuActorLink(std::vector<std::vector<float>> states,std::vector<std::vector<float>> all_actions,int agentNumber,bool withTime);

		[[maybe_unused]] void printAboutNN();
		// static train_Critic(torch::Tensor critic_loss);
		// void set_batch_size(size_t i){batch_size = i;}
		// size_t get_batch_size() {return batch_size;}

		
	protected:
		CriticNN qNN, qtNN;
		torch::optim::RMSprop qOptimizer, qtOptimizer;
};

