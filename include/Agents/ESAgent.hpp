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


class ESAgent{
	public:
		ESAgent(size_t state_space, size_t action_space);
		~ESAgent();

		void updateNNWeights(float scalar);
		void setNN(esNN* nn);
		std::vector<float> evaluateNN(const std::vector<float>& s);
		esNN* NN;
	protected:



};