#pragma once

#include <vector>

struct experience_replay{
	std::vector<float> current_state;
	std::vector<float> next_state;
	std::vector<float> action;
	float reward;
};

struct experience_replayDQN{
	std::vector<float> current_state;
	std::vector<float> next_state;
	std::vector<float> action;
	std::vector<float> reward;
};