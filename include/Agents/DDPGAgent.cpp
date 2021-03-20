#include "DDPGAgent.h"


DDPGAgent::DDPGAgent(size_t state_space, size_t action_space){
	//Create NNs
	q_criticNN = new NeuralNet(state_space + action_space,
		1, state_space + action_space);
	q_target_criticNN = new NeuralNet(state_space + action_space,
		1, state_space + action_space);
	mu_actorNN = new NeuralNet(action_space, action_space, action_space*2);
	mu_target_actorNN = new NeuralNet(action_space, action_space, action_space*2);

	//Randomize weights of NNs
	//Possibly already implemented
	//TODO targets equal with critics(memcpy)



}

DDPGAgent::~DDPGAgent(){
}


void ResetEpochEvals(){
	
}