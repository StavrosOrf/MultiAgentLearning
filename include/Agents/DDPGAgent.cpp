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


void DDPGAgent::ResetEpochEvals(){
	
}

VectorXd DDPGAgent::EvaluateActorNN_DDPG(VectorXd s){
	return mu_actorNN->EvaluateNN(s);
}

VectorXd DDPGAgent::EvaluateCriticNN_DDPG(VectorXd s){
	return q_criticNN->EvaluateNN(s);
}

VectorXd DDPGAgent::EvaluateTargetActorNN_DDPG(VectorXd s){
	return mu_target_actorNN->EvaluateNN(s);
}

VectorXd DDPGAgent::EvaluateTargetCriticNN_DDPG(VectorXd s){
	return q_target_criticNN->EvaluateNN(s);
}	

