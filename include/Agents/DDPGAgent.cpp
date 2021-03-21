#include "DDPGAgent.h"


DDPGAgent::DDPGAgent(size_t state_space, size_t action_space){
	//Create NNs
	q_criticNN = new NeuralNet(state_space + action_space,
		1, (state_space + action_space) * 2);
	q_target_criticNN = new NeuralNet(state_space + action_space,
		1, (state_space + action_space) * 2);
	mu_actorNN = new NeuralNet(action_space, action_space, action_space*2);
	mu_target_actorNN = new NeuralNet(action_space, action_space, action_space*2);

	//TODO Randomize weights of NNs, Possibly already implemented
	/*
	for (int i=0; i<q_criticNN->GetWeightsA().size(); i++)
		assert(!q_criticNN->GetWeightsA()(i));
	for (int i=0; i<q_criticNN->GetWeightsB().size(); i++)
		assert(!q_criticNN->GetWeightsB()(i));
	for (int i=0; i<mu_actorNN->GetWeightsA().size(); i++)
		assert(!mu_actorNN->GetWeightsA()(i));
	for (int i=0; i<mu_actorNN->GetWeightsB().size(); i++)
		assert(!mu_actorNN->GetWeightsB()(i));
	*/
	//Hopefully this should confirm if it is random
	//TODO yes they are not random enough

	q_target_criticNN->SetWeights(mu_actorNN->GetWeightsA(),
		mu_actorNN->GetWeightsB());
	mu_target_actorNN->SetWeights(mu_actorNN->GetWeightsA(),
		mu_actorNN->GetWeightsB());

	replay_buffer.reserve(REPLAY_BUFFER_SIZE);
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

void DDPGAgent::addToReplayBuffer(replay r){
	replay_buffer.push_back(r);
	//TODO check if replay buffer is full
}

vector<replay> DDPGAgent::getReplayBufferBatch(size_t size){
	std::vector<replay> temp(size);
	for (size_t i = 0; i < size ; i++){
		temp.push_back(replay_buffer[i]);
	}
	return temp;
	//TODO get random minibatch
}