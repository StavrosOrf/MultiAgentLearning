#include "DDPGAgent.h"


DDPGAgent::DDPGAgent(size_t state_space, size_t action_space){
	//Create NNs
	qNN = new Net(state_space + action_space,
		1, (state_space + action_space) * 2);
	qtNN = new Net(state_space + action_space,
		1, (state_space + action_space) * 2);
	muNN = new Net(action_space, action_space, action_space*2);
	mutNN = new Net(action_space, action_space, action_space*2);

	//copy {Q', Mu'} <- {Q, Mu}
	for (int i = 0; i < qNN->parameters().size(); i++ )
		qtNN->parameters()[i] = qNN->parameters()[i].clone();
	for (int i = 0; i < muNN->parameters().size(); i++ )
		mutNN->parameters()[i] = muNN->parameters()[i].clone();

	replay_buffer.reserve(REPLAY_BUFFER_SIZE);
	// torch::optim::SGD optimezerQNN(qNN->parameters(),0.01);
	// torch::optim::SGD optimezerQtNN(qNN->parameters(),0.01);
	// torch::optim::SGD optimezerMuNN(qNN->parameters(),0.01);
	// torch::optim::SGD optimezerMuTNN(qNN->parameters(),0.01);
}

DDPGAgent::~DDPGAgent(){
	delete(qNN);
	delete(qtNN);
	delete(muNN);
	delete(mutNN);
	qNN = qtNN = muNN = mutNN = NULL;
}

/************************************************************************************************
 * *Input:a vector [s] of the input state and a vector [a] of input actions			*
 * *Method:Does a foward pass of the associated NN						*
 * *Output:Returns a vector of the final nodes of the NN					*
 * ************************************************************************************************/
std::vector<float> DDPGAgent::EvaluateActorNN_DDPG(std::vector<float> s){
	torch::Tensor t = torch::tensor(s).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = muNN->forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}
std::vector<float> DDPGAgent::EvaluateCriticNN_DDPG(std::vector<float> s,std::vector<float> a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qNN->forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}
std::vector<float> DDPGAgent::EvaluateTargetActorNN_DDPG(std::vector<float> s){
	torch::Tensor t = torch::tensor(s).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = mutNN->forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}
std::vector<float> DDPGAgent::EvaluateTargetCriticNN_DDPG(std::vector<float> s,  std::vector<float> a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qtNN->forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}

/************************************************************************************************
 * *Input:	a replay [r]									*
 * *Method:Adds [r] to the [replay_buffer], if [replay_buffer] is full it evicts a tuple	*
 * ************************************************************************************************/
void DDPGAgent::addToReplayBuffer(replay r){
	assert(r.next_state.size() == r.current_state.size());

	if (replay_buffer.size() < REPLAY_BUFFER_SIZE)
		replay_buffer.push_back(r);
	else
		replay_buffer[rand()%REPLAY_BUFFER_SIZE] = r;
}
/************************************************************************************************
 * *Input:[size] of batch to return								*
 * *Method:Selects a non-inclusive (with unique items) minibanch from the replay buffer		*
 * *Output:Returns a non-inclusive miniBatch of [size]						*
 * ************************************************************************************************/
std::vector<replay> DDPGAgent::getReplayBufferBatch(size_t size){
	assert(replay_buffer.size() >= size);
	std::vector<replay> temp;

	for (int i = 0; i != size; i++){
		int r = rand()%replay_buffer.size();
		temp.push_back(replay_buffer[r]);
		replay_buffer.erase(replay_buffer.begin()+r);
	}
	for (size_t i = 0; i < size; i++)
		replay_buffer.push_back(temp[i]);

	assert(temp.size() == size);
	return temp;
}

/************************************************************************************************
 * *Method:updates the network parameters (aka network transition weights) of target networks	*
 * *	by slow updating (with learning rate [TAU]) from non-targer networks			*
 * ************************************************************************************************/
void DDPGAgent::updateTargetWeights(){
	for (int i = 0; i < qNN->parameters().size(); i++ ){
		torch::Tensor t = qNN->parameters()[i].detach().clone();
		torch::Tensor tt = qtNN->parameters()[i].detach().clone();

		qtNN->parameters()[i].set_data(TAU*t + (1-TAU)*tt);
	}
	for (int i = 0; i < muNN->parameters().size(); i++ ){
		torch::Tensor t = muNN->parameters()[i].detach().clone();
		torch::Tensor tt = mutNN->parameters()[i].detach().clone();

		mutNN->parameters()[i].set_data(TAU*t + (1-TAU)*tt);
	}

	// std::cout<<torch::all(qtNN->parameters()[0].eq(qNN->parameters()[0]))<<std::endl;
	// std::cout<<torch::all(mutNN->parameters()[0].eq(muNN->parameters()[0]))<<std::endl;
}
void DDPGAgent::updateQCritic(std::vector<float> Qvals, std::vector<float> Qprime){
	//TODO try other optimizers too(etc SGD)
	torch::optim::Adam optimezerQNN(qNN->parameters(),0.01);

	torch::Tensor QprimeTensor = torch::tensor(Qprime).unsqueeze(0);
	torch::Tensor QvalsTensor = torch::tensor(Qvals).unsqueeze(0);

	//Update critic loss
	torch::Tensor loss= torch::mse_loss(QvalsTensor,QprimeTensor);

	optimezerQNN.zero_grad();
	loss.backward();
	optimezerQNN.step();

	// std::cout<<"QVals \t\t Qprime"<<std::endl;
	// for (int i = 0; i != BATCH_SIZE; i++){
	// 	std::cout<<Qvals[i]<<"\t\t"<<Qprime[i]<<std::endl;
	// }

	// std::cout<<"QcriticLoss:\t"<<loss.item<float>()<<std::endl;
}

void DDPGAgent::updateMuActor(std::vector<std::vector<float>> states){
	//TODO SCALE INPUTS OF Q NETWORK(if necessary)
	torch::optim::Adam optimezerMuNN(muNN->parameters(),0.01);

	//Update actor
	torch::Tensor states_ = torch::tensor(states[0]).unsqueeze(0);
	for (int i = 1; i != BATCH_SIZE; i++){
		torch::Tensor temp = torch::tensor(states[i]).unsqueeze(0);
		states_ = torch::cat({states_,temp},0);
	}
	states_ = states_.to(torch::kFloat32);

	torch::Tensor actions = muNN->forward(states_);
	torch::Tensor input = torch::cat({states_,actions},1);

	// std::cout<<actions<<std::endl;
	// std::cout<<input<<std::endl;
	// std::cout<<qNN->forward(input)<<std::endl;

	torch::Tensor policy_loss = -torch::mean(qNN->forward(input));

	optimezerMuNN.zero_grad();
	policy_loss.backward();
	optimezerMuNN.step();

	// std::cout<<"ActorLoss:\t"<<policy_loss.item<float>()<<std::endl;
}
