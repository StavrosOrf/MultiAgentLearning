#include "DDPGAgent.h"


DDPGAgent::DDPGAgent(size_t state_space, size_t action_space){
	//Create NNs
	
	qNN = new Net(state_space + action_space,
		1, (state_space + action_space) * 2);
	qtNN = new Net(state_space + action_space,
		1, (state_space + action_space) * 2);
	//qtNN = (Net*) malloc(sizeof(Net));
	muNN = new Net(action_space, action_space, action_space*2);
	mutNN = new Net(action_space, action_space, action_space*2);

	//copy {Q', Mu'} <- {Q, Mu}
	//assert (sizeof(Net) > sizeof(size_t));
	qtNN->weightsA = qNN->weightsA.clone();
	qtNN->weightsB = qNN->weightsB.clone();
	mutNN->weightsA = muNN->weightsA.clone();
	mutNN->weightsB = muNN->weightsB.clone();
	//TODO verify clone
	//std::copy(qNN, qNN+sizeof(Net), qtNN);
	//std::copy(muNN, muNN+sizeof(Net), mutNN);
	//qtNN->weightsA[0][0].item<double>() = 0.1;
	//assert(qtNN->weightsA[0].item<double> != qNN->weightsA[0].item<double>);

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
std::vector<double> DDPGAgent::EvaluateActorNN_DDPG(std::vector<double> s){	
	torch::Tensor t = torch::tensor(s).unsqueeze(0);
	t = t.to(torch::kFloat32);	

	torch::Tensor t1 = muNN->forward(t);	
	std::vector<double> to_return;
	for (int i = 0; i != t1.numel(); i++)
			to_return.push_back(t1[0][i].item<double>());
	assert(to_return.size() == t1.numel());	
	return to_return;
}
std::vector<double> DDPGAgent::EvaluateCriticNN_DDPG(std::vector<double> s,std::vector<double> a){

	std::vector<double> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());
	
	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);	

	torch::Tensor t1 = qNN->forward(t);	
	std::vector<double> to_return;	
	for (int i = 0; i != t1.numel(); i++)
		to_return.push_back(t1[0][i].item<double>());			

	return to_return;
}
std::vector<double> DDPGAgent::EvaluateTargetActorNN_DDPG(std::vector<double> s){
	// torch::Tensor t = torch::from_blob(s.data(), {1, s.size()});
	torch::Tensor t = torch::tensor(s).unsqueeze(0);
	t = t.to(torch::kFloat32);	

	torch::Tensor t1 = mutNN->forward(t);
	std::vector<double> to_return;
	for (int i = 0; i != t1.numel(); i++)
		to_return.push_back(t1[0][i].item<double>());
	assert(to_return.size() == t1.numel());
	return to_return;
}
std::vector<double> DDPGAgent::EvaluateTargetCriticNN_DDPG(std::vector<double> s,  std::vector<double> a){
	std::vector<double> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);	

	torch::Tensor t1 = qtNN->forward(t);
	std::vector<double> to_return;
	for (int i = 0; i != t1.numel(); i++)
			to_return.push_back(t1[0][i].item<double>());
	return to_return;
}

/************************************************************************************************
 * *Input:	a replay [r]									*
 * *Method:Adds [r] to the [replay_buffer], if [replay_buffer] is full it evicts a tuple	*
 * ************************************************************************************************/
void DDPGAgent::addToReplayBuffer(replay r){
	assert(r.next_state.size() == r.current_state.size() && r.current_state.size() == 38);

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
vector<replay> DDPGAgent::getReplayBufferBatch(size_t size){
	std::vector<replay> temp;
	assert(replay_buffer.size() >= size);

	for (int i = 0; i != size; i++){
		int r = rand()%size;
		temp.push_back(replay_buffer[r]);
		replay_buffer.erase(replay_buffer.begin()+r);
	}
	assert(temp.size() == size);
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
	// MatrixXd QtA = TAU*q_criticNN->GetWeightsA() + (1-TAU)*q_target_criticNN->GetWeightsA();
	// std::cout<<muNN->parameters()<<std::endl;
	// std::cout<<mutNN->parameters()<<std::endl;
	// std::cout<<qNN->parameters().size()<<std::endl;		
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
	
	// assert(torch::all(mutNN->parameters()[0].eq(muNN->parameters()[0])));
	// assert(torch::all(qtNN->parameters()[0].eq(qNN->parameters()[0])));	
	// std::cout<<torch::all(qtNN->parameters()[0].eq(qNN->parameters()[0]))<<std::endl;	
	// std::cout<<torch::all(mutNN->parameters()[0].eq(muNN->parameters()[0]))<<std::endl;	
}
void DDPGAgent::updateQCritic(std::vector<double> Qvals, std::vector<double> Qprime){
	
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

void DDPGAgent::updateMuActor(std::vector<std::vector<double>> states){
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
