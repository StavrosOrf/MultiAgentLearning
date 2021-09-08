#include "COMAAgent.hpp"

const int hiddensize = 256;

COMAAgent::COMAAgent(size_t state_space, size_t action_space)
	: muNN(state_space, action_space, hiddensize),
	optimizerMuNN(muNN.parameters(),0.01),
	optimizerQNN(qNN.parameters(),0.01) {}

COMAAgent::~COMAAgent() = default;

void COMAAgent::init_critic_NNs(size_t global_state_space, size_t global_action_space){
	const int hiddensize = 256;

	COMAAgent::qNN = CriticNN(global_state_space+global_action_space, 1, hiddensize);
	COMAAgent::qtNN = CriticNN(global_state_space+global_action_space, 1, hiddensize);

	//copy {Q', Mu'} <- {Q, Mu}
	for (size_t i = 0; i < qNN.parameters().size(); i++)
		qtNN.parameters()[i].set_data(qNN.parameters()[i].detach().clone());

	//assert that copy was successfull
	for (size_t i = 0; i < qNN.parameters().size(); i++ )
		assert(torch::sum(qNN.parameters()[i] == qtNN.parameters()[i]).item<float>() == qNN.parameters()[i].numel());
}

/************************************************************************************************
**Input:a vector [s] of the input state and a vector [a] of input actions			*
**Method:Does a foward pass of the associated NN						*
**Output:Returns a vector of the final nodes of the NN						*
*************************************************************************************************/
std::vector<float> COMAAgent::evaluate_actor_NN(const std::vector<float>& s){
	torch::Tensor t = torch::tensor(std::move(s)).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = muNN.forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}

std::vector<float> COMAAgent::evaluate_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qNN.forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}
// std::vector<float> DDPGAgent::EvaluateTargetActorNN_DDPG(const std::vector<float>& s){
// 	torch::Tensor t = torch::tensor(s).unsqueeze(0);
// 	t = t.to(torch::kFloat32);

// 	torch::Tensor t1 = mutNN->forward(t);
// 	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
// 	return to_return;
// }

// static std::vector<float> COMAAgent::evaluate_target_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
std::vector<float> COMAAgent::evaluate_target_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qtNN.forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}




/************************************************************************************************
 * *Method:updates the network parameters (aka network transition weights) of target networks	*
 * *	by slow updating (with learning rate [TAU]) from non-targer networks			*
//  * ************************************************************************************************/
// void DDPGAgent::updateTargetWeights(){
// 	for (size_t i = 0; i < qNN->parameters().size(); i++ ){
// 		torch::Tensor t = qNN->parameters()[i].detach().clone();
// 		torch::Tensor tt = qtNN->parameters()[i].detach().clone();

// 		qtNN->parameters()[i].set_data(TAU*t + (1-TAU)*tt);
// 	}
// 	for (size_t i = 0; i < muNN->parameters().size(); i++ ){
// 		torch::Tensor t = muNN->parameters()[i].detach().clone();
// 		torch::Tensor tt = mutNN->parameters()[i].detach().clone();

// 		mutNN->parameters()[i].set_data(TAU*t + (1-TAU)*tt);
// 	}

// 	// std::cout<<torch::all(qtNN->parameters()[0].eq(qNN->parameters()[0]))<<std::endl;
// 	// std::cout<<torch::all(mutNN->parameters()[0].eq(muNN->parameters()[0]))<<std::endl;
// }
// void DDPGAgent::updateQCritic(std::vector<float> Qvals, std::vector<float> Qprime,bool verbose){
// 	//TODO try other optimizers too(etc SGD)
// 	// torch::optim::Adam optimizerQNN(qNN->parameters(),0.01);

// 	torch::Tensor QprimeTensor = torch::tensor(Qprime).unsqueeze(0);
// 	torch::Tensor QvalsTensor = torch::tensor(Qvals).unsqueeze(0);

// 	//Update critic loss
// 	torch::Tensor loss= torch::mse_loss(QvalsTensor,QprimeTensor);

// 	optimizerQNN.zero_grad();
// 	loss.backward();
// 	optimizerQNN.step();

// 	// std::cout<<"QVals \t\t Qprime"<<std::endl;
// 	// for (int i = 0; i != batch_size; i++){
// 	// 	std::cout<<Qvals[i]<<"\t\t"<<Qprime[i]<<std::endl;
// 	// }
// 	if (verbose)
// 		std::cout<<"QcriticLoss:\t"<<loss.item<float>()<<std::endl;
// }

// void DDPGAgent::updateMuActorLink(std::vector<std::vector<float>> states,std::vector<std::vector<float>> all_actions,int agentNumber,bool withTime){
// 	//TODO SCALE INPUTS OF Q NETWORK(if necessary)
// 	// torch::optim::Adam optimizerMuNN(muNN->parameters(),0.01);

// 	//Update actor
// 	torch::Tensor states_ = torch::tensor(states[0]).unsqueeze(0);
// 	for (size_t i = 1; i != batch_size; i++){
// 		torch::Tensor temp = torch::tensor(states[i]).unsqueeze(0);
// 		states_ = torch::cat({states_,temp},0);
// 	}
// 	states_ = states_.to(torch::kFloat32);

// 	torch::Tensor states_N;
// 	// make a tensor from only agent's n actions			
// 	if(!withTime){
// 		states_N = states_.slice(1,agentNumber,agentNumber+1);	
// 	}else{
// 		states_N = states_.slice(1,agentNumber,agentNumber+1);
// 		torch::Tensor states_N_t = states_.slice(1,agentNumber+all_actions[0].size(),agentNumber+1+all_actions[0].size());		
// 		states_N = torch::cat({states_N,states_N_t},1);
// 		// std::cout<<states_N<<std::endl;
// 	}
	
// 	torch::Tensor actions = muNN->forward(states_N);

// 	torch::Tensor final_actions = torch::tensor(all_actions[0]).unsqueeze(0);	
// 	final_actions[0][agentNumber] = actions[0][0];
// 	for (size_t i = 1; i != batch_size; i++){
// 		torch::Tensor temp = torch::tensor(all_actions[i]).unsqueeze(0);		
// 		temp[0][agentNumber] = actions[i][0];		
// 		final_actions = torch::cat({final_actions,temp},0);		
// 	}

// 	torch::Tensor input = torch::cat({states_,final_actions},1);
// 	// std::cout<<actions<<std::endl;
// 	// std::cout<<all_actions<<std::endl;
// 	// std::cout<<final_actions<<std::endl;

// 	// std::cout<<input<<std::endl;
// 	// std::cout<<agentNumber<<std::endl;	
// 	// std::cout<<qNN->forward(input)<<std::endl;
// 	torch::Tensor policy_loss = -torch::mean(qNN->forward(input));

// 	optimizerMuNN.zero_grad();
// 	policy_loss.backward();
// 	optimizerMuNN.step();

// 	// std::cout<<"ActorLoss:\t"<<policy_loss.item<float>()<<std::endl;
// }

// void DDPGAgent::updateMuActor(std::vector<std::vector<float>> states,bool verbose){
// 	//TODO SCALE INPUTS OF Q NETWORK(if necessary)
// 	// torch::optim::Adam optimizerMuNN = torch::optim::Adam(muNN->parameters(),0.01);

// 	//Update actor
// 	torch::Tensor states_ = torch::tensor(states[0]).unsqueeze(0);
// 	for (size_t i = 1; i != batch_size; i++){
// 		torch::Tensor temp = torch::tensor(states[i]).unsqueeze(0);
// 		states_ = torch::cat({states_,temp},0);
// 	}
// 	states_ = states_.to(torch::kFloat32);
// 	// std::cout<<states_<<std::endl;	
// 	torch::Tensor actions = muNN->forward(states_);
// 	torch::Tensor input = torch::cat({states_,actions},1);

// 	// std::cout<<actions<<std::endl;

// 	// std::cout<<"Q input:"<<input<<std::endl;
// 	// std::cout<<qNN->forward(input)<<std::endl;

// 	torch::Tensor policy_loss = -torch::mean(qNN->forward(input));

// 	optimizerMuNN.zero_grad();
// 	policy_loss.backward();
// 	optimizerMuNN.step();

// 	if(verbose)
// 		std::cout<<"ActorLoss:\t"<<policy_loss.item<float>()<<std::endl;
// }

/************************************************************************************************
**Method:Print Various Metrics about the agent's neural Networks				*
*************************************************************************************************/
void COMAAgent::printAboutNN(){
	for (size_t i = 0; i < muNN.parameters().size(); i++ ){
		torch::Tensor t = muNN.parameters()[i].detach().clone();
		// torch::Tensor tt = mutNN->parameters()[i].detach().clone();

		std::cout<<" MuNN "<<i<<": "<< torch::sum(t).item<float>();
			// <<" MutNN "<<i<<": "<< torch::sum(tt).item<float>();
		std::cout << '\n';
	}

	// for (size_t i = 0; i < qNN->parameters().size(); i++ ){

	// 	torch::Tensor q = qNN->parameters()[i].detach().clone();
	// 	torch::Tensor qq = qtNN->parameters()[i].detach().clone();

	// 	std::cout<<" QuNN "<<i<<": "<< torch::sum(q).item<float>()
	// 		<<" QutNN "<<i<<": "<< torch::sum(qq).item<float>();
	// 	std::cout << '\n';
	// }
	
}
