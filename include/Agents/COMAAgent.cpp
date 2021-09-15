#include "COMAAgent.hpp"

const int hiddensize = 64;

COMAAgent::COMAAgent(size_t state_space, size_t action_space):
	muNN(state_space, action_space, hiddensize),
	optimizerMuNN(muNN.parameters(), COMA_consts::tau_mu)
	{}

COMAAgent::~COMAAgent() = default;

void COMAAgent::init_critic_NNs(size_t global_state_space, size_t global_action_space){

	const int hiddensize = 64;

	COMAAgent::qNN = CriticNN(global_state_space + global_action_space + 1,
		COMA_consts::actions_size, hiddensize);
	COMAAgent::qtNN = CriticNN(global_state_space + global_action_space + 1,
		COMA_consts::actions_size, hiddensize);

	optimizerQNN = std::make_unique<torch::optim::Adam>(qNN.parameters(), COMA_consts::tau_q);

	//copy {Q', Mu'} <- {Q, Mu}
	for (size_t i = 0; i < qNN.parameters().size(); i++)
		qtNN.parameters()[i].set_data(qNN.parameters()[i].detach().clone());

	//assert that copy was successful
	for (size_t i = 0; i < qNN.parameters().size(); i++ )
		assert(torch::sum(qNN.parameters()[i] == qtNN.parameters()[i]).item<float>() == qNN.parameters()[i].numel());
}

/************************************************************************************************
**Input:a vector [s] of the input state and a vector [a] of input actions			*
**Method:Does a forward pass of the associated NN						*
**Output:Returns a vector of the final nodes of the NN						*
*************************************************************************************************/
std::vector<float> COMAAgent::evaluate_actor_NN(const std::vector<float>& s){
	torch::Tensor t = torch::tensor(std::move(s)).unsqueeze(0);
	// std::cout<<"-- "<<t<<std::endl;
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = muNN.forward(t);	
	// std::cout<<t1<<std::endl;
	// std::cout<<torch::argmax(t1)<<std::endl;
	std::vector<float> to_return(t1.data_ptr<float>(), t1.data_ptr<float>() + t1.numel());
	// std::vector<float> to_return;
	// to_return.push_back(COMA_consts::actions[torch::argmax(t1).item<int>()]);
	// std::cout<<to_return<<std::endl;
	return to_return;
}

torch::Tensor COMAAgent::evaluate_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());

	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = torch::reshape(t,{2, static_cast<int>(input.size()/2)});
	t = torch::transpose(t,0,1);
	t = t.to(torch::kFloat32);
	// std::cout<<t<<std::endl;
	torch::Tensor t1 = qNN.forward(t);
	// std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return t1;
}




// std::vector<float> DDPGAgent::EvaluateTargetActorNN_DDPG(const std::vector<float>& s){
// 	torch::Tensor t = torch::tensor(s).unsqueeze(0);
// 	t = t.to(torch::kFloat32);

// 	torch::Tensor t1 = mutNN->forward(t);
// 	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
// 	return to_return;
// }

// static std::vector<float> COMAAgent::evaluate_target_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
torch::Tensor COMAAgent::evaluate_target_critic_NN(const std::vector<float>& s, const std::vector<float>& a){
	std::vector<float> input;
	// for (int i = 0; i < s.size(); ++i)
	// {
	// 	std::cout<<s[i]<<" "<<std::endl;
	// }
	// std::cout<<s<<std::endl;
	// std::cout<<a<<std::endl;

	input.insert(input.begin(),s.begin(),s.end());
	input.insert(input.end(),a.begin(),a.end());


	// std::cout<<input<<std::endl;
	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = torch::reshape(t,{2, static_cast<int>(input.size()/2)});
	t = torch::transpose(t,0,1);
	// std::cout<<t<<std::endl;
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qtNN.forward(t);
	// std::cout<<t1<<std::endl;
	// std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return t1;
}

void COMAAgent::reset_target_critic(){
	for (size_t i = 0; i < qNN.parameters().size(); i++ ){
		torch::Tensor t = qNN.parameters()[i].detach().clone();
		// torch::Tensor tt = qtNN.parameters()[i].detach().clone();
		qtNN.parameters()[i].set_data(t);
	}
}


/************************************************************************************************
**Method:Print Various Metrics about the agent's neural Networks				*
*************************************************************************************************/
void COMAAgent::printAboutNN(){
	for (size_t i = 0; i < muNN.parameters().size(); i++ ){
		torch::Tensor t = muNN.parameters()[i].detach().clone();

		std::cout<<" MuNN "<<i<<": "<< torch::sum(t).item<float>() << '\n';
	}
	// for (size_t i = 0; i < qNN->parameters().size(); i++ ){
	// 	torch::Tensor q = qNN->parameters()[i].detach().clone();
	// 	torch::Tensor qq = qtNN->parameters()[i].detach().clone();

	// 	std::cout<<" QuNN "<<i<<": "<< torch::sum(q).item<float>()
	// 		<<" QutNN "<<i<<": "<< torch::sum(qq).item<float>();
	// 	std::cout << '\n';
	// }
	
}
