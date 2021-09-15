#include "DQNAgent.hpp"

const int hiddensize = 64;

DQNAgent::DQNAgent(size_t state_space, size_t action_space){
	
	qNN = CriticNN(state_space, action_space, hiddensize);
	qtNN = CriticNN(state_space,action_space, hiddensize);
	
	// optimizerQNN(Q.parameters(), 0.001);
	// optimizerQtNN(Qt.parameters(), 0.001);
	

	//copy {Q', Mu'} <- {Q, Mu}
	for (size_t i = 0; i < qNN.parameters().size(); i++)
		qtNN.parameters()[i].set_data(qNN.parameters()[i].detach().clone());

	//assert that copy was successful
	for (size_t i = 0; i < qNN.parameters().size(); i++ )
		assert(torch::sum(qNN.parameters()[i] == qtNN.parameters()[i]).item<float>() == qNN.parameters()[i].numel());
}

DQNAgent::~DQNAgent() = default;

std::vector<float> DQNAgent::evaluate_critic_NN(const std::vector<float>& s,const std::vector<float>& a){
	std::vector<float> input;
	input.insert(input.begin(),s.begin(),s.end());
	// input.insert(input.end(),a.begin(),a.end());
	// std::cout<<s<<std::endl;
	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	// t = torch::reshape(t,{2, static_cast<int>(input.size()/2)});
	// t = torch::transpose(t,0,1);
	t = t.to(torch::kFloat32);
	// std::cout<<t<<std::endl;
	torch::Tensor t1 = qNN.forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}


std::vector<float> DQNAgent::evaluate_target_critic_NN(const std::vector<float>& s,const std::vector<float>& a){
	std::vector<float> input;
	// for (int i = 0; i < s.size(); ++i)
	// {
	// 	std::cout<<s[i]<<" "<<std::endl;
	// }
	// std::cout<<s<<std::endl;
	// std::cout<<a<<std::endl;

	input.insert(input.begin(),s.begin(),s.end());
	// input.insert(input.end(),a.begin(),a.end());


	// std::cout<<input<<std::endl;
	torch::Tensor t = torch::tensor(input).unsqueeze(0);
	t = torch::reshape(t,{2, static_cast<int>(input.size()/2)});
	t = torch::transpose(t,0,1);
	// std::cout<<t<<std::endl;
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = qtNN.forward(t);
	// std::cout<<t1<<std::endl;
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}

void DQNAgent::reset_target_critic(){
	for (size_t i = 0; i < qNN.parameters().size(); i++ ){
		torch::Tensor t = qNN.parameters()[i].detach().clone();
		// torch::Tensor tt = qtNN.parameters()[i].detach().clone();
		qtNN.parameters()[i].set_data(t);
	}
}


/************************************************************************************************
**Method:Print Various Metrics about the agent's neural Networks				*
*************************************************************************************************/
void DQNAgent::printAboutNN(){
	for (size_t i = 0; i < qNN.parameters().size(); i++ ){
		torch::Tensor t = qNN.parameters()[i].detach().clone();

		std::cout<<" qNN "<<i<<": "<< torch::sum(t).item<float>() << '\n';
	}
	// for (size_t i = 0; i < qNN->parameters().size(); i++ ){
	// 	torch::Tensor q = qNN->parameters()[i].detach().clone();
	// 	torch::Tensor qq = qtNN->parameters()[i].detach().clone();

	// 	std::cout<<" QuNN "<<i<<": "<< torch::sum(q).item<float>()
	// 		<<" QutNN "<<i<<": "<< torch::sum(qq).item<float>();
	// 	std::cout << '\n';
	// }
	
}
