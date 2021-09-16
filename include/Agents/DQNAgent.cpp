#include "DQNAgent.hpp"

const int hiddensize = 64;

DQNAgent::DQNAgent(size_t state_space, size_t action_space) :
	qNN(state_space, DQN_consts::actions_size, hiddensize),
	qtNN(state_space, DQN_consts::actions_size, hiddensize),
	qOptimizer(qNN.parameters(),torch::optim::AdamOptions(DQN_consts::a)),
	qtOptimizer(qtNN.parameters(),torch::optim::AdamOptions(DQN_consts::a))
	{
		//copy {Q'} <- {Q}
		for (size_t i = 0; i < qNN.parameters().size(); i++)
			qtNN.parameters()[i].set_data(qNN.parameters()[i].detach().clone());

		//assert that copy was successful
		for (size_t i = 0; i < qNN.parameters().size(); i++ )
			assert(torch::sum(qNN.parameters()[i] == qtNN.parameters()[i]).item<float>() == qNN.parameters()[i].numel());

	}

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
	std::vector<float> to_return(t1.data_ptr<float>(), t1.data_ptr<float>() + t1.numel());
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
	std::vector<float> to_return(t1.data_ptr<float>(), t1.data_ptr<float>() + t1.numel());
	return to_return;
}

void DQNAgent::reset_target_critic(){
	for (size_t i = 0; i < qNN.parameters().size(); i++ ){
		torch::Tensor t = qNN.parameters()[i].detach().clone();
		// torch::Tensor tt = qtNN.parameters()[i].detach().clone();
		qtNN.parameters()[i].set_data(t);
	}
}

void DQNAgent::trainCritic(const std::vector<experience_replayDQN>& samples,const int agentNumber){
	
	std::vector<float> states(DQN_consts::batch_size);
	std::vector<float> next_states(DQN_consts::batch_size);
	std::vector<float> rewards(DQN_consts::batch_size);
	std::vector<float> actions(DQN_consts::batch_size);

	for (int i = 0; i < DQN_consts::batch_size; ++i){
		states[i] = samples[i].current_state[agentNumber];
		next_states[i] = samples[i].next_state[agentNumber];
		rewards[i] = samples[i].reward[agentNumber];
		actions[i] = samples[i].action[agentNumber];
	}

	torch::Tensor s = torch::tensor(states);
	s = torch::reshape(s,{DQN_consts::batch_size,1});
	torch::Tensor n_s = torch::tensor(next_states);
	n_s = torch::reshape(n_s,{DQN_consts::batch_size,1});
	torch::Tensor a = torch::tensor(actions);
	torch::Tensor r = torch::tensor(rewards);

	torch::Tensor temp = qtNN.forward(n_s);
	// std::cout<< temp<<std::endl;
	torch::Tensor t1 = std::get<0>(torch::max(temp,1));
	// std::cout<< t1<<std::endl;

	torch::Tensor temp2 = qNN.forward(s);
	// std::cout<< temp2<<std::endl;

	// std::cout<< a<<std::endl;

	std::vector<int> index(DQN_consts::batch_size);
	for (int i = 0; i < DQN_consts::batch_size; ++i){

		for (int j = 0; j < DQN_consts::actions_size; ++j){
			if(DQN_consts::actions[j] == (int)actions[i]){
				index[i] = j;			
			}
		}
	}
	torch::Tensor in = torch::tensor(index);
	// std::cout<<in<<std::endl;
	torch::Tensor aa = torch::zeros(DQN_consts::batch_size);

	for (int z = 0; z < temp2.sizes()[0]; ++z)
	{
		aa[z] = temp2[z][index[z]];
	}
	
    // std::cout<<aa<<std::endl;
    // std::cout<<r - aa<<std::endl;
    // std::cout<<DQN_consts::gamma *t1<<std::endl;

	torch::Tensor loss = (r + DQN_consts::gamma *t1 - aa);
	// std::cout<< loss<<std::endl;
	loss = torch::mean(torch::pow(loss,2));
	// std::cout<< loss<<std::endl;

	qOptimizer.zero_grad();
	loss.backward();
	qOptimizer.step();
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
