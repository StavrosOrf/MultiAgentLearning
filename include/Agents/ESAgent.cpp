#include "ESAgent.hpp"


ESAgent::ESAgent(size_t state_space, size_t action_space){
	//int hiddensize = 256;
	// TODO Parametrize NN to fit in cache
	NN = new esNN(state_space, action_space);

}

ESAgent::~ESAgent(){
	delete(NN);
	NN = NULL;
}

void ESAgent::updateNNWeights(float scalar){	

	for (size_t i = 0; i < NN->parameters().size(); i++ ){
		torch::Tensor t = NN->parameters()[i].detach().clone();
		NN->parameters()[i].set_data(t + scalar);
	}

}


void ESAgent::setNN(esNN* nn){
	for (size_t i = 0; i < nn->parameters().size(); i++ )
		NN->parameters()[i].set_data(nn->parameters()[i].detach().clone());
}

std::vector<float> ESAgent::evaluateNN(const std::vector<float>& s){
	//TODO try to use toDevice(GPU)
	torch::Tensor t = torch::tensor(std::move(s)).unsqueeze(0);
	t = t.to(torch::kFloat32);

	torch::Tensor t1 = NN->forward(t);
	std::vector<float> to_return(t1.data<float>(), t1.data<float>() + t1.numel());
	return to_return;
}