#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>

#include "Agents/DDPGAgent.h"



//Various Tests here
int main(int argc, char* argv[]){
	//at::Tensor to std::vector
	at::Tensor t = torch::rand({5, 1});
	std::vector<float> v(t.data<float>(), t.data<float>() + t.numel());
	for (int i = 0; i != t.numel(); i++)
		assert(t[i][0].item<float>() == v[i]);
	t = torch::rand({1, 5});
	std::vector<float> v0(t.data<float>(), t.data<float>() + t.numel());
	for (int i = 0; i != t.numel(); i++)
		assert(t[0][i].item<float>() == v0[i]);

	const int input_nodes = 20, output_nodes=1, hidden_count = 1;
	Net nn (input_nodes, output_nodes, hidden_count);

	std::cout << "Number of hidden layers: " << hidden_count << std::endl;
	for (int i = 0; i != 15; i++)
		//std::cout << nn.forward(torch::rand({output_nodes,input_nodes})).item<float>() << std::endl;
		std::cout << nn.forward(torch::zeros({output_nodes,input_nodes})).item<float>() << std::endl;

	/*
	torch::Tensor fir = torch::rand({10, 5});
	for (int i = 0; i != 15; i++)
		std::cout << torch::mm(fir, torch::rand({5,1})) << std::endl;
	*/

}
