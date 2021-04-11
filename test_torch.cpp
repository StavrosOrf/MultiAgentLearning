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


	{//test div
		assert(torch::ones(1).div(2).item<float>() == 0.5);
	}

	{//test minus
		assert((torch::ones(1)-1 == torch::zeros(1)).item<bool>());
	}

	{//assert that fowarding the same NN with the same inputs provides the same output
		const int input_nodes = 20, output_nodes=10, hidden_nodes = 2, hidden_count = 1;
		Net nn0 (input_nodes, output_nodes, hidden_nodes, hidden_count);
		const torch::Tensor input = torch::rand({1,input_nodes});
		assert(torch::sum(nn0.forward(input) == nn0.forward(input)).item<float>() == (float) output_nodes);
	}

	/*
	torch::Tensor fir = torch::rand({10, 5});
	for (int i = 0; i != 15; i++)
		std::cout << torch::mm(fir, torch::rand({5,1})) << std::endl;
	*/

	{//copy parameters
		//torch::Tensor t0 = torch::rand(1);
		torch::Tensor t0 = torch::zeros(1), t1 = t0.detach().clone();
		t1 = t0.detach().clone();//works
		//t1 = t0.clone();//does not work
		//t1 = t0.detach();//does not work
		//t1 = t0.clone().detach();//does not work
		//t1.data().copy(t0.data());//does not compile
		//t1.data() = t0.data();//does not compile
		//t1.data_ptr() = t0.data();//does not compile
		//t1.data_ptr() = *t0.data_ptr();//does not compile
		//t1 = t0 //works but i can't verify if it copies the data or the reference
		//t1 = t0.data;
		assert((t1 == t0).item<bool>());//Verify that copy was succefull

		const int input_nodes = 20, output_nodes=10, hidden_nodes = 2, hidden_count = 1;
		Net nn0 (input_nodes, output_nodes, hidden_nodes, hidden_count);
		Net nn1 (input_nodes, output_nodes, hidden_nodes, hidden_count);
		for (size_t i = 0; i < nn0.parameters().size(); i++ ){
			torch::Tensor t = nn0.parameters()[i].detach().clone();
			nn1.parameters()[i].set_data(t);
			//nn1.parameters()[i] = nn0.parameters()[i].detach().clone();
		}
		for (size_t i = 0; i < nn0.parameters().size(); i++ )//assert that all the parameters are equal
			assert(torch::sum(nn1.parameters()[i] == nn0.parameters()[i]).item<float>() == nn0.parameters()[i].numel());

		//std::cout << "nn0 first parameter:"<<nn0.parameters()[0];
		//std::cout << "nn0 first:" <<nn0.first;
		//std::cout << "nn1 first parameter:"<<nn1.parameters()[0];
		//std::cout << "nn1 first:" <<nn1.first;
		assert(torch::sum(nn1.parameters()[0] == nn1.first).item<float>() == nn1.parameters()[0].numel());

		const torch::Tensor input = torch::ones({1,input_nodes});
		//assert that the foward run of 2 identical NNs gives the same output
		assert(torch::sum(nn0.forward(input) == nn1.forward(input)).item<float>() == (float) output_nodes);
	}

	{// test
	}
}
