#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>


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

}

