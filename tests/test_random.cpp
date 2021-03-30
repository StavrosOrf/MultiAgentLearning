#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>
#include <torch/torch.h>


//Various Tests here
int main(int argc, char* argv[]){
	std::normal_distribution<double> n_process(0.0, 0.1);
	std::default_random_engine n_generator;

	for (int i = 0; i != 45; i++)
		std::cout << n_process(n_generator) << std::endl;
}

