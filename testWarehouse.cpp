#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>
#include <ctime>
#include <filesystem>
#include <cassert>
#include <stdio.h>
#include <unistd.h>

#include "Domains/Warehouse.hpp"
#include "Domains/Warehouse_DDPG.hpp"
#include "Domains/Warehouse_ES_container.hpp"
#include "Domains/Warehouse_DQN.hpp"
#include "Domains/Warehouse_hardcoded.hpp"

Warehouse* create_warehouse(YAML::Node configs);
std::string create_results_folder(YAML::Node configs);
std::string config_file;

void warehouse_simulate_hardcoded(YAML::Node configs, [[maybe_unused]] size_t n_threads) {
	[[maybe_unused]] const bool verbose = configs["simulation"]["verbose"].as<bool>();

	Warehouse_hardcoded trainDomain = Warehouse_hardcoded(configs);
	epoch_results t = trainDomain.simulate_epoch(verbose);//simulate
}


void warehouse_simulate_DQN(YAML::Node configs, [[maybe_unused]] size_t n_threads){
	const int nEps = configs["DQN"]["epochs"].as<int>();
	const int runs = configs["DQN"]["runs"].as<int>();
	[[maybe_unused]] const bool verbose = configs["simulation"]["verbose"].as<bool>();
	const std::string warehouse_type = configs["domain"]["warehouse"].as<std::string>();
	const std::string agentType = configs["domain"]["agents"].as<std::string>();
	const std::string resFolder = create_results_folder(configs);

	std::clock_t start, startTotalRun, startTotalExperiment = std::clock();

	double duration;
	uint max_G = 0; //Maximum Total Deliveries

	std::ofstream eval_file(resFolder + warehouse_type + '_' + "DQN" + '_' + agentType + ".csv");
	for (int run = 0; run != runs; run++){
		assert(eval_file.is_open());
		eval_file << "run,Epoch,G,tMove,tEnter,tWait\n";
		startTotalRun = std::clock();
		Warehouse_DQN trainDomain = Warehouse_DQN(configs);
		trainDomain.InitialiseMATeam();

		std::cout << "Starting Run: " << run << std::endl;
		for (int e = 0; e < nEps; e++){
			start = std::clock();
			epoch_results t = trainDomain.simulate_epoch_DQN(verbose);//simulate
			duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
			if (t.totalDeliveries > max_G){
				std::printf("Epoch %3d (%5.1f sec): \e[1mG=%4u\e[0m, tMove=%6u, tEnter=%6u, tWait=%6u\n",
					e,duration,t.totalDeliveries,t.totalMove,t.totalEnter,t.totalWait);
				max_G = t.totalDeliveries;
			}
			else
				std::printf("Epoch %3d (%5.1f sec): G=%4u, tMove=%6u, tEnter=%6u, tWait=%6u\n",
					e,duration,t.totalDeliveries,t.totalMove,t.totalEnter,t.totalWait);
			eval_file<<run<<','<<e<<','<<t.totalDeliveries<<','<<t.totalMove<<','<<t.totalEnter<<','<<t.totalWait << std::endl;
		}
		duration = ( std::clock() - startTotalRun ) / ((double) CLOCKS_PER_SEC );
		std::cout<<"Total time elapsed for Run "<<run<<" ( "<<duration<<" sec)"<<std::endl; 
		
	}
	eval_file.close();
	duration = ( std::clock() - startTotalExperiment ) / ((double) CLOCKS_PER_SEC );
	std::cout<<"Total time elapsed for Experiment:( "<<duration<<" sec)"<<std::endl;
}

void warehouse_simulate_ES(YAML::Node configs, [[maybe_unused]] size_t n_threads){
	const size_t runs = configs["ES"]["runs"].as<size_t>();
	[[maybe_unused]] const bool verbose = configs["simulation"]["verbose"].as<bool>();
	const std::string warehouse_type = configs["domain"]["warehouse"].as<std::string>();
	const std::string agentType = configs["domain"]["agents"].as<std::string>();
	const std::string resFolder = create_results_folder(configs);
	
	std::ofstream eval_file(resFolder + warehouse_type + '_' + "ES" + '_' + agentType + ".csv");
	for (size_t i = 0; i != runs; i++){
		assert(eval_file.is_open());

		eval_file <<",Epoch,MAX_G,MAX_MOVE,MAX_ENTER,MAX_WAIT,AVG_G"<< std::endl;
		Warehouse_ES_container esc(configs);
		[[maybe_unused]] uint G = esc.evolution_strategy(n_threads, verbose, i, eval_file);
	}
}

void warehouse_simulate_ADAM_ES(YAML::Node configs, [[maybe_unused]] size_t n_threads){
	const size_t runs = configs["ES"]["runs"].as<size_t>();
	[[maybe_unused]] const bool verbose = configs["simulation"]["verbose"].as<bool>();
	const std::string warehouse_type = configs["domain"]["warehouse"].as<std::string>();
	const std::string agentType = configs["domain"]["agents"].as<std::string>();
	const std::string resFolder = create_results_folder(configs);
	
	std::ofstream eval_file(resFolder + warehouse_type + '_' + "ADAM_ES" + '_' + agentType + ".csv");
	for (size_t i = 0; i != runs; i++){
		assert(eval_file.is_open());

		eval_file <<",Epoch,MAX_G,MAX_MOVE,MAX_ENTER,MAX_WAIT,AVG_G"<< std::endl;
		Warehouse_ES_container esc(configs);
		[[maybe_unused]] uint G = esc.evolution_strategy_ADAM(n_threads, verbose, i, eval_file);
	}
}

void WarehouseSimulationDDPG(YAML::Node configs){
	#ifndef ENABLE_DDPG
		assert(0);
	#else
	srand(time(NULL)); // increment random seed
	const int nEps = configs["DDPG"]["epochs"].as<int>();
	const int runs = configs["DDPG"]["runs"].as<int>();
	const std::string warehouse_type = configs["domain"]["warehouse"].as<std::string>();
	const std::string agentType = configs["domain"]["agents"].as<std::string>();
	const bool verbose = configs["simulation"]["verbose"].as<bool>();
	const std::string resFolder = create_results_folder(configs);

	std::clock_t start, startTotalRun, startTotalExperiment = std::clock();

	double duration;
	uint max_G = 0; //Maximum Total Deliveries

	std::ofstream eval_file(resFolder + warehouse_type + '_' + "DDPG" + '_' + agentType + ".csv");
	for (int run = 0; run != runs; run++){
		assert(eval_file.is_open());
		eval_file << "run,Epoch,G,tMove,tEnter,tWait\n";
		startTotalRun = std::clock();
		Warehouse_DDPG trainDomain = Warehouse_DDPG(configs);
		trainDomain.InitialiseMATeam();

		std::cout << "Starting Run: " << run << std::endl;
		for (int n = 0; n < nEps; n++){
			start = std::clock();
			epoch_results t = trainDomain.simulate_epoch_DDPG(verbose,n);//simulate
			duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
			if (t.totalDeliveries > max_G){
				std::printf("Epoch %3d (%5.1f sec): \e[1mG=%4u\e[0m, tMove=%6u, tEnter=%6u, tWait=%6u\n",
					n,duration,t.totalDeliveries,t.totalMove,t.totalEnter,t.totalWait);
				max_G = t.totalDeliveries;
			}
			else
				std::printf("Epoch %3d (%5.1f sec): G=%4u, tMove=%6u, tEnter=%6u, tWait=%6u\n",
					n,duration,t.totalDeliveries,t.totalMove,t.totalEnter,t.totalWait);
			eval_file<<run<<','<<n<<','<<t.totalDeliveries<<','<<t.totalMove<<','<<t.totalEnter<<','<<t.totalWait << std::endl;
		}
		duration = ( std::clock() - startTotalRun ) / ((double) CLOCKS_PER_SEC );
		std::cout<<"Total time elapsed for Run "<<run<<" ( "<<duration<<" sec)"<<std::endl; 
		eval_file.close();
	}
	duration = ( std::clock() - startTotalExperiment ) / ((double) CLOCKS_PER_SEC );
	std::cout<<"Total time elapsed for Experiment:( "<<duration<<" sec)"<<std::endl;
	#endif
}

void WarehouseSimulation(const std::string &config_file, size_t n_threads){
	std::cout << "Reading configuration file: " << config_file << "\n";

	YAML::Node configs = YAML::LoadFile(config_file);

	std::string algo = configs["mode"]["algo"].as<std::string>();
	//std::string mode = configs["mode"]["type"].as<std::string>();

	if (algo == "DDPG") {
		std::cout << "(MA)DDPG has been deprecated)" << std::endl;
		WarehouseSimulationDDPG(configs);
	}else if (algo == "ES"){
		warehouse_simulate_ES(configs, n_threads);
	}else if (algo == "ADAM_ES"){
		warehouse_simulate_ADAM_ES(configs, n_threads);
	}else if (algo == "DQN"){
		warehouse_simulate_DQN(configs, n_threads);
	}else if (algo == "HARDCODED") {
		warehouse_simulate_hardcoded(configs, n_threads);
	}
	else{
		std::cout << "Error: unknown algo! Exiting.\n";
		exit(EXIT_FAILURE);
	}
}

/************************************************************************************************
**Input:Creates a Result_folder as defined by the config					*
**Output:Returns the string of the file result's folder file path				*
*************************************************************************************************/
std::string create_results_folder(YAML::Node configs){
	const int64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::string resFolder = configs["results"]["folder"].as<std::string>()
		+ std::to_string(timestamp) + '/';
	char mkdir[10000];
	sprintf(mkdir,"mkdir -p %s",(resFolder).c_str());
	system(mkdir);
	std::filesystem::copy_file(config_file, resFolder+config_file.substr(3));
	return resFolder;
}

static void show_usage(const std::string &name){
	std::cerr << "Usage: " << name << " -c CONFIG_FILE <options>\n"
		<< "options:\n"
		<< "\t-h, --help\t\t\tShow this help message\n"
		<< "\t-c, --config CONFIG_FILE_DIR\tSpecify the configuration file path\n"
		<< "\t-t, --threads N_THREADS\t\tSpecify the number of parallel threads (default: 2)"
		<< std::endl;
}

int main(int argc, char* argv[]){
	if (argc < 3){
		show_usage(argv[0]);
		return 0;
	}

	int thrds = 1; // Default number of threads
	for (int i = 1; i < argc; ++i){
		std::string arg = argv[i];
		//std::cout << "main"<<std::endl;
		// Display help
		if ((arg == "-h") || (arg == "--help")){
			show_usage(argv[0]);
			return 0;
		}
		// Path to configuration file
		else if ((arg == "-c") || (arg == "--config")){
			if (i + 1 < argc)
				config_file = argv[++i];
			else {
				std::cerr << "--config option requires one argument.\n";
				return 1;
			}
		}
		// Number of parallel threads

		else if ((arg == "-t") || (arg == "--threads")){
			if (i + 1 < argc) {
				thrds = atoi(argv[++i]);
				std::cout << "Using " << thrds << " threads.\n";
			}
			else
				std::cout << "Using default 2 threads for parallel compute.\n";
		}
		// Unknown input
		else {
			std::cerr << "Unknown option " << arg << ". Exiting.\n";
			return 1;
		}
	}

	//torch::Device device = torch::kCPU;
	if(torch::cuda::is_available()) {
		std::cout<<"Cuda is available!"<<"\n";
		//device = torch::kCUDA;
	}else
		std::cout<<"No Cuda is available!"<<"\n";

	WarehouseSimulation(config_file, thrds);

	return 0;
}
