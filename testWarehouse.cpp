#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>
#include <ctime>
#include <stdio.h>

#include "Domains/Warehouse.h"
#include "Domains/WarehouseCentralised.h"
//#include "threadpool.hpp"

char mkdir[100];
string domainDir;
string resFolder;

Warehouse* create_warehouse(std::string agentType, YAML::Node configs);
void create_results_folder(Warehouse* trainDomain, YAML::Node configs, size_t r);

void WarehouseSimulationDDPG(int r, YAML::Node configs){
	srand(time(NULL)); // increment random seed
	// Initialise appropriate domain
	size_t nEps = configs["DDPG"]["epochs"].as<size_t>();
	string agentType = configs["domain"]["agents"].as<string>();
	int runs = configs["DDPG"]["runs"].as<int>();
	
	std::clock_t start;
	std::clock_t startTotalRun;
	std::clock_t startTotalExperiment = std::clock();

  double duration;

	for (int i = 0; i != runs; i ++){
		startTotalRun = std::clock();
		Warehouse * trainDomain = create_warehouse(agentType, configs);

		create_results_folder(trainDomain, configs, r);

		std::cout << "Starting Run: " << i << std::endl;
		for (size_t n = 0; n < nEps; n++){
			start = std::clock();
			epoch_results t = trainDomain->SimulateEpochDDPG(false);// simulate
   		duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
   		// std::cout<<"Epoch "<< n <<" ("<<duration<<" sec): ";
   		std::printf("Epoch %3d (%5.1f sec): G=%4lu, tMove=%6lu, tEnter=%6lu, tWait=%6lu\n",
   				(int)n,duration,t.totalDeliveries,t.totalMove,t.totalEnter,t.totalWait);   		
		}
		duration = ( std::clock() - startTotalRun ) / ((double) CLOCKS_PER_SEC );
		std::cout<<"Total time elapsed for Run "<<i<<" ( "<<duration<<" sec)"<<std::endl; 
	}
	duration = ( std::clock() - startTotalExperiment ) / ((double) CLOCKS_PER_SEC );
		std::cout<<"Total time elapsed for Experiment:( "<<duration<<" sec)"<<std::endl;

	exit(0);
}

void WarehouseSimulation(string config_file, int thrds){
	std::cout << "Reading configuration file: " << config_file << "\n";

	YAML::Node configs = YAML::LoadFile(config_file);

	string algo = configs["mode"]["algo"].as<string>();
	string mode = configs["mode"]["type"].as<string>();
	//ThreadPool pool(thrds);

	if (algo == "DDPG") {
		if(mode == "train"){
			int r = 0;
			WarehouseSimulationDDPG(r, configs);
		}
	}
	else{
		std::cout << "Error: unknown algo! Exiting.\n";
		exit(1);
	}

}

/************************************************************************************************
 * *Input: A string named [agentType] which indicates the type of the Warehouse			*
 * *Output:A Warehouse of that type								*
 ************************************************************************************************/
Warehouse* create_warehouse(std::string agentType, YAML::Node configs){
	Warehouse* new_warehouse;
	if (agentType.compare("centralised") == 0)
		new_warehouse = new WarehouseCentralised(configs);
	else{
		std::cout << "ERROR: Currently only configured for 'intersection', 'link' or 'centralised' agents! Exiting.\n";
		exit(1);
	}

	new_warehouse->InitialiseMATeam();
	return new_warehouse;
}

void create_results_folder(Warehouse* trainDomain, YAML::Node configs, size_t r){
	domainDir = configs["domain"]["folder"].as<string>();
	resFolder = configs["results"]["folder"].as<string>();
	std::stringstream ss_eval;
	ss_eval << domainDir << resFolder << configs["results"]["evaluation"].as<string>() << "_" << r << ".csv";
	string eval_str = ss_eval.str();
	sprintf(mkdir,"mkdir -p %s",(domainDir + resFolder).c_str());
	system(mkdir);
	trainDomain->OutputPerformance(eval_str);
}

static void show_usage(std::string name){
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

	string config_file;
	int thrds = 2; // Default number of threads
	for (int i = 1; i < argc; ++i){
		string arg = argv[i];
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
				//config_file = "../config.yaml";i++;
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

	WarehouseSimulation(config_file, thrds);

	return 0;
}
