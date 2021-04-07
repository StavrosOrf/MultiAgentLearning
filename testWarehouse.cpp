#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>
#include <time.h>
#include <ctime>
#include <cassert>
#include <stdio.h>
#include <unistd.h>

#include "Domains/Warehouse.h"
#include "Domains/Warehouse_DDPG.h"
#include "Domains/Warehouse_DDPG_merged_step.h"

Warehouse* create_warehouse(YAML::Node configs);
std::string create_results_folder(YAML::Node configs);

void WarehouseSimulationDDPG(YAML::Node configs){
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

	for (int run = 0; run != runs; run++){
		std::ofstream eval_file(resFolder + warehouse_type + '_' + agentType + '_' + std::to_string(run) + ".csv");
		assert(eval_file.is_open());
		eval_file << "run,Epoch,G,tMove,tEnter,tWait\n";
		startTotalRun = std::clock();
		Warehouse * trainDomain = create_warehouse(configs);

		std::cout << "Starting Run: " << run << std::endl;
		for (int n = 0; n < nEps; n++){
			start = std::clock();
			epoch_results t = trainDomain->SimulateEpoch(verbose,n);//simulate
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
}

void WarehouseSimulation(std::string config_file){
	std::cout << "Reading configuration file: " << config_file << "\n";

	YAML::Node configs = YAML::LoadFile(config_file);

	std::string algo = configs["mode"]["algo"].as<std::string>();
	std::string mode = configs["mode"]["type"].as<std::string>();

	if (algo == "DDPG") {
		if(mode == "train")
			WarehouseSimulationDDPG(configs);
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
Warehouse* create_warehouse(YAML::Node configs){
	Warehouse* new_warehouse;
	if(configs["mode"]["algo"].as<std::string>() == "DDPG")
		new_warehouse = new Warehouse_DDPG(configs);
	else if(configs["mode"]["algo"].as<std::string>() == "DDPG_MERGED")
		new_warehouse = new Warehouse_DDPG_merged_step(configs);
	else{
		std::cout << "error: currently only configured for 'ddpg' and ''! exiting.\n";
		exit(1);
	}

	new_warehouse->InitialiseMATeam();

	return new_warehouse;
}

/************************************************************************************************
**Input:Creates a Result_folder as defined by the config					*
**Output:Returns the string of the file result's folder file path				*
*************************************************************************************************/
std::string create_results_folder(YAML::Node configs){
	std::string resFolder = configs["results"]["folder"].as<std::string>();
	char mkdir[100];
	sprintf(mkdir,"mkdir -p %s",(resFolder).c_str());
	system(mkdir);
	return resFolder;
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

	std::string config_file;
	int thrds = 2; // Default number of threads
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

	WarehouseSimulation(config_file);

	return 0;
}
