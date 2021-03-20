#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <Eigen/Eigen>
#include <yaml-cpp/yaml.h>

#include "Domains/Warehouse.h"
#include "Domains/WarehouseIntersectionsTime.h"
#include "Domains/WarehouseIntersections.h"
#include "Domains/WarehouseLinksTime.h"
#include "Domains/WarehouseLinks.h"
#include "Domains/WarehouseCentralisedTime.h"
#include "Domains/WarehouseCentralised.h"
#include "threadpool.hpp"

using std::vector;
using std::string;
using namespace Eigen;

Warehouse* create_warehouse(std::string agentType, YAML::Node configs);

void WarehouseSimulationSingleRun(int r, YAML::Node configs){
	srand(r+1); // increment random seed

	// Initialise appropriate domain
	size_t nEps = configs["neuroevo"]["epochs"].as<size_t>();
	string agentType = configs["domain"]["agents"].as<string>();
	Warehouse * trainDomain = create_warehouse(agentType, configs);
	trainDomain->InitialiseMATeam() ;
	
	// Create results folder
	int runs = configs["neuroevo"]["runs"].as<int>();
	string domainDir = configs["domain"]["folder"].as<string>() ;
	string resFolder = configs["results"]["folder"].as<string>() ;
	std::stringstream ss_eval ;
	ss_eval << domainDir << resFolder << configs["results"]["evaluation"].as<string>() << "_" << r << ".csv" ;
	string eval_str = ss_eval.str() ;
	char mkdir[100] ;
	sprintf(mkdir,"mkdir -p %s",(domainDir + resFolder).c_str()) ;
	system(mkdir) ;
	trainDomain->OutputPerformance(eval_str) ;
	
	// Execute learning episodes of current stat run
	for (size_t n = 0; n < nEps; n++){
		if (r == runs-1){ // store the first and last episodes of the final stat run for replay
			if (n == 0 || n == nEps-1){
				sprintf(mkdir,"mkdir -p %s",(domainDir + resFolder + "Replay/").c_str()) ;
				system(mkdir) ;
				std::stringstream ss_agv_s ;
				std::stringstream ss_agv_e ;
				ss_agv_s << domainDir << resFolder << "Replay/AGV_states_" << n << ".csv" ;
				ss_agv_e << domainDir << resFolder << "Replay/AGV_edges_" << n << ".csv" ;
				std::stringstream ss_a_s ;
				std::stringstream ss_a_a ;
				ss_a_s << domainDir << resFolder << "Replay/agent_states_" << n << ".csv" ;
				ss_a_a << domainDir << resFolder << "Replay/agent_actions_" << n << ".csv" ;
				trainDomain->OutputEpisodeReplay(ss_agv_s.str(), ss_agv_e.str(), ss_a_s.str(), ss_a_a.str()) ;
			}
			else{ // do not record domain states if not the final stat run
				trainDomain->DisableEpisodeReplayOutput() ;
			}
		}
		
		// Main evolution routine for each episode
		std::cout << "Epoch " << n << "...\n" ;
		trainDomain->EvolvePolicies(n==0) ;		 // compete (except on first episode), then mutate
		trainDomain->ResetEpochEvals() ;				// reset domain
		trainDomain->SimulateEpoch() ;					// simulate
	}

	// Record learned policies of final stat run
	if (r == runs-1){
		string nn_str = domainDir + resFolder + configs["results"]["policies"].as<string>() ;
		std::cout << "Writing control policies to file: " << nn_str << "..." ;
		trainDomain->OutputControlPolicies(nn_str) ;
		std::cout << "complete.\n" ;
	}
	
	delete trainDomain ;
	trainDomain = 0 ;
	
	std::cout << "Training complete!\n" ;
}

void WarehouseSimulationTestSingleRun(int r, YAML::Node configs){
	srand(r+1); // increment random seed

	// Initialise appropriate domain
	string agentType = configs["domain"]["agents"].as<string>();
	Warehouse * testDomain = create_warehouse(agentType, configs); ;
	testDomain->InitialiseMATeam() ;
	
	// Create results folder
	int runs = configs["neuroevo"]["runs"].as<int>();
	string domainDir = configs["domain"]["folder"].as<string>() ;
	string resFolder = configs["results"]["folder"].as<string>() ;
	std::stringstream ss_eval ;
	ss_eval << domainDir << resFolder << configs["results"]["evaluation"].as<string>() << "_" << r << ".csv" ;
	string eval_str = ss_eval.str() ;
	char mkdir[100] ;
	sprintf(mkdir,"mkdir -p %s",(domainDir + resFolder).c_str()) ;
	system(mkdir) ;
	testDomain->OutputPerformance(eval_str) ;
	
	// Store the final stat run for replay
	if (r == runs-1){
		sprintf(mkdir,"mkdir -p %s",(domainDir + resFolder + "Replay/").c_str()) ;
		system(mkdir) ;
		std::stringstream ss_agv_s ;
		std::stringstream ss_agv_e ;
		ss_agv_s << domainDir << resFolder << "Replay/AGV_states.csv" ;
		ss_agv_e << domainDir << resFolder << "Replay/AGV_edges.csv" ;
		std::stringstream ss_a_s ;
		std::stringstream ss_a_a ;
		ss_a_s << domainDir << resFolder << "Replay/agent_states.csv" ;
		ss_a_a << domainDir << resFolder << "Replay/agent_actions.csv" ;
		testDomain->OutputEpisodeReplay(ss_agv_s.str(), ss_agv_e.str(), ss_a_s.str(), ss_a_a.str()) ;
	}
	
	// Extract the champion team for execution
	cout << "Reading champion team from file: " ;
	string ev_str = configs["mode"]["eval_file"].as<string>() ;
	ifstream evalFile(ev_str.c_str()) ;
	cout << ev_str.c_str() << "..." ;
	if (!evalFile.is_open()){
		cout << "\nFile: " << ev_str.c_str() << " not found, exiting.\n" ;
		exit(1) ;
	}
	vector< vector<size_t> > evals ;
	std::string line ;
	while (getline(evalFile,line))
	{
		stringstream lineStream(line) ;
		string cell ;
		vector<size_t> ev ;
		while (getline(lineStream,cell,','))
		{
			ev.push_back((size_t)atoi(cell.c_str())) ;
		}
		evals.push_back(ev) ;
	}
	vector<size_t> team ;
	for (size_t i = 7; i < evals[evals.size()-1].size(); i++){
		team.push_back(evals[evals.size()-1][i]) ;
	}
	cout << "complete.\n" ;
	
	// Simulate
	testDomain->LoadPolicies(configs) ;
	testDomain->ResetEpochEvals() ;
	testDomain->SimulateEpoch(team) ;

	// Store the control policies in the same test results folder
	if (r == runs-1){
		string nn_str = domainDir + resFolder + configs["results"]["policies"].as<string>() ;
		std::cout << "Writing control policies to file: " << nn_str << "..." ;
		testDomain->OutputControlPolicies(nn_str) ;
		std::cout << "complete.\n" ;
	}
	
	delete testDomain ;
	testDomain = 0 ;
	
	std::cout << "Testing complete!\n" ;
}

void WarehouseSimulationDDPG(YAML::Node configs){

	// Initialise appropriate domain
	string agentType = configs["domain"]["agents"].as<string>();
	Warehouse * trainDomain = create_warehouse(agentType, configs);
	if (agentType == "centralised_t" || agentType == "centralised"){
	}
	exit(0);


}

void WarehouseSimulation(string config_file, int thrds){
	std::cout << "Reading configuration file: " << config_file << "\n" ;
	
	YAML::Node configs = YAML::LoadFile(config_file);
	
	string algo = configs["mode"]["algo"].as<string>();
	string mode = configs["mode"]["type"].as<string>() ;
	int runs = configs["neuroevo"]["runs"].as<int>();
	ThreadPool pool(thrds) ;
	
	if (algo == "DDPG") {
		if(mode.compare("train") == 1){
			WarehouseSimulationIterDDPG(configs);
		}
		exit(0);
	}else if (algo == "nueroevo" || true){
		if (mode.compare("train") == 0){
			// Start the training runs
			for (int r = 0; r < runs; r++)
			{
				pool.schedule(std::bind(WarehouseSimulationSingleRun, r, configs));
			}
		}
		else if (mode.compare("test") == 0){
			// Start the testing runs
			for (int r = 0; r < runs; r++)
			{
				pool.schedule(std::bind(WarehouseSimulationTestSingleRun, r, configs));
			}
		}
		else{
			std::cout << "Error: unknown mode! Exiting.\n" ;
			exit(1) ;
		}
	}
	else{
		std::cout << "Error: unknown algo! Exiting.\n" ;
		exit(1) ;
	}

}

/************************************************************************************************
 * *Input: A string named [agentType] which indicates the type of the Warehouse			*
 * *Output:A Warehouse of that type								*
 ************************************************************************************************/
Warehouse* create_warehouse(std::string agentType, YAML::Node configs){
	if (agentType.compare("intersection_t") == 0)
		return new WarehouseIntersectionsTime(configs);
	else if (agentType.compare("intersection") == 0)
		return new WarehouseIntersections(configs);
	else if (agentType.compare("link_t") == 0)
		return new WarehouseLinksTime(configs);
	else if (agentType.compare("link") == 0)
		return new WarehouseLinks(configs);
	else if (agentType.compare("centralised_t") == 0)
		return new WarehouseCentralisedTime(configs);
	else if (agentType.compare("centralised") == 0)
		return new WarehouseCentralised(configs);
	else{
		std::cout << "ERROR: Currently only configured for 'intersection', 'link' or 'centralised' agents! Exiting.\n" ;
		exit(1) ;
	}
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
		show_usage(argv[0]) ;
		return 0 ;
	}
	
	string config_file ;
	int thrds = 2 ; // Default number of threads
	for (int i = 1; i < argc; ++i){
		string arg = argv[i] ;
		std::cout << "TEST";
		// Display help
		if ((arg == "-h") || (arg == "--help")){
			show_usage(argv[0]) ;
			return 0 ;
		}
		// Path to configuration file
		else if ((arg == "-c") || (arg == "--config")){
			if (i + 1 < argc) {
				config_file = argv[++i] ;
			}
			else {
				std::cerr << "--config option requires one argument.\n" ;
				return 1 ;
			}
		}
		// Number of parallel threads
		
		else if ((arg == "-t") || (arg == "--threads")){
			if (i + 1 < argc) {
				thrds = atoi(argv[++i]) ;
				std::cout << "Using " << thrds << " threads.\n" ;
			}
			else {
				std::cout << "Using default 2 threads for parallel compute.\n" ;
			}
		}
		// Unknown input
		else {
			std::cerr << "Unknown option " << arg << ". Exiting.\n" ;
			return 1 ;
		}
	}
	
	WarehouseSimulation(config_file, thrds) ;
	
	return 0 ;
}
