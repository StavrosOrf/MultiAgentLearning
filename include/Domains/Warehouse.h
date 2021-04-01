#ifndef WAREHOUSE_H_
#define WAREHOUSE_H_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <csignal>
#include <float.h>
#include <yaml-cpp/yaml.h>
#include "Planning/Graph.h"
#include "Planning/Edge.h"
#include "AGV.h"
#include "Agents/DDPGAgent.h"

#define N_EDGES whGraph->GetEdges().size()

using std::vector;
using std::list;
using std::string;
using std::ifstream;
using std::stringstream;

enum algo_type{
    ddpg
};

struct epoch_results{
    size_t totalDeliveries;
    size_t totalMove;
    size_t totalEnter;
    size_t totalWait;
};

class Warehouse{
	public:
		Warehouse(YAML::Node);
		virtual ~Warehouse(void);

		virtual void InitialiseMATeam(){ // create agents for the graph
			std::cout << "This function initialises the multiagent team.\n";
		}
		void EvolvePolicies(bool init = false) __attribute__ ((deprecated));

		void OutputPerformance(string) __attribute__ ((deprecated));
		void OutputControlPolicies(string) __attribute__ ((deprecated));
		void OutputEpisodeReplay(string, string, string, string) __attribute__ ((deprecated));
		void DisableEpisodeReplayOutput()__attribute__ ((deprecated)) {outputEpReplay = false;}

		void LoadPolicies(YAML::Node) __attribute__ ((deprecated));
		virtual epoch_results SimulateEpochDDPG(bool verbose){return {0,0,0,0};}

		bool get_incorporates_time(){return incorporates_time;}
	protected:
		void replan_AGVs(const std::vector<float> final_cost);
		void transition_AGVs(bool verbose = false);
		void traverse_one_step(const std::vector<float> final_cost);
		void GetJointState(vector<Edge *> e, vector<size_t> &s) ;//__attribute__((deprecated))
		void print_warehouse_state();
		vector<float> get_edge_utilization() __attribute__ ((pure));

		size_t nSteps; //number of steps per simulation
		size_t nAgents;
		size_t nAGVs;
		vector<float> baseCosts;
		vector<size_t> capacities;

		algo_type algo;
		struct iAgent{
			size_t vID;					// graph vertex ID associated with agent (edge ID if link agent)
			vector<size_t> eIDs; // graph edge IDs associated with incoming edges to agent vertex (vertex IDs if link agent)
			list<size_t> agvIDs; // agv IDs waiting to cross intersection
		};

		vector<DDPGAgent *> ddpg_maTeam;
		vector<iAgent *> whAgents; // manage agent vertex and edge lookups from graph
		Graph * whGraph; // vertex and edge definitions, access to change edge costs at each step
		vector<AGV *> whAGVs; // manage AGV A* search and movement through graph

		inline void InitialiseGraph(string, string, string, YAML::Node); // read in configuration files and construct Graph
		void InitialiseAGVs(YAML::Node); // create AGVs to move in graph
		void InitialiseNewEpoch(); // reset simulation for each episode/epoch

		vector< vector<size_t> > RandomiseTeams(size_t) __attribute__ ((deprecated)); // shuffle agent populations

		void UpdateGraphCosts(vector<float>);

		bool outputEvals;
		bool outputEpReplay;

		std::ofstream evalFile;
		std::ofstream agvStateFile;
		std::ofstream agvEdgeFile;
		std::ofstream agentStateFile;
		std::ofstream agentActionFile;

		bool incorporates_time; //True means that the domain incorpated time to it's state
		//as described in github.io/anthropomorphic/Papers/Chung2018multiagent.pdf
};

#endif // WAREHOUSE_H_
