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

#define N_EDGES whGraph->GetEdges().size()

enum class algo_type{ddpg};

enum class agent_def{
	centralized,
	link,
	intersection
};

struct epoch_results{
	uint totalDeliveries;
	uint totalMove;
	uint totalEnter;
	uint totalWait;
};

class Warehouse{
	public:
		Warehouse(YAML::Node);
		virtual ~Warehouse(void);

		virtual void InitialiseMATeam() = 0; // create agents for the graph
		void initialise_wh_agents();

		void OutputPerformance(std::string) __attribute__ ((deprecated));
		void OutputControlPolicies(std::string) __attribute__ ((deprecated));
		void OutputEpisodeReplay(std::string, std::string, std::string, std::string) __attribute__ ((deprecated));
		void DisableEpisodeReplayOutput()__attribute__ ((deprecated));

		void LoadPolicies(YAML::Node) __attribute__ ((deprecated));
		virtual epoch_results SimulateEpoch(bool verbose) = 0;

	protected:
		void replan_AGVs(const std::vector<float> final_cost);
		void transition_AGVs(bool verbose = false);
		void traverse_one_step(const std::vector<float> final_cost);
		void GetJointState(std::vector<size_t> &s);//__attribute__((deprecated))
		void print_warehouse_state();
		std::vector<float> get_edge_utilization() __attribute__ ((pure));
		std::vector<float> get_vertex_utilization() __attribute__ ((pure));

		size_t nSteps; //number of steps per simulation
		std::vector<float> baseCosts;
		std::vector<size_t> capacities;

		algo_type algo;
		struct iAgent{
			size_t vID;					// graph vertex ID associated with agent (edge ID if link agent)
			std::vector<size_t> eIDs; // graph edge IDs associated with incoming edges to agent vertex (vertex IDs if link agent)
			std::list<size_t> agvIDs; // agv IDs waiting to cross intersection
		};

		std::vector<iAgent *> whAgents; // manage agent vertex and edge lookups from graph
		Graph * whGraph; // vertex and edge definitions, access to change edge costs at each step
		std::vector<AGV *> whAGVs; // manage AGV A* search and movement through graph

		inline void InitialiseGraph(std::string, std::string, std::string, YAML::Node, bool verbose = false); // read in configuration files and construct Graph
		inline void InitialiseAGVs(YAML::Node, bool verbose = false); // create AGVs to move in graph
		void InitialiseNewEpoch(); // reset simulation for each episode/epoch

		void UpdateGraphCosts(std::vector<float>);

		bool incorporates_time; //True means that the domain incorpated time to it's state
		//as described in github.io/anthropomorphic/Papers/Chung2018multiagent.pdf

		agent_def agent_type;
};

#endif // WAREHOUSE_H_
