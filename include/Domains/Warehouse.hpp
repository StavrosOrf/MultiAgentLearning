#pragma once

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
#include "AGV.hpp"
#include <torch/torch.h>

#define N_EDGES whGraph->GetEdges().size()


enum class algo_type{ddpg, evolutionary_strategies, coma ,dqn};

enum class agent_def{centralized, link, intersection};

struct epoch_results{
	uint totalDeliveries = 0;
	uint totalMove = 0;
	uint totalEnter = 0;
	uint totalWait = 0;
	void update(uint tD, uint tM, uint tE, uint tW){
		totalDeliveries += tD;
		totalMove += tM;
		totalEnter += tE;
		totalWait += tW;
	}
};

struct epoch_resultsES{
	uint totalDeliveries = 0;
	uint totalMove = 0;
	uint totalEnter = 0;
	uint totalWait = 0;
	std::vector<std::vector<torch::Tensor>> samples; 
	void update(uint tD, uint tM, uint tE, uint tW, const std::vector<std::vector<torch::Tensor>> &s){
		totalDeliveries += tD;
		totalMove += tM;
		totalEnter += tE;
		totalWait += tW;
		samples = s;
	}
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
		void printAgvPaths();

	protected:
		void replan_AGVs(const std::vector<float> &final_cost);
		void transition_AGVs(bool verbose = false);
		void traverse_one_step(const std::vector<float> &final_cost);
		void GetJointState(std::vector<size_t> &s);//__attribute__((deprecated))
		void print_warehouse_state();
		std::vector<float> get_edge_utilization() __attribute__ ((pure));
		std::vector<float> get_edge_utilization(bool care_about_last_time, bool normalize=false, bool care_about_avg_time=false) __attribute__ ((pure));
		std::vector<vertex_t> get_vertex_utilization(const bool include_start_vertexes=true) __attribute__ ((pure));
		std::vector<vertex_t> get_vertex_utilization(const vertex_t goal_vertex, const bool include_start_vertexes=true) __attribute__ ((pure));
		float get_vertex_remaining_outgoing_capacity(vertex_t vertex) __attribute__ ((pure));
		float max_base_travel_cost() const __attribute__ ((const)) {return *std::max_element(baseCosts.begin(), baseCosts.end());}
		float max_edge_capacity() const __attribute__ ((const)) {return *std::max_element(capacities.begin(), capacities.end());}
		float value_of_current_state();

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

		inline void InitialiseGraph(const std::string&, const std::string&, const std::string&, const YAML::Node&, bool verbose = false); // read in configuration files and construct Graph
		inline void InitialiseAGVs(YAML::Node, bool verbose = false); // create AGVs to move in graph
		void InitialiseNewEpoch(); // reset simulation for each episode/epoch

		void UpdateGraphCosts(const std::vector<float> &);

		bool incorporates_time; //True means that the domain incorpated time to it's state
		//as described in github.io/anthropomorphic/Papers/Chung2018multiagent.pdf
		bool incorporates_avg_time = false;
		agent_def agent_type;
};