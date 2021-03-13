#ifndef WAREHOUSE_H_
#define WAREHOUSE_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <chrono>
#include <random>
#include <algorithm>
#include <float.h>
#include <Eigen/Eigen>
#include <yaml-cpp/yaml.h>
#include "Agents/Agent.h"
#include "Agents/Intersection.h"
#include "Agents/Link.h"
#include "Planning/Graph.h"
#include "Planning/Edge.h"
#include "AGV.h"

using std::vector ;
using std::list ;
using std::string ;
using std::ifstream ;
using std::stringstream ;

class Warehouse{
  public:
    Warehouse(YAML::Node) ;
    virtual ~Warehouse(void) ;
    
    virtual void SimulateEpoch(bool train = true){
      std::cout << "This function simulates a single learning epoch.\n" ;
    }
    virtual void SimulateEpoch(vector<size_t> team){
      std::cout << "This function simulates a single epoch with a given multiagent team.\n" ;
    }
    virtual void InitialiseMATeam(){ // create agents for the graph
      std::cout << "This function initialises the multiagent team.\n" ;
    }
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals() ;
    
    void OutputPerformance(string) ;
    void OutputControlPolicies(string) ;
    void OutputEpisodeReplay(string, string, string, string) ;
    void DisableEpisodeReplayOutput(){outputEpReplay = false ;}

    void LoadPolicies(YAML::Node) ;
    
  protected:
    size_t nSteps ;
    size_t nPop ;
    size_t nAgents ;
    size_t nAGVs ;
    vector<double> baseCosts ;
    vector<size_t> capacities ;
    bool neLearn ;
    
    struct iAgent{
      size_t vID ;          // graph vertex ID associated with agent (edge ID if link agent)
      vector<size_t> eIDs ; // graph edge IDs associated with incoming edges to agent vertex (vertex IDs if link agent)
      list<size_t> agvIDs ; // agv IDs waiting to cross intersection
    } ;
    
    vector<Agent *> maTeam ; // manage agent NE routines
    vector<iAgent *> whAgents ; // manage agent vertex and edge lookups from graph
    Graph * whGraph ; // vertex and edge definitions, access to change edge costs at each step
    vector<AGV *> whAGVs ; // manage AGV A* search and movement through graph
    
    void InitialiseGraph(string, string, string, YAML::Node) ; // read in configuration files and construct Graph
    void InitialiseAGVs(YAML::Node) ; // create AGVs to move in graph
    void InitialiseNewEpoch() ; // reset simulation for each episode/epoch
    
    vector< vector<size_t> > RandomiseTeams(size_t) ; // shuffle agent populations
    
    virtual void QueryMATeam(vector<size_t> memberIDs, vector<double> &a, vector<size_t> &s){
      std::cout << "This function queries the multiagent team for its graph costs.\n" ;
    }

    void UpdateGraphCosts(vector<double>) ;
    
//    virutal void GetJointState(vector<Edge *> e, vector<size_t> &eNum, vector<double> &eTime) ;
//    virtual void GetJointState(vector<Edge *> e, vector<size_t> &s) ;
//    virtual size_t GetAgentID(int) ;
    
    bool outputEvals ;
    bool outputEpReplay ;
    
    std::ofstream evalFile ;
    std::ofstream agvStateFile ;
    std::ofstream agvEdgeFile ;
    std::ofstream agentStateFile ;
    std::ofstream agentActionFile ;
    
};

#endif // WAREHOUSE_H_
