#ifndef MULTI_NIGHT_BAR_H_
#define MULTI_NIGHT_BAR_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "Agents/BarAgent.h"
#include "Bar.h"

using std::string ;
using std::vector ;
using std::shuffle ;
using namespace Eigen ;

class MultiNightBar{
  public:
    MultiNightBar(size_t numNights, size_t c, size_t numPop, string evalFunc, size_t agents) ;
    ~MultiNightBar() ;
    
    void InitialiseEpoch() ;
    
    void SimulateEpoch(bool train = true) ;
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals() ;
    
    void OutputPerformance(char *) ;
    void OutputActions(char *, char *) ;
    void OutputControlPolicies(char *) ;
    
    void ExecutePolicies(char * readFile, char * storeJoint, char * storeNights, char * storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in control policies and execute in random world, store joint action and bar parameters in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure

  private:
    size_t nNights ;
    size_t capacity ;
    size_t nPop ;
    string evaluationFunction ;
    size_t nAgents ;
    
    vector<BarAgent *> agentTeam ;
    vector<Bar> barNights ;
    
    bool outputEvals ;
    bool outputActs ;
    
    std::ofstream evalFile ;
    std::ofstream actFile ;
    std::ofstream barFile ;
    std::ofstream NNFile ;
    
    vector< vector<size_t> > RandomiseTeams(size_t) ;
} ;

#endif // MULTI_NIGHT_BAR_H_
