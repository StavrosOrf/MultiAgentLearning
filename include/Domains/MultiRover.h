#ifndef MULTIROVER_H_
#define MULTIROVER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "Agents/Rover.h"
#include "Target.h"
#include "POMDPs/POMDP.h"

using std::string ;
using std::vector ;
using std::shuffle ;
using namespace Eigen ;

class MultiRover{
  public:
    MultiRover(vector<double>, size_t, size_t, size_t, string, size_t, int c = 1) ;
    ~MultiRover() ;
    
    void InitialiseEpoch() ;
    
    void SimulateEpoch(bool train = true) ;
    void SimulateEpoch(size_t goalPOI, char * pomdpEnv, char * pomdpPolicy, VectorXd prior) ;
    void EvolvePolicies(bool init = false) ;
    void ResetEpochEvals() ;
    void SetLearningEvaluation(double) ;
    
    void OutputPerformance(char *) ;
    void OutputTrajectories(char *, char *) ;
    void OutputControlPolicies(char *) ;
    void OutputQueries(char *) ;
    void OutputBeliefs(char *) ;
    void OutputAverageStepwise(char *) ;
    void OutputImpacts(char *) ;
    void OutputLearners(char *) ;
    
    void ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char * storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure
    
    void ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char * storeEval, char * storeQury, char* storeBlf, size_t numIn, size_t numOut, size_t numHidden, size_t goalPOI, char * pomdpEnv, char * pomdpPolicy, VectorXd prior) ; // goalPOI observation triggers mission change, {pomdpEnv file stores POMDP environment, pomdpPolicy stores pomdp policy, prior stores prior belief} for determining inquiry action based on policy expertise belief
    
    void ExecutePolicies(char * expFile, char * novFile, char * storeTraj, char * storePOI, char* storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in expert and novice control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure
  private:
    vector<double> world ;
    size_t nSteps ;
    size_t nPop ;
    size_t nPOIs ;
    string evaluationFunction ;
    size_t nRovers ;
    int coupling ;
    
    vector<Vector2d> initialXYs ;
    vector<double> initialPsis ;
    
    vector<Rover *> roverTeam ;
    vector<Target> POIs ;
    bool gPOIObs ;
    
    bool outputEvals ;
    bool outputTrajs ;
    bool outputNNs ;
    bool outputQury ;
    bool outputBlf ;
    bool outputAvgStepR ;
    
    std::ofstream evalFile ;
    std::ofstream trajFile ;
    std::ofstream POIFile ;
    std::ofstream NNFile ;
    std::ofstream quryFile ;
    std::ofstream blfFile ;
    std::ofstream avgStepRFile ;
    
    vector< vector<size_t> > RandomiseTeams(size_t) ;
} ;

#endif // MULTIROVER_H_
