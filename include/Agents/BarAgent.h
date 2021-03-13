#ifndef BAR_AGENT_H_
#define BAR_AGENT_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Domains/Bar.h"
#include "Utilities/Utilities.h"

using std::vector ;
using std::string ;
using namespace Eigen ;
using namespace easymath ;

class BarAgent{
  public:
    BarAgent(size_t nPop, string evalFunc, size_t nNights, actFun afType=LOGISTIC) ;
    ~BarAgent() ;
    
    void ResetEpochEvals() ;
    void InitialiseNewLearningEpoch(vector<Bar>) ;
    
    int ExecuteNNControlPolicy(size_t) ; // executes NN_i, outputs index of action (assumes discrete set of actions)
    void ComputeEval(vector<int>, size_t, double) ;
    void SetEpochPerformance(double G, size_t i) ;
    
    void EvolvePolicies(bool init = false) ;
    
    void UseProbabilisticEvolution(){
      probabilisticEvolution = true ;
      pEvolveScale = 1.0 ;
    }
    
    // Probability of entering evolution and mutation step
    double ProbabilityOfEvolution() ;
    
    void OutputNNs(char *) ;
    NeuroEvo * GetNEPopulation(){return AgentNE ;}
  private:
    size_t popSize ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    size_t numNights ;
    
    vector<Bar> barNights ;
    bool isD ;
    double D ;
    vector<double> epochEvals ;
    NeuroEvo * AgentNE ;
    
    int curAction ;
    vector<double> allActions ;
    
    bool probabilisticEvolution ;
    double pEvolveScale ;
    
    void DifferenceEvaluationFunction(vector<int>, size_t, double) ;
    
    // Compute \delta pi across mutation
    vector<double> ComputeChange(bool) ;
} ;

#endif // BAR_AGENT_H_
