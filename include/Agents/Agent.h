#ifndef AGENT_H_
#define AGENT_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"

using std::vector ;
using std::string ;

class Agent{
  public:
    Agent(size_t nPop, size_t nIn, size_t nOut, size_t nHidden) ;
    ~Agent() ;
    
    void ResetEpochEvals() ;
    
    VectorXd ExecuteNNControlPolicy(size_t, VectorXd) ;
    void SetEpochPerformance(double G, size_t i) ;
    vector<double> GetEpochEvals(){return epochEvals ;}
    
    void EvolvePolicies(bool init = false) ;
    
    void OutputNNs(string) ;
    NeuroEvo * GetNEPopulation(){return AgentNE ;}
    
    size_t GetNumIn(){return numIn ;}
    size_t GetNumHidden(){return numHidden ;}
    size_t GetNumOut(){return numOut ;}
    
  protected:
    size_t popSize ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    vector<double> epochEvals ;
    NeuroEvo * AgentNE ;
};

#endif // AGENT_H_
