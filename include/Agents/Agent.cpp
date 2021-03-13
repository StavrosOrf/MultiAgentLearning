#include "Agent.h"

Agent::Agent(size_t nPop, size_t nIn, size_t nOut, size_t nHidden): popSize(nPop), numIn(nIn), numOut(nOut), numHidden(nHidden){
  AgentNE = new NeuroEvo(numIn, numOut, numHidden, popSize, LOGISTIC) ;
}

Agent::~Agent(){
  delete(AgentNE) ;
  AgentNE = 0 ;
}
  
void Agent::ResetEpochEvals(){
  // Re-initialise size of evaluations vector
  vector<double> evals(2*popSize,0) ;
  epochEvals = evals ;
}
    
VectorXd Agent::ExecuteNNControlPolicy(size_t i, VectorXd s){
  VectorXd output = AgentNE->GetNNIndex(i)->EvaluateNN(s);
  return output ;
}

void Agent::SetEpochPerformance(double G, size_t i){
  epochEvals[i] = G ;
}

void Agent::EvolvePolicies(bool init){
  if (!init){
    AgentNE->EvolvePopulation(epochEvals) ;
  }
  
  AgentNE->MutatePopulation() ;
}
    
void Agent::OutputNNs(string A){
  // Filename to write to stored in A
  std::ofstream NNFile ;
  NNFile.open(A.c_str(),std::ios::app) ;
  
  // Write in all policies
  for (size_t i = 0; i < popSize*2; i++){
    NeuralNet * NN = AgentNE->GetNNIndex(i) ;
    MatrixXd NNA = NN->GetWeightsA() ;
    for (int j = 0; j < NNA.rows(); j++){
      for (int k = 0; k < NNA.cols(); k++)
        NNFile << NNA(j,k) << "," ;
      NNFile << "\n" ;
    }
    
    MatrixXd NNB = NN->GetWeightsB() ;
    for (int j = 0; j < NNB.rows(); j++){
      for (int k = 0; k < NNB.cols(); k++)
        NNFile << NNB(j,k) << "," ;
      NNFile << "\n" ;
    }
  }
  NNFile.close() ;
}
