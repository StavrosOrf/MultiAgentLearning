#include <iostream>
#include "NeuroEvo.h"

// Constructor: Initialises all NN in population, given NN layer sizes and population size, also sets SurvivalFunction
NeuroEvo::NeuroEvo(size_t nIn, size_t nOut, size_t nHidden, size_t pSize, actFun afType): numIn(nIn), numOut(nOut), numHidden(nHidden), activationFunction(afType), populationSize(pSize){
  for (size_t i = 0; i < populationSize; i++)
    populationNN.push_back(new NeuralNet(numIn, numOut, numHidden, afType)) ;
  SurvivalFunction = &NeuroEvo::BinaryTournament ; // how to decide which NNs to retain after each round of evolution
  computeMutationNorms = false ;
}

// Destructor: Deletes all NN objects from population
NeuroEvo::~NeuroEvo(){
  for (size_t i = 0; i < populationNN.size(); i++){
    delete(populationNN[i]) ;
    populationNN[i] = 0 ;
  }
}

// Double population size by adding NN with mutated weights of existing NN 
void NeuroEvo::MutatePopulation(){
  if (computeMutationNorms){
    mutationFrobeniusNorm.clear() ;
  }
  
  for (size_t i = 0; i < populationSize; i++){
    size_t j = i + populationSize ;
    populationNN.push_back(new NeuralNet(numIn, numOut, numHidden, activationFunction)) ;
    populationNN[j]->SetWeights(populationNN[i]->GetWeightsA(),populationNN[i]->GetWeightsB()) ;
    populationNN[j]->MutateWeights() ;
    
    if (computeMutationNorms){
      mutationFrobeniusNorm.push_back(ComputeFrobeniusNorm(populationNN[i]->GetWeightsA(),populationNN[i]->GetWeightsB(),populationNN[j]->GetWeightsA(),populationNN[j]->GetWeightsB())) ;
    }
  }
}

// Evolve population according to evaluation signal and survival function
void NeuroEvo::EvolvePopulation(vector<double> evaluation){
  for (size_t i = 0; i < 2*populationSize; i++)
    populationNN[i]->SetEvaluation(evaluation[i]) ;
  
  // Shuffle in preparation for comparisons
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;
  shuffle(populationNN.begin(), populationNN.end(), std::default_random_engine(seed)) ;
  
  (this->*SurvivalFunction)() ;
}

// Binary tournament for survival, head to head competition between random pairs of NNs
void NeuroEvo::BinaryTournament(){
  vector<size_t> toErase ;
  for (size_t i = 0; i < populationSize; i++){
    size_t j = i + populationSize ;
    if (populationNN[i]->GetEvaluation() >= populationNN[j]->GetEvaluation()){
      delete(populationNN[j]) ;
      populationNN[j] = 0 ;
      toErase.push_back(j) ;
    }
    else {
      delete(populationNN[i]) ;
      populationNN[i] = 0 ;
      toErase.push_back(i) ;
    }
  }
  std::sort(toErase.begin(),toErase.end()) ;
  for (size_t i = 0; i < toErase.size(); i++){
    size_t j = toErase.size()-1-i ;
    populationNN.erase(populationNN.begin()+toErase[j]) ;
  }
}

// Comparitor function to sort NNs according to evaluation signal (must have strict weak ordering)
bool NeuroEvo::CompareEvaluations(NeuralNet* NN0, NeuralNet* NN1){
  return (NN0->GetEvaluation() > NN1->GetEvaluation()) ;
}

// Retain the best half of the population
void NeuroEvo::RetainBestHalf(){
  std::sort(populationNN.begin(),populationNN.end(),CompareEvaluations) ;
  
  for (size_t i = populationSize; i < 2*populationSize; i++){
    delete(populationNN[i]) ;
    populationNN[i] = 0 ;
  }
  
  // Pop from the back
  for (size_t i = 0; i < populationSize; i++)
    populationNN.pop_back() ;
}

// Return evaluations of all current NNs in population (used for debugging)
vector<double> NeuroEvo::GetAllEvaluations(){
  vector<double> evals ;
  for (size_t i = 0 ; i < populationNN.size(); i++)
    evals.push_back(populationNN[i]->GetEvaluation()) ;
  return evals ;
}

double NeuroEvo::ComputeFrobeniusNorm(MatrixXd A, MatrixXd B, MatrixXd Am, MatrixXd Bm){
  MatrixXd diffA = A-Am ;
  MatrixXd diffB = B-Bm ;
  MatrixXd diffAT = diffA.transpose() ; // congugate transpose of real matrix = transpose of matrix
  MatrixXd diffBT = diffB.transpose() ;
  MatrixXd diffATA = diffAT*diffA ;
  MatrixXd diffBTB = diffBT*diffB ;
  double traceA = diffATA.trace() ;
  double traceB = diffBTB.trace() ;
  return sqrt(traceA) + sqrt(traceB) ;
}
