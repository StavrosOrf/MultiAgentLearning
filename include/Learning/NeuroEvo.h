#ifndef NEUR0_EVO_H_
#define NEUR0_EVO_H_

#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <math.h>
#include "NeuralNet.h"

using std::vector ;
using std::sort ;
using namespace Eigen ;

class NeuroEvo{
  public:
    NeuroEvo(size_t, size_t, size_t, size_t, actFun afType=TANH) ; // nIn, nOut, nHidden, popSize, activation function
    ~NeuroEvo() ;
    
    void MutatePopulation() ;
    void EvolvePopulation(vector<double>) ;
    vector<double> GetAllEvaluations() ;
    
    NeuralNet * GetNNIndex(size_t i){return populationNN[i] ;}
    size_t GetCurrentPopSize(){return populationNN.size() ;}
    
    void SetMutationNormLog(bool b=true){computeMutationNorms = b ;}
    vector<double> GetMutationNorm(){return mutationFrobeniusNorm ;}
  private:
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    actFun activationFunction ;
    
    size_t populationSize ;
    vector<NeuralNet *> populationNN ;
    
    void (NeuroEvo::*SurvivalFunction)() ;
    void BinaryTournament() ;
    void RetainBestHalf() ;
    static bool CompareEvaluations(NeuralNet *, NeuralNet *) ;
    
    bool computeMutationNorms ;
    vector<double> mutationFrobeniusNorm ;
    double ComputeFrobeniusNorm(MatrixXd, MatrixXd, MatrixXd, MatrixXd) ;
} ;
#endif // NEUR0_EVO_H_
