// Some code migrated from rebhuhnc/libraries/SingleAgent/NeuralNet/NeuralNet.h
#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

enum actFun {TANH, LOGISTIC} ; // available activation functions
enum nnOut {BOUNDED, UNBOUNDED} ; // bounded output will apply activation function on last layer

#include <stdio.h>
#include <iostream>
#include <float.h>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include "Utilities/Utilities.h"

using namespace Eigen ;
using easymath::rand_interval ;
using std::vector ;

class NeuralNet{
  public:
    NeuralNet(size_t numIn, size_t numOut, size_t numHidden, actFun afType=TANH, nnOut bOut=BOUNDED) ; // single hidden layer
    ~NeuralNet(){}
    
    VectorXd EvaluateNN(VectorXd inputs) ;
    VectorXd EvaluateNN(VectorXd inputs, VectorXd & hiddenLayer) ;
    void MutateWeights() ;
    void SetWeights(MatrixXd, MatrixXd) ;
    MatrixXd GetWeightsA() {return weightsA ;}
    MatrixXd GetWeightsB() {return weightsB ;}
    void OutputNN(const char *, const char *) ; // write NN weights to file
    double GetEvaluation() {return evaluation ;}
    void SetEvaluation(double eval) {evaluation = eval ;}
    void BackPropagation(vector<VectorXd> trainInputs, vector<VectorXd> trainTargets) ;
  private:
    double bias ;
    MatrixXd weightsA ;
    MatrixXd weightsB ;
    double mutationRate ;
    double mutationStd ;
    double evaluation ;
    double eta ;
    vector<size_t> layerActivation ;

    void InitialiseWeights(MatrixXd &) ;
    VectorXd (NeuralNet::*ActivationFunction)(VectorXd, size_t) ;
    VectorXd HyperbolicTangent(VectorXd, size_t) ; // outputs between [-1,1]
    VectorXd LogisticFunction(VectorXd, size_t) ; // outputs between [0,1]
    double RandomMutation() ;
    void WriteNN(MatrixXd, std::stringstream &) ;
} ;
#endif // NEURAL_NET_H_
