#include <iostream>
#include "Learning/NeuralNet.h"
#include "Utilities/Utilities.h"
#include <Eigen/Eigen>

int main(){
  std::cout << "Testing NeuralNet class in NeuralNet.h\n" ;
  std::cout << "Initialising neural network...\n" ;
  size_t numIn = 3 ;
  size_t numOut = 2 ;
  size_t numHidden = 4 ;
  NeuralNet myNN(numIn,numOut,numHidden) ; // numIn, numOut, numHidden
  
  const char strA[] = "initialA.txt" ;
  const char strB[] = "initialB.txt" ;
  
  std::cout << "Saving initial weights...\n" ;
  myNN.OutputNN(strA,strB) ;
  MatrixXd A0 = myNN.GetWeightsA() ;
  MatrixXd B0 = myNN.GetWeightsB() ;
  for (int i = 0; i < A0.rows(); i++){
    for (int j = 0; j < A0.cols(); j++){
      std::cout << A0(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B0.rows(); i++){
    for (int j = 0; j < B0.cols(); j++){
      std::cout << B0(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Testing input/output calculations...\n" ;
  VectorXd input(numIn) ;
  std::cout << "Input values: [" ;
  for (size_t i = 0; i < numIn; i++){
    input(i) = rand_interval(-1,1) ;
    std::cout << input(i) << "," ;
  }
  std::cout << "]\n" ;
  VectorXd output = myNN.EvaluateNN(input) ;
  std::cout << "Output values: [" ;
  for (size_t i = 0; i < numOut; i++)
    std::cout << output(0) << "," ;
  std::cout << "]\n" ;
  
  std::cout << "Testing mutation...\n" ;
  myNN.MutateWeights() ;
  
  const char strC[] = "newA.txt" ;
  const char strD[] = "newB.txt" ;
  
  std::cout << "Saving new weights...\n" ;
  myNN.OutputNN(strC,strD) ;
  MatrixXd A1 = myNN.GetWeightsA() ;
  MatrixXd B1 = myNN.GetWeightsB() ;
  
  for (int i = 0; i < A1.rows(); i++){
    for (int j = 0; j < A1.cols(); j++){
      std::cout << A1(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B1.rows(); i++){
    for (int j = 0; j < B1.cols(); j++){
      std::cout << B1(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Resetting to previous weights...\n" ;
  myNN.SetWeights(A0,B0) ;
  MatrixXd A2 = myNN.GetWeightsA() ;
  MatrixXd B2 = myNN.GetWeightsB() ;
  
  for (int i = 0; i < A2.rows(); i++){
    for (int j = 0; j < A2.cols(); j++){
      std::cout << A2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B2.rows(); i++){
    for (int j = 0; j < B2.cols(); j++){
      std::cout << B2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Testing function approximation via backpropagation..\n" ;
  std::cout << "Initialising neural network...\n" ;
  size_t newIn = 1 ;
  size_t newOut = 1 ;
  size_t newHidden = 4 ;
  NeuralNet backpropNN(newIn,newOut,newHidden,TANH,UNBOUNDED) ;
  vector<VectorXd> trainInputs ;
  vector<VectorXd> trainTargets ;
  size_t numTrain = 1000 ;
  for (size_t i = 0; i < numTrain; i++){
    VectorXd x(1) ;
    x(0) = easymath::rand_interval(-2.0,2.0) ;
    trainInputs.push_back(x) ;
    VectorXd y(1) ;
    y(0) = pow(x(0),3) ; // approximate f(x) = x^3
    trainTargets.push_back(y) ;
  }
  std::cout << "Training via backpropagation...\n" ;
  backpropNN.BackPropagation(trainInputs,trainTargets) ;
  
  
  size_t numTest = 10 ;
  std::cout << "(Truth, approximated, difference, percentage)\n" ;
  for (size_t i = 0; i < numTest; i++){
    VectorXd x(1) ;
    x(0) = 5.0*(-1.0 + 2.0*(double)i/(double)numTest) ;
    VectorXd y = backpropNN.EvaluateNN(x) ;
    double t = pow(x(0),3) ;
    std::cout << "(" << t << "," << y(0) << "," << t-y(0) << "," << 100.0*(t-y(0))/t << ")\n" ; ;
  }
  std::cout << "\n" ;
  
  std::cout << "Testing complete!\n" ;
  return 0 ;
}
