#include <iostream>
#include "Learning/NeuralNet.h"
#include "Utilities/MatrixTypes.h"

int main(){
  std::cout << "Testing NeuralNet class in NeuralNet.h\n" ;
  std::cout << "Initialising neural network...\n" ;
  NeuralNet myNN(4,2,4) ; // numIn, numOut, numHidden
  
  const char strA[] = "initialA.txt" ;
  const char strB[] = "initialB.txt" ;
  
  std::cout << "Saving initial weights...\n" ;
  myNN.OutputNN(strA,strB) ;
  matrix2d A0 = myNN.GetWeightsA() ;
  matrix2d B0 = myNN.GetWeightsB() ;
  for (size_t i = 0; i < A0.size(); i++){
    for (size_t j = 0; j < A0[0].size(); j++){
      std::cout << A0[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  for (size_t i = 0; i < B0.size(); i++){
    for (size_t j = 0; j < B0[0].size(); j++){
      std::cout << B0[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Testing input/output calculations...\n" ;
  matrix1d input ;
  for (size_t i = 0; i < 4; i++)
    input.push_back((double)i) ;
  std::cout << "Input values: [" << input[0] << "," << input[1] << "," << input[2] << "," << input[3] << "]\n" ;
  matrix1d output = myNN.EvaluateNN(input) ;
  std::cout << "Output values: [" << output[0] << "," << output[1] << "]\n" ;
  
  std::cout << "Testing mutation...\n" ;
  myNN.MutateWeights() ;
  
  const char strC[] = "newA.txt" ;
  const char strD[] = "newB.txt" ;
  
  std::cout << "Saving new weights...\n" ;
  myNN.OutputNN(strC,strD) ;
  matrix2d A1 = myNN.GetWeightsA() ;
  matrix2d B1 = myNN.GetWeightsB() ;
  
  for (size_t i = 0; i < A1.size(); i++){
    for (size_t j = 0; j < A1[0].size(); j++){
      std::cout << A1[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  for (size_t i = 0; i < B1.size(); i++){
    for (size_t j = 0; j < B1[0].size(); j++){
      std::cout << B1[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Resetting to previous weights...\n" ;
  myNN.SetWeights(A0,B0) ;
  matrix2d A2 = myNN.GetWeightsA() ;
  matrix2d B2 = myNN.GetWeightsB() ;
  
  for (size_t i = 0; i < A2.size(); i++){
    for (size_t j = 0; j < A2[0].size(); j++){
      std::cout << A2[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  for (size_t i = 0; i < B2.size(); i++){
    for (size_t j = 0; j < B2[0].size(); j++){
      std::cout << B2[i][j] << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Testing complete!\n" ;
  return 0 ;
}
