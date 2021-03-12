#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include "Learning/NeuroEvo.h"
#include "Utilities/MatrixTypes.h"

int main(){
  std::cout << "Testing NeuroEvo class in NeuroEvo.h\n" ;
  
  size_t inputs = 4 ;
  size_t outputs = 2 ;
  size_t hidden = 4 ;
  size_t popSize = 10 ;
  
  std::cout << "Initialising population...\n" ;
  NeuroEvo myNE(inputs, outputs, hidden, popSize) ;
  
  for (size_t k = 0; k < 10; k++){
  
    std::cout << "LOOP " << k << std::endl ;
  
    std::cout << "Doubling population via mutations...\n" ;
    myNE.MutatePopulation() ;

    std::cout << "Generating evaluations...\n" ;
    matrix1d evaluations ;
    for (size_t i = 0; i < 2*popSize; i++)
      evaluations.push_back((double) i) ;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;
    shuffle(evaluations.begin(),evaluations.end(),std::default_random_engine(seed)) ;

    for (size_t i = 0; i < 2*popSize; i++)
      std::cout << evaluations[i] << " " ;
    std::cout << std::endl ;

    std::cout << "Evolving population according to evaluations...\n" ;
    myNE.EvolvePopulation(evaluations) ;

    std::cout << "Retained population has evaluations:\n" ;
    matrix1d retainedEvals = myNE.GetAllEvaluations() ;
    for (size_t i = 0; i < retainedEvals.size(); i++)
      std::cout << retainedEvals[i] << " " ;
    std::cout << std::endl ;
  
  }
  
  std::cout << "Testing complete!\n" ;
  return 0 ;
}
