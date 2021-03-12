#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiNightBar.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  std::cout << "Testing MultiNightBar class in MultiNightBar.h\n" ;
  
  size_t capacity = 10 ;
  
  size_t nNights = 10 ;
  size_t nPop = 15 ;
  size_t agents = 1000 ;
  string evalFunc = "D" ;
  size_t nEps = 1000 ;
  
  size_t nInputs = 1 ;
  size_t nHidden = 16 ;
  size_t nOutputs = 1 ;
  
  std::cout << "This program will evolve a " << agents << "-agent team over " << nEps << " learning epochs\n" ;
  std::cout << "Agent NN control policy parameters:\n" ;
  std::cout << "  Input dimensions: " << nInputs << "\n" ;
  std::cout << "  Hidden units: " << nHidden << "\n" ;
  std::cout << "  Output dimensions: " << nOutputs << "\n" ;
  std::cout << "CCEA parameters:\n" ;
  std::cout << "  Population size: " << nPop << "\n" ;
  std::cout << "  Evaluation function: " << evalFunc << "\n" ;
  std::cout << "Environment parameters:\n" ;
  std::cout << "  Bar capacity: " << capacity << "\n" ;
  std::cout << "  Number of nights: " << nNights << "\n" ;
  
  int trialNum ;
  std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  std::cin >> trialNum ;
  
  MultiNightBar trainDomain(nNights, capacity, nPop, evalFunc, agents) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/%d_nights/%d_epochs/MultiNightBar/%s/%d",(int)nNights,(int)nEps,evalFunc.c_str(),trialNum) ;
  char mkdir[buffSize] ;
  sprintf(mkdir,"mkdir -p %s",fileDir) ;
  system(mkdir) ;
  
  std::cout << "\nWriting log files to: " << fileDir << "\n\n" ;
  
  char eFile[buffSize] ;
  sprintf(eFile,"%s/results.txt",fileDir) ;
  char aFile[buffSize] ;
  sprintf(aFile,"%s/actions.txt",fileDir) ;
  char cFile[buffSize] ;
  sprintf(cFile,"%s/capacities.txt",fileDir) ;
  
  trainDomain.OutputPerformance(eFile) ;
  
  for (size_t n = 0; n < nEps; n++){
    std::cout << "Episode " << n << "..." ;
    if (n == 0){
      trainDomain.EvolvePolicies(true) ;
      trainDomain.InitialiseEpoch() ;
    }
    else
      trainDomain.EvolvePolicies() ;
    
    if (n == nEps-1)
      trainDomain.OutputActions(aFile, cFile) ;
    
    trainDomain.ResetEpochEvals() ;
    trainDomain.SimulateEpoch() ;
  }
  
  char NNFile[buffSize] ;
  sprintf(NNFile,"%s/NNs.txt",fileDir) ;
  
  std::cout << "\nWriting final control policies to file...\n" ;
  
  trainDomain.OutputControlPolicies(NNFile) ;
  
  int isTest ;
  std::cout << "Please enter [0] to end program, [1] to test stored NN policies in new environment: " ;
  std::cin >> isTest ;
  if (isTest < 0 || isTest > 1){
    std::cout << "Input is out of range. Exiting program.\n" ;
    isTest = 0 ;
  } 
  
  if (isTest == 1){
    
    capacity = 10 ;
    nNights = 10 ;
    std::cout << "Test parameters:\n" ;
    std::cout << "  Bar capacity: " << capacity << "\n" ;
    std::cout << "  Number of nights: " << nNights << "\n" ;
    
    std::cout << "Testing stored control policies on new world...\n" ;
    
    MultiNightBar testDomain(nNights, capacity, nPop, evalFunc, agents) ;

    char eeFile[buffSize] ;
    sprintf(eeFile,"%s/results_test.txt",fileDir) ;
    char aaFile[buffSize] ;
    sprintf(aaFile,"%s/actions_test.txt",fileDir) ;
    char ccFile[buffSize] ;
    sprintf(ccFile,"%s/bars_test.txt",fileDir) ;
    
    testDomain.ExecutePolicies(NNFile, aaFile, ccFile, eeFile, nInputs, nOutputs, nHidden) ;
  }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
