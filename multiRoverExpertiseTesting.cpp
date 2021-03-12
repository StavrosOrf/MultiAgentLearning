#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  std::cout << "Generating expert policies for rover domain using D fitness.\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  
  size_t rovs = 4 ;         // Variable team size from 2-5
  size_t nPOIs = 25 ;       // Keep fixed at 25
  
  size_t nSteps = 200 ;     // Keep fixed at 200
  
  size_t nInputs = 8 ;      // Keep fixed at 8
  size_t nHidden = 16 ;     // Keep fixed at 16
  size_t nOutputs = 2 ;     // Keep fixed at 2
  
  size_t nPop = 15 ;        // Keep fixed at 15
  string evalFunc = "D" ;   // Keep fixed at D
  
  std::cout << "This program will test a " << rovs << "-rover team over " << nSteps << " timesteps.\n" ;
  std::cout << "Rover NN control policy parameters:\n" ;
  std::cout << "  Input dimensions: " << nInputs << "\n" ;
  std::cout << "  Hidden units: " << nHidden << "\n" ;
  std::cout << "  Output dimensions: " << nOutputs << "\n" ;
  std::cout << "CCEA parameters:\n" ;
  std::cout << "  Population size: " << nPop << "\n" ;
  std::cout << "  Evaluation function: " << evalFunc << "\n" ;
  std::cout << "Environment parameters:\n" ;
  std::cout << "  World size: " << world[1] << " x " << world[3] << "\n" ;
  std::cout << "  Number of POIs: " << nPOIs << "\n" ;
  std::cout << "  Training worlds: each randomly generated\n" ;
  
  int trialNum ;
  std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  std::cin >> trialNum ;
  
  MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/Multirover_experts/Mixed_teams/%d",trialNum) ;
  char mkdir[buffSize] ;
  sprintf(mkdir,"mkdir -p %s",fileDir) ;
  system(mkdir) ;
  
  std::cout << "\nWriting log files to: " << fileDir << "\n\n" ;
  
  char cFile[buffSize] ;
  sprintf(cFile,"%s/config.txt",fileDir) ;
  std::stringstream fileName ;
  fileName << cFile ;
  std::ofstream configFile ;
  if (configFile.is_open())
    configFile.close() ;
  configFile.open(fileName.str().c_str(),std::ios::app) ;
  
  configFile << "world: [" << world[0] << "," << world[1] << "," << world[2] << "," << world[3] << "]\n" ;
  configFile << "world_type: random\n" ;
  configFile << "rovers: " << rovs << "\n" ;
  configFile << "POIs: " << nPOIs << "\n" ;
  configFile << "timesteps: " << nSteps << "\n" ;
  configFile << "epochs: 1\n" ;
  configFile << "NN:\n" ;
  configFile << "  inputs: " << nInputs << "\n" ;
  configFile << "  hidden: " << nHidden << "\n" ;
  configFile << "  outputs: " << nOutputs << "\n" ;
  configFile << "pop_size: " << nPop << "\n" ;
  configFile << "fitness: " << evalFunc << "\n" ;
  configFile.close() ;
  
  std::cout << "Testing stored control policies on new world...\n" ;
  
  MultiRover testDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs) ;

  char eeFile[buffSize] ;
  sprintf(eeFile,"%s/results_test.txt",fileDir) ;
  char ttFile[buffSize] ;
  sprintf(ttFile,"%s/trajectories_test.txt",fileDir) ;
  char ppFile[buffSize] ;
  sprintf(ppFile,"%s/POIs_test.txt",fileDir) ;
  char rrFile[buffSize] ;
  sprintf(rrFile,"%s/avgR_test.txt",fileDir) ;
  
  char expNNFile[buffSize] ;
  sprintf(expNNFile,"Results/Multirover_experts/5-team/NNs.txt") ;
  char novNNFile[buffSize] ;
  sprintf(novNNFile,"Results/Multirover_experts/1-team/NNs.txt") ;
  
  testDomain.OutputAverageStepwise(rrFile) ; // store average stepwise reward values
  
  testDomain.ExecutePolicies(expNNFile, novNNFile, ttFile, ppFile, eeFile, nInputs, nOutputs, nHidden) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
