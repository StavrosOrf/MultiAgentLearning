#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;   // Dimensions of testing arena [xmin, xmax, ymin, ymax]
  
  size_t rovs = 5 ;         // Number of rovers
  size_t nPOIs = 10 ;        // Number of POIs
  int coupling = 1 ;        // Number of simultaneous observations required
  
  size_t nSteps = 30 ;      // Number of timesteps in each learning epoch
  size_t nEps = 200 ;      // Number of learning epochs
  
  size_t nInputs = 8 ;      // Dimension of state inputs to NN: keep fixed at 8
  size_t nHidden = 16 ;     // Number of hidden units
  size_t nOutputs = 2 ;     // Dimension of action outputs of NN: keep fixed at 2
  
  size_t nPop = 15 ;        // Number of members in original population
  string evalFunc = "D" ;   // Fitness evaluation function {"D","G"}
  
  std::cout << "Generating expert policies for rover domain using " << evalFunc << " fitness.\n" ;
  
  int staticOrRandom = 0 ;  // 0 - training epochs use the same POI and rover initial configuration, 1 - randomized configurations for each learning epoch. Note that each of the 2k multiagent teams in each epoch are still trained on the same configuration.
  
  std::cout << "This program will evolve a " << rovs << "-rover team over " << nEps << " learning epochs, each of " << nSteps << " timesteps.\n" ;
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
  std::cout << "  Simultaneous observation requirements: " << coupling << "\n" ;
  if (staticOrRandom == 0)
    std::cout << "  Training worlds: each identical\n" ;
  else
    std::cout << "  Training worlds: each randomly generated\n" ;
  
  int trialNum ;
  std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  std::cin >> trialNum ;
  
  MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs, coupling) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/Multirover_experts/%d",trialNum) ;
  char mkdir[buffSize] ;
  sprintf(mkdir,"mkdir -p %s",fileDir) ;
  system(mkdir) ;
  
  std::cout << "\nWriting log files to: " << fileDir << "\n\n" ;
  
  char eFile[buffSize] ;
  sprintf(eFile,"%s/results.txt",fileDir) ;
  char tFile[buffSize] ;
  sprintf(tFile,"%s/trajectories.txt",fileDir) ;
  char pFile[buffSize] ;
  sprintf(pFile,"%s/POIs.txt",fileDir) ;
  
  char cFile[buffSize] ;
  sprintf(cFile,"%s/config.txt",fileDir) ;
  std::stringstream fileName ;
  fileName << cFile ;
  std::ofstream configFile ;
  if (configFile.is_open())
    configFile.close() ;
  configFile.open(fileName.str().c_str(),std::ios::app) ;
  
  configFile << "world: [" << world[0] << "," << world[1] << "," << world[2] << "," << world[3] << "]\n" ;
  if (staticOrRandom == 0)
    configFile << "world_type: static\n" ;
  else
    configFile << "world_type: random\n" ;
  configFile << "rovers: " << rovs << "\n" ;
  configFile << "POIs: " << nPOIs << "\n" ;
  configFile << "coupling: " << coupling << "\n" ;
  configFile << "timesteps: " << nSteps << "\n" ;
  configFile << "epochs: " << nEps << "\n" ;
  configFile << "NN:\n" ;
  configFile << "  inputs: " << nInputs << "\n" ;
  configFile << "  hidden: " << nHidden << "\n" ;
  configFile << "  outputs: " << nOutputs << "\n" ;
  configFile << "pop_size: " << nPop << "\n" ;
  configFile << "fitness: " << evalFunc << "\n" ;
  configFile.close() ;
  
  trainDomain.OutputPerformance(eFile) ;
  
  for (size_t n = 0; n < nEps; n++){
    std::cout << "Episode " << n << "..." ;
    if (n == 0){
      trainDomain.EvolvePolicies(true) ;
      if (staticOrRandom == 0)
        trainDomain.InitialiseEpoch() ; // Static world
    }
    else
      trainDomain.EvolvePolicies() ;
    
    if (staticOrRandom == 1)
      trainDomain.InitialiseEpoch() ; // Random worlds
    
    if (n == nEps-1)
      trainDomain.OutputTrajectories(tFile, pFile) ;
    
    trainDomain.ResetEpochEvals() ;
    trainDomain.SimulateEpoch() ;
  }
  
  char NNFile[buffSize] ;
  sprintf(NNFile,"%s/NNs.txt",fileDir) ;
  
  std::cout << "\nWriting final control policies to file...\n" ;
  
  trainDomain.OutputControlPolicies(NNFile) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
