#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>
#include <stdlib.h>

#include "Domains/MultiRover.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(25.0) ;
  world.push_back(0.0) ; 
  world.push_back(25.0) ;   // Dimensions of testing arena [xmin, xmax, ymin, ymax]
  
  size_t rovs = 10 ;         // Number of rovers
  size_t nPOIs = 10 ;        // Number of POIs
  int coupling = 1 ;        // Number of simultaneous observations required
  
  size_t nSteps = 25 ;      // Number of timesteps in each learning epoch
  size_t nEps = 1000 ;      // Number of learning epochs
  
  size_t nInputs = 8 ;      // Dimension of state inputs to NN: keep fixed at 8
  size_t nHidden = 16 ;     // Number of hidden units
  size_t nOutputs = 2 ;     // Dimension of action outputs of NN: keep fixed at 2
  
  size_t nPop = 25 ;        // Number of members in original population
  string evalFunc = "D" ;   // Fitness evaluation function {"D","G"}
  bool pLearn = true ;      // Apply pLearners
  double tau = 1.0 ;       // Temperature value for pLearners
  
  std::cout << "Generating expert policies for rover domain using " << evalFunc << " fitness.\n" ;
  
  bool staticOrRandom = true ;  // 0 - training epochs use the same POI and rover initial configuration, 1 - randomized configurations for each learning epoch. Note that each of the 2k multiagent teams in each epoch are still trained on the same configuration.
  
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
  if (!staticOrRandom)
    std::cout << "  Training worlds: each identical\n" ;
  else
    std::cout << "  Training worlds: each randomly generated\n" ;
  if (pLearn){
    std::cout << "Applying probabilistic learner agents\n" ;
    std::cout << "  tau: " << tau << "\n" ;
  }
  
  size_t totalTrials = 20 ;
  std::cout << "Total number of statistical trials: " << totalTrials << "\n" ;
  
  for (size_t trialNum = 0; trialNum < totalTrials; trialNum++){
  
    srand(trialNum) ;
    MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs, coupling) ;
    string pLearnDir ;
    if (pLearn){
      trainDomain.SetLearningEvaluation(tau) ;
      pLearnDir = "pL" ;
    }
    else{
      pLearnDir = "L" ;
    }
    
    int buffSize = 100 ;
    char fileDir[buffSize] ;
    sprintf(fileDir,"Results/Multirover_probabilistic_learners/%s/%s/%d_square/tau_%.1f/Gmax/%d",evalFunc.c_str(),pLearnDir.c_str(),(int)world[1],tau,trialNum) ;
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
    char iFile[buffSize] ;
    sprintf(iFile,"%s/impacts_",fileDir) ;
    char lFile[buffSize] ;
    sprintf(lFile,"%s/learners.txt",fileDir) ;
    
    char cFile[buffSize] ;
    sprintf(cFile,"%s/config.txt",fileDir) ;
    std::stringstream fileName ;
    fileName << cFile ;
    std::ofstream configFile ;
    if (configFile.is_open())
      configFile.close() ;
    configFile.open(fileName.str().c_str(),std::ios::app) ;
    
    configFile << "world: [" << world[0] << "," << world[1] << "," << world[2] << "," << world[3] << "]\n" ;
    if (!staticOrRandom)
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
    if (pLearn){
      configFile << "tau: " << tau << "\n" ;
    }
    configFile.close() ;
    
    trainDomain.OutputPerformance(eFile) ;
    
    for (size_t n = 0; n < nEps; n++){
      std::cout << "Trial: " << trialNum << ", episode " << n << "..." ;
      if (n == 0){
        trainDomain.EvolvePolicies(true) ;
        if (!staticOrRandom)
          trainDomain.InitialiseEpoch() ; // Static world
      }
      else
        trainDomain.EvolvePolicies() ;
      
      if (staticOrRandom)
        trainDomain.InitialiseEpoch() ; // Random worlds
      
      if (n == nEps-1)
        trainDomain.OutputTrajectories(tFile, pFile) ;
      
      trainDomain.ResetEpochEvals() ;
      trainDomain.SimulateEpoch() ;
      if (pLearn){
        trainDomain.OutputImpacts(iFile) ;
      }
      trainDomain.OutputLearners(lFile) ;
    }
    
    char NNFile[buffSize] ;
    sprintf(NNFile,"%s/NNs.txt",fileDir) ;
    
    std::cout << "\nWriting final control policies to file...\n" ;
    
    trainDomain.OutputControlPolicies(NNFile) ;
  
  }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
