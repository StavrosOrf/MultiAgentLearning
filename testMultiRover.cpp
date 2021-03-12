#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  std::cout << "Testing MultiRover class in MultiRover.h\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  
  size_t nPOIs = 25 ;
  size_t nSteps = 100 ;
  size_t nPop = 15 ;
  size_t rovs = 5 ;
  string evalFunc = "D" ;
  size_t nEps = 1000 ;
  
  size_t nInputs = 8 ;
  size_t nHidden = 16 ;
  size_t nOutputs = 2 ;
  
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
  
  int trialNum ;
  std::cout << "Please enter trial number [NOTE: no checks enabled to prevent overwriting existing files, user must make sure trial number is unique]: " ;
  std::cin >> trialNum ;
  
  int staticOrRandom ;
  std::cout << "Please enter [0] for static world training, [1] for random world training: " ;
  std::cin >> staticOrRandom ;
  if (staticOrRandom < 0 || staticOrRandom > 1){
    std::cout << "Input is out of range. Setting to static world training.\n" ;
    staticOrRandom = 0 ;
  } 
  
  MultiRover trainDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  if (staticOrRandom == 0)
    sprintf(fileDir,"Results/%d_square/Static_world/%d_epochs/MultiRover/UpdatedActivation/%s/%d",(int)world[1],(int)nEps,evalFunc.c_str(),trialNum) ;
  else
    sprintf(fileDir,"Results/%d_square/Random_worlds/%d_epochs/MultiRover/UpdatedActivation/%s/%d",(int)world[1],(int)nEps,evalFunc.c_str(),trialNum) ;
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
  
  int isTest ;
  std::cout << "Please enter [0] to end program, [1] to test stored NN policies in new environment: " ;
  std::cin >> isTest ;
  if (isTest < 0 || isTest > 1){
    std::cout << "Input is out of range. Exiting program.\n" ;
    isTest = 0 ;
  } 
  
  if (isTest == 1){
  
    int isIntrospection ;
    std::cout << "Please enter [0] to run basic test, [1] to include robot introspection and querying for mission objective changes: " ;
    std::cin >> isIntrospection ;
    if (isIntrospection < 0 || isIntrospection > 1){
      std::cout << "Input is out of range. Running basic test.\n" ;
      isIntrospection = 0 ;
    }
    
    world.clear() ;
    world.push_back(0.0) ; 
    world.push_back(100.0) ;
    world.push_back(0.0) ; 
    world.push_back(100.0) ;
    nPOIs = 25 ;
    nSteps = 1000 ;
    std::cout << "Test parameters:\n" ;
    std::cout << "  World size: " << world[1] << " x " << world[3] << "\n" ;
    std::cout << "  Number of POIs: " << nPOIs << "\n" ;
    std::cout << "  Number of timesteps: " << nSteps << "\n" ;
    
    std::cout << "Testing stored control policies on new world...\n" ;
    
    MultiRover testDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs) ;

    char eeFile[buffSize] ;
    sprintf(eeFile,"%s/results_test.txt",fileDir) ;
    char ttFile[buffSize] ;
    sprintf(ttFile,"%s/trajectories_test.txt",fileDir) ;
    char ppFile[buffSize] ;
    sprintf(ppFile,"%s/POIs_test.txt",fileDir) ;
    char rrFile[buffSize] ;
    sprintf(rrFile,"%s/avgD_test.txt",fileDir) ;
    
    testDomain.OutputAverageStepwise(rrFile) ; // store average stepwise reward values
    
    if (isIntrospection == 0)
      testDomain.ExecutePolicies(NNFile, ttFile, ppFile, eeFile, nInputs, nOutputs, nHidden) ;
    
    else{
      char qqFile[buffSize] ;
      sprintf(qqFile,"%s/queries_test.txt",fileDir) ;
      char bbFile[buffSize] ;
      sprintf(bbFile,"%s/beliefs_test.txt",fileDir) ;
      char pomdpDir[buffSize] ;
      sprintf(pomdpDir,"../include/POMDPs") ;
      char envFile[buffSize] ;
      sprintf(envFile,"%s/rover_6.pomdp",pomdpDir) ;
      char polFile[buffSize] ;
      sprintf(polFile,"%s/rover_6.policy",pomdpDir) ;
      
      VectorXd prior ;
      prior.setZero(2) ;
      prior(0) = 0.1 ;
      prior(1) = 1.0 - prior(0) ; // begin with high expectation of being an expert
      
      int gPoiID ;
      std::cout << "Please enter the target POI ID [0," << nPOIs << "]: " ;
      std::cin >> gPoiID ;
      if (gPoiID < 0 || gPoiID > (int)nPOIs){
        std::cout << "Input is out of range. Setting target POI to POI 0.\n" ;
        gPoiID = 0 ;
      } 
      
      testDomain.ExecutePolicies(NNFile, ttFile, ppFile, eeFile, qqFile, bbFile, nInputs, nOutputs, nHidden, gPoiID, envFile, polFile, prior) ;
    }
  }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
