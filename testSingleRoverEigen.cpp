#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "Domains/SingleRover.h"

int main(){
  std::cout << "Testing SingleRover class in SingleRover.h\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  
  size_t nPOIs = 10 ;
  size_t nSteps = 30 ;
  size_t nPop = 15 ;
  size_t nRuns = 1 ;
  
  for (size_t i = 0; i < nRuns; i++){
    std::cout << "Initialising rover domain " << i << "...\n" ;
    SingleRover testDomain(world, nPOIs, nSteps, nPop) ;

    int buffSize = 100 ;
    char fileDir[buffSize] ;
    sprintf(fileDir,"Results/30_square/Random_worlds/100_epochs/Eigen/%d",(int)i) ;
    char mkdir[buffSize] ;
    sprintf(mkdir,"mkdir -p %s",fileDir) ;
    system(mkdir) ;
    char eFile[buffSize] ;
    sprintf(eFile,"%s/results.txt",fileDir) ;
    std::cout << eFile << std::endl ;
    testDomain.OutputPerformance(eFile) ;

    char tFile[buffSize] ;
    sprintf(tFile,"%s/trajectories.txt",fileDir) ;
    char pFile[buffSize] ;
    sprintf(pFile,"%s/POIs.txt",fileDir) ;
    testDomain.OutputTrajectories(tFile,pFile) ;
    
    char nnFile[buffSize] ;
    sprintf(nnFile,"%s/NNs",fileDir) ;
    testDomain.OutputNNs(nnFile) ;

    size_t nEp = 100 ;
    std::cout << "Executing " << nEp << " epochs...\n" ;
    testDomain.ExecuteLearning(nEp) ;
    
    std::cout << "Executing learned NN policies from stored file...\n" ;
    char ttFile[buffSize] ;
    sprintf(ttFile,"%s/trajectories_test.txt",fileDir) ;
    char ppFile[buffSize] ;
    sprintf(ppFile,"%s/POIs_test.txt",fileDir) ;
    testDomain.ExecutePolicy(nnFile, ttFile, ppFile, 4, 2, 8) ;
  }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
