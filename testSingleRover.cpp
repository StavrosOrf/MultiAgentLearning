#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "Domains/SingleRover.h"
#include "Utilities/MatrixTypes.h"

int main(){
  std::cout << "Testing SingleRover class in SingleRover.h\n" ;
  
  matrix1d world ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  
  size_t nPOIs = 20 ;
  size_t nSteps = 100 ;
  size_t nPop = 15 ;
  size_t nRuns = 100 ;
  
  for (size_t i = 0; i < nRuns; i++){
    std::cout << "Initialising rover domain " << i << "...\n" ;
    SingleRover testDomain(world, nPOIs, nSteps, nPop) ;

    int buffSize = 100 ;
    char fileDir[buffSize] ;
    sprintf(fileDir,"Results/100_square/Random_worlds/100_epochs/%d",(int)i) ;
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

    size_t nEp = 100 ;
    std::cout << "Executing " << nEp << " epochs...\n" ;
    testDomain.ExecuteLearning(nEp) ;
  }
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
