#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "Domains/MultiRover.h"

using std::vector ;
using std::string ;
using namespace Eigen ;

int main(){
  std::cout << "***** Multi-rover introspection and assistance request experiments *****\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  world.push_back(0.0) ; 
  world.push_back(100.0) ;
  
  int trialNum = 11 ;
  
  size_t nPOIs = 25 ;
  size_t nSteps = 1000 ;
  size_t nPop = 15 ;
  size_t rovs = 5 ;
  string evalFunc = "D" ;
  size_t nEps = 1000 ;
  
  MultiRover testDomain(world, nSteps, nPop, nPOIs, evalFunc, rovs) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/100_square/Random_worlds/%d_epochs/MultiRover/%s/%d",(int)nEps,evalFunc.c_str(),0) ;
  
  char NNFile[buffSize] ;
  sprintf(NNFile,"%s/NNs.txt",fileDir) ;

  char eeFile[buffSize] ;
  sprintf(eeFile,"%s/results_test_%d.txt",fileDir,trialNum) ;
  char ttFile[buffSize] ;
  sprintf(ttFile,"%s/trajectories_test_%d.txt",fileDir,trialNum) ;
  char ppFile[buffSize] ;
  sprintf(ppFile,"%s/POIs_test_%d.txt",fileDir,trialNum) ;
  char ddFile[buffSize] ;
  sprintf(ddFile,"%s/avgD_test_%d.txt",fileDir,trialNum) ;
  
  testDomain.OutputAverageStepwise(ddFile) ; // store average stepwise D values
  
  char qqFile[buffSize] ;
  sprintf(qqFile,"%s/queries_test_%d.txt",fileDir,trialNum) ;
  char bbFile[buffSize] ;
  sprintf(bbFile,"%s/beliefs_test_%d.txt",fileDir,trialNum) ;
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
  
  testDomain.ExecutePolicies(NNFile, ttFile, ppFile, eeFile, qqFile, bbFile, 8, 2, 16, 0, envFile, polFile, prior) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
