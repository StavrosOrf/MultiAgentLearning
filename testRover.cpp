#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <Eigen/Eigen>
#include <vector>

#include "Domains/Target.h"
#include "Agents/Rover.h"
#include "Utilities/Utilities.h"

using std::string ;
using std::vector ;
using easymath::rand_interval ;
using namespace Eigen ;

// Wrapper for writing epoch evaluations to specified files
void OutputPerformance(char * A, vector<double> evals){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  std::ofstream evalFile ;
  evalFile.open(fileName.str().c_str(),std::ios::app) ;
  
  for (size_t i = 0; i < evals.size(); i++)
    evalFile << evals[i] << "," ;
  
  evalFile << "\n" ;
  evalFile.close() ;
}

// Wrapper for writing final trajectories to specified files
void OutputTrajectories(char * A, vector<Vector2d> xy){
	// Filename to write trajectories to stored in A
	std::stringstream tfileName ;
  tfileName << A ;
  std::ofstream trajFile ;
  trajFile.open(tfileName.str().c_str(),std::ios::app) ;
  
  for (size_t i = 0; i < xy.size(); i++)
    trajFile << xy[i](0) << "," << xy[i](1) << "," ;
  
  trajFile << "\n" ;
  trajFile.close() ;
}

void OutputPOIs(char * A, vector<Target> POIs){
  // Filename to write POIs to stored in B
	std::stringstream pfileName ;
  pfileName << A ;
  std::ofstream POIFile ;
  POIFile.open(pfileName.str().c_str(),std::ios::app) ;
  
  for (size_t i = 0; i < POIs.size(); i++)
    POIFile << POIs[i].GetLocation()(0) << "," << POIs[i].GetLocation()(1) << "," << POIs[i].GetValue() << "\n" ;
  
  POIFile.close() ;
}

int main(){
  std::cout << "Testing Rover class in Rover.h\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  
  size_t nPOIs = 10 ;
  size_t nSteps = 30 ;
  size_t nPop = 15 ;
  size_t nEp = 100 ;
  
  string evalFunc = "G" ;
  
  Rover testRover(nSteps, nPop, evalFunc) ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"Results/30_square/Static_world/100_epochs/Rover/%d",0) ;
  char mkdir[buffSize] ;
  sprintf(mkdir,"mkdir -p %s",fileDir) ;
  system(mkdir) ;
  char eFile[buffSize] ;
  sprintf(eFile,"%s/results.txt",fileDir) ;
  std::cout << eFile << std::endl ;

  char tFile[buffSize] ;
  sprintf(tFile,"%s/trajectories.txt",fileDir) ;
  char pFile[buffSize] ;
  sprintf(pFile,"%s/POIs.txt",fileDir) ;
  
  // Initial XY location and heading in global frame (restricted to within inner region of 9 grid)
  vector<Target> POIs ;
  Vector2d initialXY ;
  double initialPsi ;
  
  for (size_t i = 0; i < nEp; i++){
    std::cout << "Episode " << i << "..." ;
    
    // NN evolution
    if (i == 0)
      testRover.EvolvePolicies(true);
    else
      testRover.EvolvePolicies() ;
    
    testRover.ResetEpochEvals() ;
    
    if (i == 0){
      double rangeX = world[1] - world[0] ;
      double rangeY = world[3] - world[2] ;
      
      initialXY(0) = rand_interval(world[0]+rangeX/3.0,world[1]-rangeX/3.0) ;
      initialXY(1) = rand_interval(world[2]+rangeY/3.0,world[3]-rangeX/3.0) ;
      initialPsi = rand_interval(-PI,PI) ;
      
      // POI locations and values in global frame (restricted to within outer regions of 9 grid)
      for (size_t p = 0; p < nPOIs; p++){
        Vector2d xy ;
        double x, y ;
        bool accept = false ;
        while (!accept){
          x = rand_interval(world[0],world[1]) ;
          y = rand_interval(world[2],world[3]) ;
          if (x > world[0]+rangeX/3.0 && x < world[1]-rangeX/3.0 && y > world[2]+rangeY/3.0 && y < world[3]-rangeX/3.0) {}
          else accept = true ;
        }
        xy(0) = x ; // x location
        xy(1) = y ; // y location
        double v = rand_interval(1,10) ; // value
        POIs.push_back(Target(xy,v)) ;
      }
    }
    
    double maxEval = 0.0 ;
    for (size_t n = 0; n < nPop*2; n++){ // loop over NN control policies
      testRover.InitialiseNewLearningEpoch(POIs, initialXY, initialPsi) ;
      vector<Vector2d> jointState ;
      jointState.push_back(initialXY) ;
        
      if (i == nEp-1)
        OutputTrajectories(tFile, jointState) ;
      
      for (size_t t = 0; t < nSteps; t++){ // loop over timesteps
        Vector2d xy = testRover.ExecuteNNControlPolicy(n,jointState) ;
        jointState.clear() ;
        jointState.push_back(xy) ;
      
        // Compute observations
        for (size_t j = 0; j < nPOIs; j++)
          for (size_t k = 0; k < jointState.size(); k++)
            POIs[j].ObserveTarget(jointState[k]) ;
        
        if (i == nEp-1)
          OutputTrajectories(tFile, jointState) ;
      }
      
      // Evaluate G for current NN
      double eval = 0.0 ;
      for (size_t j = 0; j < nPOIs; j++){
        eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
        POIs[j].ResetTarget() ;
      }
      
      if (eval > maxEval)
        maxEval = eval ;
      
      testRover.SetEpochPerformance(eval, n) ;
    }
    
    OutputPerformance(eFile,testRover.GetEpochEvals()) ; // write epoch evaluations to file
    std::cout << "max achieved value: " << maxEval << "...\n" ;
  }
  
  OutputPOIs(pFile,POIs) ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
