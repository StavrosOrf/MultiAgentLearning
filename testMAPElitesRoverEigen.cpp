#include <iostream>
#include <Eigen/Eigen>
#include "Domains/MAPElitesRover.h"
#include "Learning/NeuralNet.h"

using namespace Eigen ;

int main(){
  std::cout << "Testing MAPElitesRover class in MAPElitesRover.h\n" ;
  
  vector<double> world ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  world.push_back(0.0) ; 
  world.push_back(30.0) ;
  
  MatrixXd bins ;
  bins.setZero(4,5) ;
  for (int i = 0; i < bins.rows(); i++){
    double lim = 0.0 ;
    for (int j = 0; j < bins.cols(); j++){
      bins(i,j) = lim ;
      lim += 1.0/((double)bins.cols()-1) ;
    }
  }
  std::cout << "Behaviour bins:\n" ;
  for (int i = 0; i < bins.rows(); i++){
    for (int j = 0; j < bins.cols(); j++){
      std::cout << bins(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
//  int possibleBins = 0 ;
//  for (size_t i = 0; i < bins[0].size(); i++){
//    double binSumI = bins[0][i] ;
//    for (size_t j = 0; j < bins[1].size(); j++){
//      double binSumJ = binSumI + bins[1][j] ;
//      for (size_t k = 0; k < bins[2].size(); k++){
//        double binSumK = binSumJ + bins[2][k] ;
//        for (size_t l = 0; l < bins[3].size(); l++){
//          double binSumL = binSumK + bins[3][l] ;
//          if (binSumL >= 0.5 && binSumL <= 1.25){
//            std::cout << "[" << bins[0][i] << "," << bins[1][j] << "," << bins[2][k] << "," << bins[3][l] << "]\n" ;
//            possibleBins++ ;
//          }
//        }
//      }
//    }
//  }
//  std::cout << "Percentage of possible bins: " << ((double)possibleBins*100.0)/625.0 << std::endl ;
  
  size_t nPOIs = 20 ;
  size_t nSteps = 100 ;
  size_t n_init = 1000000 ;
  size_t n_evo = 2000000 ;
  size_t nIn = 4 ;
  size_t nOut = 2 ;
  size_t nHid = nIn*2 ;
  
  size_t n_runs = 10 ;
  
  for (size_t t = 8; t < n_runs; t++){
    std::cout << "RUN "<< t << ": Creating MAPElitesRover object...\n" ;
    MAPElitesRover testMAPRover(world, nPOIs, nSteps, bins) ;
    
    std::cout << "Initialising behaviour performance map...\n" ;
    testMAPRover.InitialiseMap(n_init) ;
    
    std::cout << "Executing MAPElites evolution loop...\n" ;
    testMAPRover.EvolveMap(n_evo) ;
    
    std::cout << "Computing MAPElites statistics...\n" ;
    double bpFilled = testMAPRover.PercentageFilled() ;
    std::cout << "Percentage of behaviours filled: " << bpFilled*100.0 << "%\n" ;
    NeuralNet * bestNN = new NeuralNet(nIn,nOut,nHid) ;
    VectorXd bVec0 ;
    bVec0.setZero(bins.rows(),1) ;
    double maxEval = testMAPRover.BestPerformance(bestNN, bVec0) ;
    std::cout << "Best behaviour [" ;
    for (int i = 0; i < bVec0.size()-1; i++)
      std::cout << bVec0(i) << "," ;
    std::cout << bVec0(bVec0.size()-1) << "]; Evaluation: " << maxEval << std::endl ;
    
    int buffSize = 100 ;
    char fileDir[buffSize] ;
    sprintf(fileDir,"Results/MAPElites/%d_square/Obs_behaviour/Static_world/init_%d_evo_%d/Eigen/%d",(int)world[1],(int)n_init,(int)n_evo,(int)t) ;
    char mkdir[buffSize] ;
    sprintf(mkdir,"mkdir -p %s",fileDir) ;
    system(mkdir) ;
    char tFile[buffSize] ;
    sprintf(tFile,"%s/trajectories.txt",fileDir) ;
    char poiFile[buffSize] ;
    sprintf(poiFile,"%s/POIs.txt",fileDir) ;
    testMAPRover.OutputTrajectories(tFile,poiFile) ;
    char bFile[buffSize] ;
    char pFile[buffSize] ;
    char vFile[buffSize] ;
    sprintf(bFile,"%s/BMap",fileDir) ;
    sprintf(pFile,"%s/PMap",fileDir) ;
    sprintf(vFile,"%s/VMap",fileDir) ;
    testMAPRover.WriteToBinary(bFile,pFile,vFile) ;  
    
    std::cout << "Executing best NN in simulation...\n" ;
    VectorXd bVec ;
    bVec.setZero(bins.rows(),1) ;
    double eval = testMAPRover.SimulateController(bestNN, bVec, true) ;
    std::cout << "Behaviour:[" ;
    for (int i = 0; i < bVec.size()-1; i++)
      std::cout << bVec(i) << "," ;
    std::cout << bVec(bVec.size()-1) << "]; Evaluation: " << eval << std::endl ;
  
    delete(bestNN) ;
  }
  
  std::cout << "Test complete!\n" ;
  return 0 ;
}
