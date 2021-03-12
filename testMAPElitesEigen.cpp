#include <iostream>

#include "Learning/MAPElites.h"
#include "Learning/NeuralNet.h"
#include "Utilities/Utilities.h"

using easymath::rand_interval ;
using easymath::sum ;

int main(){
  std::cout << "Testing MAPElites class in MAPElites.h\n" ;
  
  size_t nIn = 4 ;
  size_t nOut = 2 ;
  size_t nHid = 5 ;
  size_t nBinDim = 4 ;
  size_t nBinSize = 5 ;
  
  MatrixXd bins(nBinDim,nBinSize) ;
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
  
  std::cout << "Creating MAPElites object...\n" ;
  MAPElites testMAP(bins,nIn,nOut,nHid) ;
  
  std::cout << "Selecting behaviour...[" ;
  VectorXd bMap(4) ;
  for (int i = 0; i < bMap.size(); i++){
    bMap(i) = rand_interval(0.0,1.0) ;
    std::cout << bMap(i) << "," ;
  }
  std::cout << "]\n" ;
  size_t ind = testMAP.GetIndex(bMap) ;
  std::cout << "Index: " << ind << std::endl ;
  
  std::cout << "Current NN at selected behaviour has weights:\n" ;
  NeuralNet * curNN = testMAP.GetNeuralNet(bMap) ;
  MatrixXd A0 = curNN->GetWeightsA() ;
  MatrixXd B0 = curNN->GetWeightsB() ;
  
  for (int i = 0; i < A0.rows(); i++){
    for (int j = 0; j < A0.cols(); j++){
      std::cout << A0(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B0.rows(); i++){
    for (int j = 0; j < B0.cols(); j++){
      std::cout << B0(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Replacing a neural net in the behaviour performance map...\n" ;
  NeuralNet * newNN = new NeuralNet(nIn,nOut,nHid) ;
  double eval = 10.0 ;
  testMAP.UpdateMap(newNN,bMap,eval) ;
  MatrixXd A1 = newNN->GetWeightsA() ;
  MatrixXd B1 = newNN->GetWeightsB() ;
  
  std::cout << "Replacement NN has weights:\n" ;
  for (int i = 0; i < A1.rows(); i++){
    for (int j = 0; j < A1.cols(); j++){
      std::cout << A1(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B1.rows(); i++){
    for (int j = 0; j < B1.cols(); j++){
      std::cout << B1(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Confirming NN stored in behaviour map has been updated...\n" ;
  curNN = testMAP.GetNeuralNet(bMap) ;
  MatrixXd A2 = curNN->GetWeightsA() ;
  MatrixXd B2 = curNN->GetWeightsB() ;
  
  std::cout << "NN at selected behaviour has weights:\n" ;
  for (int i = 0; i < A2.rows(); i++){
    for (int j = 0; j < A2.cols(); j++){
      std::cout << A2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B2.rows(); i++){
    for (int j = 0; j < B2.cols(); j++){
      std::cout << B2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Considering a neural net with lower performance...\n" ;
  NeuralNet * anotherNN = new NeuralNet(nIn,nOut,nHid) ;
  eval = 5.0 ;
  testMAP.UpdateMap(anotherNN,bMap,eval) ;
  MatrixXd A3 = anotherNN->GetWeightsA() ;
  MatrixXd B3 = anotherNN->GetWeightsB() ;
  
  std::cout << "Considered NN has weights:\n" ;
  for (int i = 0; i < A3.rows(); i++){
    for (int j = 0; j < A3.cols(); j++){
      std::cout << A3(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B3.rows(); i++){
    for (int j = 0; j < B3.cols(); j++){
      std::cout << B3(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Confirming NN stored in behaviour map has not been overwritten...\n" ;
  curNN = testMAP.GetNeuralNet(bMap) ;
  A2 = curNN->GetWeightsA() ;
  B2 = curNN->GetWeightsB() ;
  
  std::cout << "NN at selected behaviour has weights:\n" ;
  for (int i = 0; i < A2.rows(); i++){
    for (int j = 0; j < A2.cols(); j++){
      std::cout << A2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B2.rows(); i++){
    for (int j = 0; j < B2.cols(); j++){
      std::cout << B2(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Writing current behaviour performance map to binary file...\n" ;
  testMAP.WriteBPMapBinary("testBPWrite") ;
  
  std::cout << "Writing performance log to binary file...\n" ;
  testMAP.WritePerformanceBinary("testPLogWrite") ;
  
  std::cout << "Writing visitation booleans to binary file...\n" ;
  testMAP.WriteVisitedBinary("testVisitWrite") ;
  
  std::cout << "Creating new MAPElites object...\n" ;
  MAPElites newMAP(bins,nIn,nOut,nHid) ;
  
  std::cout << "Initial checksums...\n" ;
  double pSum0 = sum(testMAP.GetPerformanceLog()) ;
  double pSum1 = sum(newMAP.GetPerformanceLog()) ;
  std::cout << "testMAP: " << pSum0 << ", newMAP: " << pSum1 << std::endl ;
  
  std::cout << "Initial visitations checksum...\n" ;
  int vSum0 = 0 ;
  int vSum1 = 0 ;
  for (size_t i = 0; i < testMAP.GetFilledLog().size(); i++){
    if (testMAP.GetFilledLog()[i])
      vSum0++ ;
    if (newMAP.GetFilledLog()[i])
      vSum1++ ;
  }
  std::cout << "testMAP: " << vSum0 << ", newMAP: " << vSum1 << std::endl ;
  
  std::cout << "Reading in binary file of behaviour performance map...\n" ;
  newMAP.ReadBPMapBinary("testBPWrite") ;
  
  std::cout << "Reading in binary file of performance log...\n" ;
  newMAP.ReadPerformanceBinary("testPLogWrite") ;
  
  std::cout << "Reading in binary file of visitation booleans...\n" ;
  newMAP.ReadVisitedBinary("testVisitWrite") ;
  
  std::cout << "Confirming NN stored in behaviour map matches saved map...\n" ;
  curNN = newMAP.GetNeuralNet(bMap) ;
  A3 = curNN->GetWeightsA() ;
  B3 = curNN->GetWeightsB() ;
  
  std::cout << "NN at selected behaviour has weights:\n" ;
  for (int i = 0; i < A3.rows(); i++){
    for (int j = 0; j < A3.cols(); j++){
      std::cout << A3(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  for (int i = 0; i < B3.rows(); i++){
    for (int j = 0; j < B3.cols(); j++){
      std::cout << B3(i,j) << "," ;
    }
    std::cout << std::endl ;
  }
  
  std::cout << "Confirming stored performances match via checksum...\n" ;
  pSum0 = sum(testMAP.GetPerformanceLog()) ;
  pSum1 = sum(newMAP.GetPerformanceLog()) ;
  std::cout << "testMAP: " << pSum0 << ", newMAP: " << pSum1 << std::endl ;
  
  std::cout << "Confirming stored visitations match via checksum...\n" ;
  vSum0 = 0 ;
  vSum1 = 0 ;
  for (size_t i = 0; i < testMAP.GetFilledLog().size(); i++){
    if (testMAP.GetFilledLog()[i])
      vSum0++ ;
    if (newMAP.GetFilledLog()[i])
      vSum1++ ;
  }
  std::cout << "testMAP: " << vSum0 << ", newMAP: " << vSum1 << std::endl ;
  
  delete(newNN) ;
  delete(anotherNN) ;
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
