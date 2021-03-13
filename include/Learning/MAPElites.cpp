#include "MAPElites.h"

// Constructor requires matrix of behaviour bin limits and size specifications of NN controllers. Initialises the vector storing the best performance evaluations and allocates memory on the heap to store the controllers.
MAPElites::MAPElites(MatrixXd bins, size_t nIn, size_t nOut, size_t nHid): binLimits(bins){
  bDim = bins.rows() ;
  numBins.setZero(bDim,1) ;
  size_t totalBins = 1 ;
  int k = 0 ;
  for (int i = 0; i < bDim; i++){
    double maxLim = binLimits(i,0) ;
    bool endFound = false ;
    for (int j = 1; j < binLimits.cols(); j++){
      if (binLimits(i,j) < maxLim && !endFound){ // requires all valid bin limits to be increasing along the row
        endFound = true ;
        totalBins += j ;
        numBins(k) = j ;
        k++ ;
        break ;
      }
      else
        maxLim = binLimits(i,j) ;
    }
    if (!endFound){
      totalBins *= binLimits.cols() ;
      numBins(k) = binLimits.cols() ;
      k++ ;
    }
  }

  for (size_t i = 0; i < totalBins; i++){
    behaviourFilled.push_back(false) ;
    performanceLog.push_back(0.0) ;
    behaviourMap.push_back(new NeuralNet(nIn,nOut,nHid)) ; // initialise behaviour map storage space
  }
  
  // Reverse cumulative product, used for computing 1D vector index from nD indices
  cProd.setOnes(bDim,1) ;
  for (int i = 1; i < bDim; i++){
    int j = bDim-1-i ;
    cProd(j) *= numBins(j+1) ;
    if (j > 0)
      cProd(j-1) = cProd(j) ;
  }
}

// Destructor releases all heap memory allocated to storing the NN controllers
MAPElites::~MAPElites(){
  for (size_t i = 0; i < behaviourMap.size(); i++){
    delete(behaviourMap[i]) ;
    behaviourMap[i] = 0 ;
  }
}

// Outputs the best performance evaluation stored for an input behaviour vector
double MAPElites::GetPerformance(VectorXd behaviour){
  if (behaviour.size() != bDim){
    std::cout << "Error: input behaviour vector has the wrong number of elements!\n" ;
    return 0.0 ;
  }
  return performanceLog[GetIndex(behaviour)] ;
}

// Outputs boolean describing whether a behaviour has been visited before or not
bool MAPElites::IsVisited(VectorXd behaviour){
  return behaviourFilled[GetIndex(behaviour)] ;
}

// Outputs the NN controller associated with the best performance for an input behaviour vector 
NeuralNet * MAPElites::GetNeuralNet(VectorXd behaviour){
  if (behaviour.size() != bDim){
    std::cout << "Error: input behaviour vector has the wrong number of elements!\n" ;
    return 0 ;
  }
  return behaviourMap[GetIndex(behaviour)] ;
}

NeuralNet * MAPElites::GetNeuralNet(size_t ind){
  if (ind > behaviourMap.size()){
    std::cout << "Error: behaviour map index out of bounds!\n" ;
    return 0 ;
  }
  return behaviourMap[ind] ;
}

// Considers the input NN given the corresponding behaviour vector and performance evaluation. If the stated performance is superior to the stored NN, it will replace the stored NN.  
void MAPElites::UpdateMap(NeuralNet * NN, VectorXd bMap, double eval){
  double pMap = GetPerformance(bMap) ;
  if (eval > pMap || !IsVisited(bMap)){ // enter behaviour into map if better than previous or not visited before
    MatrixXd A = NN->GetWeightsA() ;
    MatrixXd B = NN->GetWeightsB() ;
    
    size_t n = GetIndex(bMap) ;
    behaviourMap[n]->SetWeights(A,B) ;
    performanceLog[n] = eval ;
    
    if (!IsVisited(bMap))
      behaviourFilled[GetIndex(bMap)] = true ;
  }
}

// Returns the 1D index of an nD behaviour vector
size_t MAPElites::GetIndex(VectorXd behaviour){
  VectorXi bMap ;
  bMap.setZero(bDim,1) ;
  int k = 0 ;
  for (int i = 0; i < behaviour.size(); i++){
    bool binFound = false ;
    bMap(k) = 0 ;
    for (int j = 0; j < numBins(i); j++){
      double curLimit ;
      double eps = 0.001 ; // manage precision issues
      if (j == numBins(i)-1)
        curLimit = binLimits(i,j) + eps ;
      else
        curLimit = (binLimits(i,j+1)+binLimits(i,j))/2.0 ;
      
      if (behaviour(i) <= curLimit){
        bMap(k) = j ;
        k++ ;
        binFound = true ;
        break ;
      }
    }
    if (!binFound){
      std::cout << "Error: behaviour vector element out of bounds! [" ;
      for (int j = 0; j < behaviour.size()-1; j++)
        std::cout << behaviour(j) << "," ;
      std::cout << behaviour(behaviour.size()-1) << "]\n" ;
      return 0 ;
    }
  }
  size_t n = bMap.dot(cProd) ; // may not like size_t conversion
  return n ;
}

// Returns the nD vector corresponding to the 1D behaviour index
VectorXd MAPElites::GetBehaviour(size_t n){
  VectorXd bMap ;
  bMap.setZero(bDim,1) ;
  for (int i = 0; i < bMap.size(); i++){
    bMap(i) = binLimits(i,n/cProd(i)) ;
    n = fmod(n,cProd(i)) ;
  }
  return bMap ;
}

void MAPElites::WriteBPMapBinary(char * fName){
  // Filename to write behaviour performance map
	std::stringstream fileName ;
  fileName << fName ;
  std::ofstream bpMapFile ;
  bpMapFile.open(fileName.str().c_str(),std::ios::out | std::ios::binary) ;
  
  // Loop through all behaviours
  for (size_t i = 0; i < behaviourMap.size(); i++){
    MatrixXd NNA = behaviourMap[i]->GetWeightsA() ;
    for (int j = 0; j < NNA.rows(); j++){
      char * pNNARow = reinterpret_cast<char *>(&NNA(j,0)) ;
      size_t bytes = NNA.cols() * sizeof(NNA(j,0)) ;
      bpMapFile.write(pNNARow, bytes) ;
    }
    
    MatrixXd NNB = behaviourMap[i]->GetWeightsB() ;
    for (int j = 0; j < NNB.rows(); j++){
      char * pNNBRow = reinterpret_cast<char *>(&NNB(j,0)) ;
      size_t bytes = NNB.cols() * sizeof(NNB(j,0)) ;
      bpMapFile.write(pNNBRow, bytes) ;
    }
  }
}

void MAPElites::ReadBPMapBinary(char * fName){
  // Filename to read behaviour performance map
	std::stringstream fileName ;
  fileName << fName ;
  std::ifstream bpMapFile ;
  bpMapFile.open(fileName.str().c_str(),std::ios::in | std::ios::binary) ;
  
  // Loop through all behaviours
  for (size_t i = 0; i < behaviourMap.size(); i++){
    MatrixXd NNA = behaviourMap[i]->GetWeightsA() ;
    for (int j = 0; j < NNA.rows(); j++){
      char * pNNARow = reinterpret_cast<char *>(&NNA(j,0)) ;
      size_t bytes = NNA.cols() * sizeof(NNA(j,0)) ;
      bpMapFile.read(pNNARow, bytes) ;
    }
    
    MatrixXd NNB = behaviourMap[i]->GetWeightsB() ;
    for (int j = 0; j < NNB.rows(); j++){
      char * pNNBRow = reinterpret_cast<char *>(&NNB(j,0)) ;
      size_t bytes = NNB.cols() * sizeof(NNB(j,0)) ;
      bpMapFile.read(pNNBRow, bytes) ;
    }
    
    behaviourMap[i]->SetWeights(NNA, NNB) ;
  }
}

void MAPElites::WritePerformanceBinary(char * fName){
  // Filename to write performance log
	std::stringstream fileName ;
  fileName << fName ;
  std::ofstream performanceFile ;
  performanceFile.open(fileName.str().c_str(),std::ios::out | std::ios::binary) ;
  
  // Write all logged performance values
  char * pPLog = reinterpret_cast<char *>(&performanceLog[0]) ;
  size_t bytes = performanceLog.size() * sizeof(performanceLog[0]) ;
  performanceFile.write(pPLog, bytes) ;
}

void MAPElites::ReadPerformanceBinary(char * fName){
  // Filename to read performance log
	std::stringstream fileName ;
  fileName << fName ;
  std::ifstream performanceFile ;
  performanceFile.open(fileName.str().c_str(),std::ios::in | std::ios::binary) ;
  
  // Read all logged performance values
  vector<double> pLog = performanceLog ;
  char * pPLog = reinterpret_cast<char *>(&pLog[0]) ;
  size_t bytes = pLog.size() * sizeof(pLog[0]) ;
  performanceFile.read(pPLog, bytes) ;
  
  performanceLog = pLog ;
}

void MAPElites::WriteVisitedBinary(char * fName){
  // Filename to write visitation log
	std::stringstream fileName ;
  fileName << fName ;
  std::ofstream visitedFile ;
  visitedFile.open(fileName.str().c_str(),std::ios::out | std::ios::binary) ;
  
  vector<size_t> bFilled ;
  for (size_t i = 0; i < behaviourFilled.size(); i++){
    if (behaviourFilled[i])
      bFilled.push_back(1) ;
    else
      bFilled.push_back(0) ;
  }
  
  // Write all logged visitation values
  char * pFLog = reinterpret_cast<char *>(&bFilled[0]) ;
  size_t bytes = bFilled.size() * sizeof(bFilled[0]) ;
  visitedFile.write(pFLog, bytes) ;
}

void MAPElites::ReadVisitedBinary(char * fName){
  // Filename to read visitation log
	std::stringstream fileName ;
  fileName << fName ;
  std::ifstream visitedFile ;
  visitedFile.open(fileName.str().c_str(),std::ios::in | std::ios::binary) ;
  
  vector<size_t> bFilled(behaviourFilled.size()) ;
  
  // Read all logged visitation values
  char * pFLog = reinterpret_cast<char *>(&bFilled[0]) ;
  size_t bytes = bFilled.size() * sizeof(bFilled[0]) ;
  visitedFile.read(pFLog, bytes) ;
  
  
  for (size_t i = 0; i < behaviourFilled.size(); i++){
    if (bFilled[i] == 1)
      behaviourFilled[i] = true ;
    else
      behaviourFilled[i] = false ;
  }
}
