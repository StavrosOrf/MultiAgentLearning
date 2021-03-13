#include "POMDPPolicy.h"

POMDPPolicy::POMDPPolicy(char * fName){
  // Filename to write to stored in fName
	stringstream fileName ;
  fileName << fName ;
  ifstream policyFile ;
  policyFile.open(fileName.str().c_str(),std::ios::in) ;
  
  // Data format: [actionNum, vector<pBeliefState>]
  string line ;
  while (getline(policyFile,line)){
    stringstream lineStream(line) ;
    string cell ;
  
    size_t elemNum = 0 ;
    vector<double> b ;
    while (getline(lineStream,cell,',')){ // comma delimiter
      if (elemNum == 0)
        actionNums.push_back(atoi(cell.c_str())) ;
      else
        b.push_back(atof(cell.c_str())) ;
      elemNum++ ;
    }
    VectorXd bb ;
    bb.setZero(b.size()) ;
    for (size_t i = 0; i < b.size(); i++)
      bb(i) = b[i] ;
    pMatrix.push_back(bb) ;
  }
}

size_t POMDPPolicy::GetBestAction(VectorXd belief){
  VectorXd res ;
  res.setZero(pMatrix.size()) ;
  double resMax = DBL_MIN ;
  size_t maxInd = -1 ;
  for (size_t i = 0; i < pMatrix.size(); i++){
    res(i) = pMatrix[i].dot(belief) ;
    if (res(i) > resMax){
      resMax = res(i) ;
      maxInd = i ;
    }
  }
  return actionNums[maxInd] ;
}
