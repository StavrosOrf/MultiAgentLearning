#include "POMDPEnvironment.h"

POMDPEnvironment::POMDPEnvironment(char * fName){
  // WARNING! SERIOUSLY HARD CODED TO EXPECT STRICT ORDERING IN .POMDP FILE
  // Filename to write to stored in fName
	stringstream fileName ;
  fileName << fName ;
  ifstream envFile ;
  envFile.open(fileName.str().c_str(),std::ios::in) ;
  
  // Collect state, action and observation definitions
  string line ;
  while (getline(envFile,line)){
    stringstream lineStream(line) ;
    string cell ;
    
    size_t elemNum = 0 ;
    int tag = -1 ; // 0: discount, 1: values, 2: states, 3: actions, 4: observations
    while (getline(lineStream,cell,' ')){ // space delimiter
      if (elemNum == 0){
        if (cell.compare("discount:") == 0){
          tag = 0 ;
//          std::cout << "Reading in discount parameter...\n" ;
        }
        else if (cell.compare("values:") == 0){
          tag = 1 ;
//          std::cout << "Reading in value parameters...\n" ;
        }
        else if (cell.compare("states:") == 0){
          tag = 2 ;
//          std::cout << "Reading in states...\n" ;
        }
        else if (cell.compare("actions:") == 0){
          tag = 3 ;
//          std::cout << "Reading in actions...\n" ;
        }
        else if (cell.compare("observations:") == 0){
          tag = 4 ;
//          std::cout << "Reading in observations...\n" ;
        }
        else
          break ; // ignore all other annotated lines
      }
      else if (tag == 0)
        discount = atof(cell.c_str()) ;
      else if (tag == 1)
        values = cell ;
      else if (tag == 2)
        states.push_back(cell) ;
      else if (tag == 3)
        actions.push_back(cell) ;
      else if (tag == 4)
        observations.push_back(cell) ;
      elemNum++ ;
    }
  }
  envFile.close() ;
  
  // Collect transition functions and observation probabilities
  size_t lineNum = 0 ;
  size_t matrixDataRow = -1 ;
  
  MatrixXd tContainer ;
  tContainer.setZero(states.size(),states.size()) ;
  vector<MatrixXd> tt(actions.size(),tContainer) ;
  
  MatrixXd zContainer ;
  zContainer.setZero(states.size(),observations.size()) ;
  vector<MatrixXd> zz(actions.size(),zContainer) ;
  
  envFile.open(fileName.str().c_str(),std::ios::in) ;
  int tag = -1 ; // 5: T, 6: O
  int aInd = -1 ;
  while (getline(envFile,line)){
    stringstream lineStream(line) ;
    string cell ;
    
    size_t elemNum = 0 ;
    while (getline(lineStream,cell,' ')){ // space delimiter
      if (aInd == -1){ // collect information on where to store matrix data
        if (elemNum == 0){
          if (cell.compare("T:") == 0){
            tag = 5 ;
            matrixDataRow = lineNum ;
//            std::cout << "\nNew matrix data begins on line: " << matrixDataRow << "\n" ;
          }
          else if (cell.compare("O:") == 0){
            tag = 6 ;
            matrixDataRow = lineNum ;
//            std::cout << "\nNew matrix data begins on line: " << matrixDataRow << "\n" ;
          }
        }
        else if (elemNum == 1 && tag > 0){
          for (size_t i = 0; i < actions.size(); i++){
            if (cell.compare(actions[i]) == 0){
              aInd = (int)i ;
              break ;
            }
          }
        }
      }
      else if (tag == 5){
        size_t i = lineNum - matrixDataRow - 1 ;
        tt[aInd](i,elemNum) = atof(cell.c_str()) ;
//        std::cout << cell << ":" ;
      }
      else if (tag == 6){
        size_t i = lineNum - matrixDataRow - 1 ;
        zz[aInd](i,elemNum) = atof(cell.c_str()) ;
//        std::cout << cell << ":" ;
      }
      elemNum++ ;
    }
    
    // tag = {5,6} if at end of matrix, reset tag and aInd to -1
    if (tag > 0 && lineNum-matrixDataRow >= states.size()){
      tag = -1 ;
      aInd = -1 ;
    }
    lineNum++ ;
  }
  // Write to member variables
  T = tt ;
  Z = zz ;
  
  envFile.close() ;
  
  // Loop through for rewards
  MatrixXd rContainer(actions.size(),observations.size()) ;
  vector<MatrixXd> r(states.size(),rContainer) ;
  vector< vector<MatrixXd> > rr(states.size(),r) ;
  
  envFile.open(fileName.str().c_str(),std::ios::in) ;
  while (getline(envFile,line)){
    stringstream lineStream(line) ;
    string cell ;
    
    vector<size_t> s0 ;
    vector<size_t> s1 ;
    vector<size_t> a ;
    vector<size_t> o ;
    
    size_t elemNum = 0 ;
    while (getline(lineStream,cell,':')){ // colon delimiter
      if (elemNum == 0 && cell.compare("R") != 0)
        break ; // skip line
      else {
        if (elemNum == 1){
          // Remove whitespace
          string::iterator newEnd = std::remove(cell.begin(), cell.end(), ' ');
          cell.assign(cell.begin(),newEnd) ;
//          std::cout << "First element: [" << cell << "]\n" ;
          if (cell.compare("*") == 0)
            for (size_t i = 0; i < actions.size(); i++)
              a.push_back(i) ;
          else {
            for (size_t i = 0; i < actions.size(); i++){
              if (cell.compare(actions[i]) == 0){
                a.push_back(i) ;
                break ;
              }
            }
          }
        }
        else if (elemNum == 2){
          // Remove whitespace
          string::iterator newEnd = std::remove(cell.begin(), cell.end(), ' ');
          cell.assign(cell.begin(),newEnd) ;
//          std::cout << "Second element: [" << cell << "]\n" ;
          if (cell.compare("*") == 0)
            for (size_t i = 0; i < states.size(); i++)
              s0.push_back(i) ;
          else {
            for (size_t i = 0; i < states.size(); i++){
              if (cell.compare(states[i]) == 0){
                s0.push_back(i) ;
                break ;
              }
            }
          }
        }
        else if (elemNum == 3){
          // Remove whitespace
          string::iterator newEnd = std::remove(cell.begin(), cell.end(), ' ');
          cell.assign(cell.begin(),newEnd) ;
//          std::cout << "Third element: [" << cell << "]\n" ;
          if (cell.compare("*") == 0)
            for (size_t i = 0; i < states.size(); i++)
              s1.push_back(i) ;
          else {
            for (size_t i = 0; i < states.size(); i++){
              if (cell.compare(states[i]) == 0){
                s1.push_back(i) ;
                break ;
              }
            }
          }
        }
        else if (elemNum == 4){
//          std::cout << "Fourth element: [" << cell << "]\n" ;
          stringstream ss(cell) ;
          string sCell ;
          size_t eNum = 0 ;
          while (getline(ss,sCell,' ')){
//            std::cout << "eNum: " << eNum << ", sCell: [" << sCell << "]\n" ;
            if (eNum == 0){
//              std::cout << "sCell: [" << sCell << "]\n" ;
              if (sCell.compare("*") == 0){
                eNum++ ;
                for (size_t i = 0; i < observations.size(); i++)
                  o.push_back(i) ;
              }
              else {
                for (size_t i = 0; i < observations.size(); i++){
                  if (sCell.compare(observations[i]) == 0){
                    o.push_back(i) ;
                    eNum++ ;
                    break ;
                  }
                }
              }
            }
            else {
//              std::cout << "s0.size(): " << s0.size() << ", s1.size(): " << s1.size() << ", a.size(): " << a.size() << ", o.size(): " << o.size() << "\n" ;
//              std::cout << "s0[0]: " << s0[0] << ", s1[0]: " << s1[0] << ", s1[1]: " << s1[1] << ", a[0]: " << a[0] << ", o[0]: " << o[0] << ", sCell: " << atof(sCell.c_str()) << "\n" ;
              for (size_t i = 0; i < s0.size(); i++)
                for (size_t j = 0; j < s1.size(); j++)
                  for (size_t k = 0; k < a.size(); k++)
                    for (size_t l = 0; l < o.size(); l++)
                      rr[s0[i]][s1[j]](a[k],o[l]) = atof(sCell.c_str()) ;
              eNum++ ;
            }
          }
        }
      }
      elemNum++ ;
    }
  }
  R = rr ; // write to reward member variable
  
//  for (size_t i = 0; i < rr.size(); i++){
//    for (size_t j = 0; j < rr[i].size(); j++){
//      for (int k = 0; k < rr[i][j].rows(); k++){
//        for (int l = 0; l < rr[i][j].cols(); l++){
//          std::cout << rr[i][j](k,l) << " " ;
//        }
//        std::cout << "\n" ;
//      }
//      std::cout << "\n" ;
//    }
//    std::cout << "\n" ;
//  }

  
  envFile.close() ;
}

VectorXd POMDPEnvironment::UpdateBelief(VectorXd b, size_t aInd, size_t oInd){
  VectorXd belief ;
  belief.setZero(states.size()) ;
  for (size_t i = 0; i < states.size(); i++){
    double pOPrime = Z[aInd](i,oInd) ;
    double summation = 0.0 ;
    for (size_t j = 0; j < states.size(); j++){
      double pSPrime = T[aInd](j,i) ;
      double bS = b[j] ;
      summation += pSPrime*bS ;
    }
    belief(i) = pOPrime * summation ;
  }
  
  // Normalise belief
  double total = belief.sum() ;
  belief /= total ;
  
  return belief ;
}
