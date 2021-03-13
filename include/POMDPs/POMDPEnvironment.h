#ifndef POMDP_ENVIRONMENT_H_
#define POMDP_ENVIRONMENT_H_

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <Eigen/Eigen>

using std::string ;
using std::getline ;
using std::stringstream ;
using std::ifstream ;
using std::vector ;
using namespace Eigen ;

class POMDPEnvironment{
  public:
    POMDPEnvironment(char *) ;
    ~POMDPEnvironment(){}
    
    VectorXd UpdateBelief(VectorXd, size_t, size_t) ;
    
    double GetDiscount(){return discount ;}
    string GetValues(){return values ;}
    vector<string> GetStates(){return states ;}
    vector<string> GetActions(){return actions ;}
    vector<string> GetObservations(){return observations ;}
    vector<MatrixXd> GetTransitions(){return T ;}
    vector<MatrixXd> GetObservationProbabilities(){return Z ;}
    vector< vector<MatrixXd> > GetRewards(){return R ;} ;
  private:
    double discount ;
    string values ;
    vector<string> states ;
    vector<string> actions ;
    vector<string> observations ;
    
    vector<MatrixXd> T ; // action, initial state, transitioned state
    vector<MatrixXd> Z ; // action, initial state, observation
    vector< vector<MatrixXd> > R ; // initial state, transitioned state, action, observation
} ;
#endif // POMDP_ENVIRONMENT_H_
