#ifndef POMDP_POLICY_H_
#define POMDP_POLICY_H_

#include <string>
#include <fstream>
#include <sstream>
#include <string>
#include <float.h>
#include <vector>
#include <Eigen/Eigen>

using std::string ;
using std::getline ;
using std::stringstream ;
using std::ifstream ;
using std::vector ;
using namespace Eigen ;

class POMDPPolicy{
  public:
    POMDPPolicy(char *) ;
    ~POMDPPolicy(){}
    
    size_t GetBestAction(VectorXd) ;
    
    vector<size_t> GetActionVector(){return actionNums ;}
    vector<VectorXd> GetPolicyMatrix(){return pMatrix ;}
  private:
    vector<size_t> actionNums ;
    vector<VectorXd> pMatrix ;
} ;
#endif // POMDP_POLICY_H_
