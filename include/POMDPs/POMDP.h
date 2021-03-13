#ifndef POMDP_H_
#define POMDP_H_

#include <vector>
#include <Eigen/Eigen>
#include "POMDPEnvironment.h"
#include "POMDPPolicy.h"

using std::vector ;
using namespace Eigen ;

class POMDP{
  public:
    POMDP(char *, char *, VectorXd) ;
    ~POMDP(){
      delete pomdpEnv ;
      delete pomdpPolicy ;
    }
    
    size_t GetBestAction() ;
    void UpdateBelief(size_t, size_t) ;
    
    POMDPEnvironment * GetPOMDPEnvironment(){return pomdpEnv ;}
    POMDPPolicy * GetPOMDPPolicy(){return pomdpPolicy ;}
    
    VectorXd GetBelief(){return belief ;}
  private:
    POMDPEnvironment * pomdpEnv ;
    POMDPPolicy * pomdpPolicy ;
    VectorXd belief ;
} ;
#endif // POMDP_H_
