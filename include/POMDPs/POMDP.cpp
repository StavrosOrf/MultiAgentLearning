#include "POMDP.h"

POMDP::POMDP(char * env, char * policy, VectorXd b){
  pomdpEnv = new POMDPEnvironment(env) ;
  pomdpPolicy = new POMDPPolicy(policy) ;
  belief = b ;
}

size_t POMDP::GetBestAction(){
  return pomdpPolicy->GetBestAction(belief) ;
}

void POMDP::UpdateBelief(size_t act, size_t obs){
  VectorXd b = pomdpEnv->UpdateBelief(belief,act,obs) ;
  belief = b ;
}

