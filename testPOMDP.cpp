#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

#include "POMDPs/POMDP.h"

using std::vector ;
using std::string ;

int main(){
  std::cout << "Testing POMDP class in POMDP.h\n" ;
  
  int buffSize = 100 ;
  char fileDir[buffSize] ;
  sprintf(fileDir,"../include/POMDPs") ;
  char envFile[buffSize] ;
  sprintf(envFile,"%s/userLevel01.pomdp",fileDir) ;
  char polFile[buffSize] ;
  sprintf(polFile,"%s/test_edited.policy",fileDir) ;
  
  std::cout << "Reading POMDP environment from: " << envFile << std::endl ;
  std::cout << "Reading POMDP policy from: " << polFile << std::endl ;
  
  std::cout << "Creating POMDP object...\n" ;
  
  VectorXd prior ;
  prior.setZero(2) ;
  prior(0) = 0.5 ;
  prior(1) = 0.5 ;
  
  POMDP testPOMDP(envFile,polFile,prior) ;
  
  POMDPEnvironment * env = testPOMDP.GetPOMDPEnvironment() ;
  POMDPPolicy * pol = testPOMDP.GetPOMDPPolicy() ;
  
  std::cout << "\nPOMDP environment properties...\n" ;
  std::cout << "discount: " << env->GetDiscount() << std::endl ;
  std::cout << "value: " << env->GetValues() << std::endl ;
  std::cout << "states: " ;
  for (size_t i = 0; i < env->GetStates().size(); i++)
    std::cout << env->GetStates()[i] << " " ;
  std::cout << std::endl ;
  std::cout << "actions: " ;
  for (size_t i = 0; i < env->GetActions().size(); i++)
    std::cout << env->GetActions()[i] << " " ;
  std::cout << std::endl ;
  std::cout << "observations: " ;
  for (size_t i = 0; i < env->GetObservations().size(); i++)
    std::cout << env->GetObservations()[i] << " " ;
  std::cout << std::endl ;
  std::cout << "Transition matrices:\n" ;
  for (size_t i = 0; i < env->GetTransitions().size(); i++){
    for (int j = 0; j < env->GetTransitions()[i].rows(); j++){
      for (int k = 0; k < env->GetTransitions()[i].cols(); k++)
        std::cout << env->GetTransitions()[i](j,k) << " " ;
      std::cout << "\n" ;
    }
    std::cout << "\n" ;
  }
  std::cout << "Observation probability matrices:\n" ;
  for (size_t i = 0; i < env->GetObservationProbabilities().size(); i++){
    for (int j = 0; j < env->GetObservationProbabilities()[i].rows(); j++){
      for (int k = 0; k < env->GetObservationProbabilities()[i].cols(); k++)
        std::cout << env->GetObservationProbabilities()[i](j,k) << " " ;
      std::cout << "\n" ;
    }
    std::cout << "\n" ;
  }
  std::cout << "Rewards:\n" ;
  for (size_t i = 0; i < env->GetRewards().size(); i++){// initial state
    for (size_t j = 0; j < env->GetRewards()[i].size(); j++){// transitioned state
      std::cout << "From " << env->GetStates()[i] << " to " << env->GetStates()[j] << "\n" ;
      for (int k = 0; k < env->GetRewards()[i][j].rows(); k++){
        for (int l = 0; l < env->GetRewards()[i][j].cols(); l++)
          std::cout << env->GetRewards()[i][j](k,l) << " " ;
        std::cout << "\n" ;
      }
      std::cout << "\n" ;
    }
  }
  
  
  std::cout << "\nPOMDP policy properties...\n" ;
  std::cout << "Action vector and policy matrix\n" ;
  for (size_t i = 0; i < pol->GetActionVector().size(); i++){
    std::cout << pol->GetActionVector()[i] << "," ;
    for (int j = 0; j < pol->GetPolicyMatrix()[i].size(); j++)
      std::cout << pol->GetPolicyMatrix()[i](j) << " " ;
    std::cout << "\n" ;
  }
  
  std::cout << "Computing best action on uninformative prior...\n" ;
  size_t a = testPOMDP.GetBestAction() ;
  std::cout << "Best action: " << a << "\n" ;
  
  std::cout << "Updating belief...\n" ;
  size_t o = 0 ;
  testPOMDP.UpdateBelief(a,o) ;
  a = testPOMDP.GetBestAction() ;
  
  std::cout << "New best action: " << a << "\n" ;
  
  std::cout << "Test complete!\n" ;
  
  return 0 ;
}
