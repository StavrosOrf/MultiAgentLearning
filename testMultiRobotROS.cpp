#include <iostream>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <Eigen/Eigen>
#include <float.h>

#include "multirobot_stage/NeuroEvo.h"
#include "multirobot_stage/NeuralNet.h"

#include <ros/ros.h>

using std::vector ;
using std::shuffle ;
using namespace Eigen ;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "multirobot_learning") ;
  
  std::cout << "Program to run neuro-evolution on a pair of collaboratively exploring robots\n" ;
  
  // Initialise learning domain
  // Domain contains 2 NeuroEvo agents, each agent maintains a population of policies which are encoded as neural networks
  // Neural networks are initialised according to size of input and output vectors, number of nodes in the hidden layer, and the activation function
  size_t nRob = 2 ;     // number of robots
  size_t nIn = 18 ;     // input state vector size
  size_t nOut = 10 ;    // output action vector size
  size_t nHidden = 20 ; // number of hidden neurons
  size_t nPop = 15 ;    // population size
  
  vector<NeuroEvo *> robotTeam ; // default activation is TANH
  for (size_t n = 0; n < nRob; n++){
    NeuroEvo * NE = new NeuroEvo(nIn, nOut, nHidden, nPop) ;
    robotTeam.push_back(NE) ;
  }
  
  // Logging
  vector<double> rewardLog ;
  double maxR = -DBL_MAX ;
  size_t maxTeam = 0 ;
  
  // Container for storing population indices
  vector<size_t> tt ;
  for (size_t p = 0; p < nPop*2; p++){
    tt.push_back(p) ;
  }
    
  // Container to store rewards
  vector<double> rr(nPop*2,0.0) ;
  
  // Initialise learning loop
  size_t nEps = 100 ;
  for (size_t i = 0; i < nEps; i++){ // For each learning epoch
  
    // Housekeeping for managing teams
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;
    vector< vector<size_t> > teams ;
    vector< vector<double> > rewards ;
    
    // Mutate populations (doubles population size)
    for (size_t n = 0; n < nRob; n++){
      robotTeam[n]->MutatePopulation() ;
      
      // Create randomised teams for this epoch
      shuffle (tt.begin(), tt.end(), std::default_random_engine(seed)) ;
      teams.push_back(tt) ;
      
      // Initialise reward container for this epoch
      rewards.push_back(rr) ;
    }
    
    // Evaluate each team
    for (size_t j = 0; j < nPop*2; j++){
      // Get policies for this team
      for (size_t n = 0; n < nRob; n++){
        NeuralNet * curNN = robotTeam[n]->GetNNIndex(teams[n][j]) ;
        
        // Reshape weight matrices to single vector
        MatrixXd A = curNN->GetWeightsA() ;
        vector<double> AA ;
        for (int ii = 0; ii < A.rows(); ii++){
          for (int jj = 0; jj < A.cols(); jj++){
            AA.push_back(A(ii,jj)) ;
          }
        }
        
        MatrixXd B = curNN->GetWeightsB() ;
        vector<double> BB ;
        for (int ii = 0; ii < B.rows(); ii++){
          for (int jj = 0; jj < B.cols(); jj++){
            BB.push_back(B(ii,jj)) ;
          }
        }
        
        // Write NN policies to rosparams
        char buffer[50] ;
        sprintf(buffer,"/robot_%lu/A",n) ;
        ros::param::set(buffer,AA) ;
        sprintf(buffer,"/robot_%lu/B",n) ;
        ros::param::set(buffer,BB) ;
      }
      
      // Run stage simulation
      // Simulation nodes must initialise robots in stage
      // Nodes must also read in ROS params for robot control policies
      // This implementation expects the reward to be written to a rosparam
      // Simulation timer will automatically terminal the episode
      system("rosrun multirobot_stage run-multi-robot-explore") ;
      
      // Read out reward
      double r ;
      ros::param::get("/reward",r) ;
      for (size_t n = 0; n < nRob; n++){
        rewards[n][teams[n][j]] = r ;
      }
      if (r > maxR){
        maxR = r ;
        maxTeam = j ;
      }
    }
    
    // Record/Evaluate champion team
    rewardLog.push_back(maxR) ;
    char strA[50] ;
    char strB[50] ;
    for (size_t n = 0; n < nRob; n++){
      sprintf(strA,"weightsA%lu.txt",n) ;
      sprintf(strB,"weightsB%lu.txt",n) ;
      NeuralNet * bestNN = robotTeam[n]->GetNNIndex(teams[n][maxTeam]) ;
      bestNN->OutputNN(strA,strB) ;
    }
    
    // Compete (halves population size)
    for (size_t n = 0; n < nRob; n++){
      robotTeam[n]->EvolvePopulation(rewards[n]) ;
    }
  }
  
  for (size_t n = 0; n < nRob; n++){
    delete robotTeam[n] ;
    robotTeam[n] = 0 ;
  }
  
  return 0 ;
}
