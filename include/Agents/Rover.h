#ifndef ROVER_H_
#define ROVER_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <float.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Utilities/Utilities.h"
#include "Domains/Target.h"
#include "POMDPs/POMDP.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using std::string ;
using std::vector ;
using std::list ;
using std::max ;
using easymath::pi_2_pi ;
using easymath::rand_interval ;

class Rover{
  public:
    Rover(size_t n, size_t nPop, string evalFunc) ;
    ~Rover() ;
    
    void ResetEpochEvals() ;
    void InitialiseNewLearningEpoch(vector<Target>, Vector2d, double) ;
    void ResetStepwiseEval() ;
    
    Vector2d ExecuteNNControlPolicy(size_t , vector<Vector2d>) ; // executes NN_i from current (x,y,psi), outputs new (x,y)
    void ComputeStepwiseEval(vector<Vector2d>, double) ;
    void SetEpochPerformance(double G, size_t i) ;
    vector<double> GetEpochEvals(){return epochEvals ;}
    
    void EvolvePolicies(bool init = false) ;
    
    void OutputNNs(char *) ;
    NeuroEvo * GetNEPopulation(){return RoverNE ;}
    
    void SetPOMDPPolicy(POMDP * pomdp) ;
    POMDP * GetPOMDPPolicy(){return expertisePOMDP ;}
    VectorXd GetPOMDPBelief(){return belief ;}
    size_t ComputePOMDPAction() ;
    void UpdateNNStateInputCalculation(bool, size_t) ;
    bool IsStateObsUpdated(){return stateObsUpdate ;}
    
    double GetAverageR() ;
    
    void SetLearningEvaluation(double, bool b = true) ;
    void OutputImpact(char *) ;
    bool GetIsLearn(){return isLearn ;}
  private:
    size_t nSteps ;
    size_t popSize ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    Vector2d initialXY ;
    double initialPsi ;
    vector<Target> POIs ;
    Vector2d currentXY ;
    double currentPsi ;
    
    bool isD ;
    double stepwiseD ;
    vector<double> epochEvals ;
    NeuroEvo * RoverNE ;
    list<double> runningAvgR ;
    size_t windowSize ;
    
    VectorXd ComputeNNInput(vector<Vector2d>) ;
    Matrix2d RotationMatrix(double) ;
    
    // Variables associated with policy expertise modeling
    size_t pomdpAction ;
    vector<double> rThreshold ;
    POMDP * expertisePOMDP ;
    VectorXd belief ;
    
    // Variables associated with whether or not to learn
    bool evalLearning ;
    bool isLearn ;
    double tau ; // temperature
    vector<double> deltaPi ;
    vector<double> deltaR ;
    vector<double> dRdPi ;
    double sumdRdPi ;
    vector<double> epochG ;
    
    bool stateObsUpdate ;
    size_t goalPOI ;
    
    void DifferenceEvaluationFunction(vector<Vector2d>, double) ;
    void UpdatedStateEvaluationFunction(vector<Vector2d>, double) ;
} ;

#endif // ROVER_H_
