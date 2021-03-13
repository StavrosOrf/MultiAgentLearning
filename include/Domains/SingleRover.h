#ifndef SINGLE_ROVER_H_
#define SINGLE_ROVER_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Utilities/Utilities.h"
#include "Target.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using easymath::rand_interval ;
using easymath::pi_2_pi ;
using std::max ;

class SingleRover{
  public:
    SingleRover(vector<double>, size_t, size_t, size_t) ;
    ~SingleRover() ;
    
    void ExecuteLearning(size_t) ;
    
    void InitialiseNewLearningEpoch() ;
    void SimulateEpoch(bool write=false) ;
    
    void ExecutePolicy(char * readFile, char * storeTraj, char * storePOI, size_t numIn, size_t numOut, size_t numHidden) ; // read in control policy and execute in random world, store trajectory and POI results in second and third inputs, fourth-sizth inputs define NN structure, final input indicates if readFile is a binary file
    
    void OutputPerformance(char *) ; // write epoch evaluations to file
    void OutputTrajectories(char *, char *) ; // write final trajectories and POIs to file
    void OutputNNs(char *) ; // write final control policies to file
  private:
    vector<double> worldLimits ;
    size_t numPOIs ;
    size_t nSteps ;
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    vector<Target> POIs ;
    Vector2d initialXY ;
    double initialPsi ;
    
    double maxPossibleEval ;
    vector<double> epochEvals ;
    NeuroEvo * RoverNE ;
    
    bool outputEval ;
    std::ofstream evalFile ;
    
    bool outputTraj ;
    std::ofstream trajFile ;
    std::ofstream POIFile ;
    
    bool outputNNs ;
    std::ofstream NNFile ;
    
    VectorXd ComputeNNInput(Vector2d, double) ;
    Matrix2d RotationMatrix(double) ;
} ;
#endif // SINGLE_ROVER_H_
