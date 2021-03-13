#ifndef MAP_ELITES_ROVER_H_
#define MAP_ELITES_ROVER_H_

#include <iostream>
#include <algorithm>
#include <math.h>
#include <Eigen/Eigen>
#include "Learning/MAPElites.h"
#include "Target.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using namespace Eigen ;
using easymath::rand_interval ;
using easymath::pi_2_pi ;
using std::max ;

class MAPElitesRover{
  public:
    MAPElitesRover(vector<double>, size_t, size_t, MatrixXd) ;
    ~MAPElitesRover() ;
    
    void InitialiseMap(size_t) ;
    void EvolveMap(size_t) ;
    double SimulateController(NeuralNet *, VectorXd &, bool write = false) ;
    
    double PercentageFilled() ;
    double BestPerformance(NeuralNet *, VectorXd &) ;
    void OutputTrajectories(char *, char *) ;
    void WriteToBinary(char *, char *, char *) ;
    void ReadFromBinary(char *, char *, char *) ;
  private:
    MAPElites * bpMap ;
    
    // NN controller properties
    size_t input_size ;
    size_t output_size ;
    size_t hidden_size ;
    
    // Simulation properties
    vector<double> worldLimits ;
    size_t numPOIs ;
    size_t nSteps ;
    Vector2d initialXY ;
    double initialPsi ;
    vector<Target> POIs ;
    double minPOIVal ;
    double maxPOIVal ;
    double maxPossibleEval ;
    
    // Write variables
    bool outputTraj ;
    std::ofstream trajFile ;
    std::ofstream POIFile ;
    
    void InitialiseSimulationWorld() ;
    
    VectorXd ComputeNNInput(Vector2d, double) ;
    Matrix2d RotationMatrix(double) ;
    void ComputeBehaviourActions(VectorXd &, double) ;
    void ComputeBehaviourObservations(VectorXd &, VectorXd) ;
} ;
#endif // MAP_ELITES_ROVER_H_
