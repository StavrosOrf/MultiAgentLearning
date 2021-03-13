#ifndef MAP_ELITES_H_
#define MAP_ELITES_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>
#include <math.h>
#include <Eigen/Eigen>
#include "Learning/NeuralNet.h"

using std::vector ;

class MAPElites{
  public:
    MAPElites(MatrixXd, size_t, size_t, size_t) ;
    ~MAPElites() ;
    
    double GetPerformance(VectorXd) ;
    bool IsVisited(VectorXd) ;
    NeuralNet * GetNeuralNet(VectorXd) ;
    NeuralNet * GetNeuralNet(size_t) ;
    void UpdateMap(NeuralNet *, VectorXd, double) ;
    
    size_t GetIndex(VectorXd) ;
    VectorXd GetBehaviour(size_t) ;
    
    size_t GetBDim(){return bDim ;}
    vector<double> GetPerformanceLog(){return performanceLog ;}
    vector<bool> GetFilledLog(){return behaviourFilled ;}
    
    void WriteBPMapBinary(char *) ;
    void ReadBPMapBinary(char *) ;
    
    void WritePerformanceBinary(char *) ;
    void ReadPerformanceBinary(char *) ;
    
    void WriteVisitedBinary(char *) ;
    void ReadVisitedBinary(char *) ;
  private:
    int bDim ;
    VectorXi numBins ;
    MatrixXd binLimits ;
    VectorXi cProd ;
    vector<NeuralNet *> behaviourMap ;
    vector<double> performanceLog ;
    vector<bool> behaviourFilled ;
} ;
#endif // MAP_ELITE_H_
