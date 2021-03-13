#ifndef TARGET_H_
#define TARGET_H_

#include <iostream>
#include <float.h>
#include <Eigen/Eigen>

using namespace Eigen ;

class Target{
  public:
    Target(Vector2d xy, double v, int c = 1): loc(xy), val(v), obsRadius(4.0), nearestObs(DBL_MAX), observed(false), coupling(c){
      nearestObsVector.setZero(c) ;
      for (int i = 0; i < nearestObsVector.size(); i++)
        nearestObsVector(i) = DBL_MAX ;
      curTime = -1 ;
    }
    ~Target(){}
    
    Vector2d GetLocation(){return loc ;}
    double GetValue(){return val ;}
    double GetNearestObs(){return nearestObs ;}
    bool IsObserved(){return observed ;}
    
    void ObserveTarget(Vector2d xy){
      Vector2d diff = xy - loc ;
      double d = diff.norm() ;
      if (observed && d < nearestObs)
        nearestObs = d ;
      else if (!observed && d <= obsRadius){
        nearestObs = d ;
        observed = true ;
      }
    }
    
    void ObserveTarget(Vector2d xy, size_t t){
      Vector2d diff = xy - loc ;
      double d = diff.norm() ;
      
      if (curTime != t){ // reset observations for current timestep
        curTime = t ;
        for (int i = 0; i < nearestObsVector.size(); i++)
          nearestObsVector(i) = DBL_MAX ;
      }
      
      if (d <= obsRadius){ // rover within sensing radius
        double maxD = 0.0 ;
        int ind = -1 ;
        for (int i = 0; i < nearestObsVector.size(); i++){
          double dObs = nearestObsVector(i) - d ;
          if (dObs > maxD){
            maxD = dObs ;
            ind = i ;
          }
        }
        
        if (ind >= 0) // replace furthest valid observation
          nearestObsVector(ind) = d ;
        
        if (!observed && ind == (coupling-1)){ // coupling requirements satisfied
          observed = true ;
        }
        
        if (ind >= 0 && observed)
          nearestObs = nearestObsVector.mean() ;
      }
    }
    
    void ResetTarget(){
      nearestObs = DBL_MAX ;
      observed = false ;
    }
  private:
    Vector2d loc ;
    double val ;
    double obsRadius ;
    double nearestObs ;
    bool observed ;
    size_t curTime ;
    int coupling ;
    VectorXd nearestObsVector ;
} ;
#endif // TARGET_H_
