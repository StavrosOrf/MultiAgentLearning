#ifndef GAUSSIAN_PROCESS_H_
#define GAUSSIAN_PROCESS_H_

#include <math.h>
#include "Utilities/UtilFunctions.h"

using easymath::matrix_mult ;
using easymath::determinant ;
using easymath::inverse ;
using easymath::zeros ;

class GaussianProcess{
  public:
    GaussianProcess(matrix2d, matrix1d) ;
    ~GaussianProcess(){}
    
    double Predict(matrix1d) ;
    double NLML(double &, double &, double &) ;
    
    void UpdateTrainingSet(matrix1d, double) ;
    void TrainHyperparameters() ;
    
    void SetHyperparameters(matrix1d hp){ hyperparameters = hp ; }
    void SetTrainingInputs(matrix2d x){ trainingInputs = x ; }
    void SetTrainingTargets(matrix1d y){ trainingTargets = y ; }
  private:
    matrix2d trainingInputs ;
    matrix1d trainingTargets ;
    matrix1d hyperparameters ;
    matrix2d Kxx ; // covariance matrix
    matrix2d Kzx ; // cross covariance matrix
    matrix2d iKxx ; // inverse covariance matrix
    
    void UpdateCovarianceMatrices() ;
    void (GaussianProcess::*CovarianceFunction)(bool = false, matrix2d = trainingInputs, bool = false) ;
    void SquaredExponential(bool = false, matrix2d = trainingInputs, bool = false) ;
    matrix2d SquareDistance(matrix2d, matrix2d) ;
} ;

GaussianProcess::GaussianProcess(matrix2d inputs, matrix1d targets){
  SetTrainingInputs(inputs) ;
  SetTrainingTargets(targets) ;
  CovarianceFunction = &GaussianProcess::SquaredExponential ;
  TrainHyperparameters() ;
  UpdateCovarianceMatrices() ; // recompute all K matrices using learned hyperparameters
}

double GaussianProcess::Predict(matrix1d z){
  matrix2d testInput ;
  testInput.push_back(z) ;
  (this->*CovarianceFunction)(true, testInput) ; // update Kzx
  matrix2d covK = matrix_mult(Kzx, iKxx) ;
  matrix1d f_z = matrix_mult(covK, trainingTargets) ;
  return f_z[0] ;
}

double GaussianProcess:NLML(double & dataFit, double & complexity, double & normalisation){
  matrix1d yiKxx = matrix_mult(trainingTargets, iKxx) ;
  matrix1d yiKxxy = matrix_mult(yiKxx, trainingTargets) ;
  dataFit = yiKxxy[0] ;
  complexity = 0.5*log(determinant(iKxx)) ;
  normalisation = ((double)trainingInputs.size())*0.5*log(2.0*PI) ;
  
  return dataFit + complexity + normalisation ;
}

void GaussianProcess::UpdateTrainingSet(matrix1d x, double y){
  trainingInputs.push_back(x) ;
  trainingTargets.push_back(y) ;
  UpdateCovarianceMatrices() ;
}

void GaussianProcess::TrainHyperparameters(){
  // solve for hyperparameters that minimize NLML
}

void GaussianProcess::UpdateCovarianceMatrices(){
  (this->*CovarianceFunction)() ;
  iKxx = inverse(Kxx) ;
}

void GaussianProcess::SquaredExponential(bool covType, matrix2d z, bool noNoise){
  size_t sf2 = hyperparameters.size()-2 ; // index of process variance hyperparameter
  size_t sn2 = hyperparameters.size()-1 ; // index of noise variance hyperparameter
  
  matrix2d x = trainingInputs ;
  for (size_t i = 0; i < x.size(); i++)
    for (size_t j = 0; j < x[i].size(); j++)
      x[i][j] /= hyperparameters[j] ;
  
  if (covType) // computing Kzx
    for (size_t i = 0; i < z.size(); i++)
      for (size_t j = 0; j < z[i].size(); j++)
        z[i][j] /= hyperparameters[j] ;
  else // computing Kxx
    z = x ;
  
  matrix2d K = SquareDistance(z,x)
  for (size_t i = 0; i < K.size(); i++){
    for (size_t j = 0; j < K[i].size(); j++){
      K[i][j] = hyperparameters[sf2]*exp(K[i][j])
      if (!covType && !noNoise && i == j)
        K[i][j] += hyperparameters[sn2] ;
    }
  }
  
  if (covType)
    Kzx = K ;
  else
    Kxx = K ;
}

matrix2d GaussianProcess::SquareDistance(matrix2d z, matrix2d x){
  matrix2d A = zeros(z.size(),x.size()) ;
  for (size_t i = 0; i < z.size(); i++)
    for (size_t j = 0; j < x.size(); j++)
      A[i][j] = pow(L2_norm(z[i],x[j]),2) ;
  return A ;
}
#endif // GAUSSIAN_PROCESS_H_
