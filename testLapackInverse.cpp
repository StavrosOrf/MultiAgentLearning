#include <iostream>
#include <math.h>
#include "Utilities/MatrixTypes.h"
#include "Utilities/UtilFunctions.h"

using easymath::zeros ;
using easymath::rand ;
using easymath::matrix_mult ;

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
    
//    // compute matrix determinant
//    void dgeicd_(double* A, int* lda, int* N, int* IOPT, double* RCOND, double* det, double* AUX, int* NAUX) ;
}

void inverse(double* A, int N)
{
    int *IPIV = new int[N];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete [] IPIV;
    delete [] WORK;
}

void LUdecomposition(double* A, int N){
  int *IPIV = new int[N];
  int INFO;

  dgetrf_(&N,&N,A,&N,IPIV,&INFO);
  for (size_t i = 0; i < N; i++)
    printf("%d ",IPIV[i]) ;
  printf("\n") ;

  delete [] IPIV;
}

//void determinant(double* A, double* det, int N){
//  int IOPT = 2 ;
//  double* RCOND ;
//  int LWORK = N*N;
//  double *WORK = new double[LWORK];
//  dgeicd_(A,&N,&N,&IOPT,RCOND,det,WORK,&LWORK) ;
//}

int main(){

  size_t N = 3 ;
  matrix2d A = zeros(N,N) ;
  for (size_t i = 0; i < A.size(); i++)
    for (size_t j = i; j < A[i].size(); j++)
      A[i][j] = rand(1.0, 10.0) ;
  
  matrix2d At = zeros(N,N) ;
  for (size_t i = 0; i < A.size(); i++)
    for (size_t j = 0; j < i+1; j++)
      At[i][j] = A[j][i] ;
  
  printf("A:\n") ;
  for (size_t i = 0; i < A.size(); i++){
    for (size_t j = 0; j < A[i].size(); j++){
      printf("%f ",A[i][j]) ;
    }
    printf("\n") ;
  }
  printf("A transpose:\n") ;
  for (size_t i = 0; i < At.size(); i++){
    for (size_t j = 0; j < At[i].size(); j++){
      printf("%f ",At[i][j]) ;
    }
    printf("\n") ;
  }
  
  
  matrix2d B = matrix_mult(A,At) ;
  printf("B = A*A: \n") ;
  for (size_t i = 0; i < B.size(); i++){
    for (size_t j = 0; j < B[i].size(); j++){
      printf("%f ",B[i][j]) ;
    }
    printf("\n") ;
  }
  
  double BB[N*N] ;
  size_t k = 0 ;
  printf("double BB[]: \n") ;
  for (size_t i = 0; i < B.size(); i++){
    for (size_t j = 0; j < B[i].size(); j++){
      BB[k++] = B[i][j] ;
      printf("%f ",BB[k-1]) ;
    }
    printf("\n") ;
  }
  
  double CC[N*N] ;
  for (size_t i = 0; i < k; i++)
    CC[i] = BB[i] ;
  
  inverse(BB, N) ;
  
  k = 0 ;
  printf("Overwritten BB now stores inverse: \n") ;
  for (size_t i = 0; i < B.size(); i++){
    for (size_t j = 0; j < B[i].size(); j++){
      printf("%f ",BB[k++]) ;
    }
    printf("\n") ;
  }
  
  matrix2d invB = zeros(N,N) ;
  k = 0 ;
  printf("matrix2d invB: \n") ;
  for (size_t i = 0; i < invB.size(); i++){
    for (size_t j = 0; j < invB[i].size(); j++){
      invB[i][j] = BB[k++] ;
      printf("%f ",invB[i][j]) ;
    }
    printf("\n") ;
  }
  
  matrix2d BinvB = matrix_mult(B,invB) ;
  printf("B*invB: \n") ;
  for (size_t i = 0; i < BinvB.size(); i++){
    for (size_t j = 0; j < BinvB[i].size(); j++){
      printf("%f ",BinvB[i][j]) ;
    }
    printf("\n") ;
  }
  
  LUdecomposition(CC,N) ;
  k = 0 ;
  printf("LU decomposition: \n") ;
  for (size_t i = 0; i < B.size(); i++){
    for (size_t j = 0; j < B[i].size(); j++){
      printf("%f ",CC[k++]) ;
    }
    printf("\n") ;
  }
  
  
//  double det[2] ;
//  determinant(CC,det,N) ;
//  
//  k = 0 ;
//  printf("Overwritten CC now stores inverse: \n") ;
//  for (size_t i = 0; i < B.size(); i++){
//    for (size_t j = 0; j < B[i].size(); j++){
//      printf("%f ",CC[k++]) ;
//    }
//    printf("\n") ;
//  }
//  
//  double detB = det[0]*pow(10.0,det[1]) ;
//  printf("Determinant of B: %f\n", detB) ;
//  printf("Determinant of invB = 1/det(B): %f\n", 1.0/detB) ;
//  
//  determinant(CC,det,N) ;
//  double detInvB = det[0]*pow(10.0,det[1]) ;
//  printf("Determinant of invB: %f\n", detInvB) ;

  return 0;
}
