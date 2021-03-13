#include "SingleRover.h"
#include <iostream>

// Constructor: Initialises physical properties of domain given size of the grid world and number of POIs. Currenly hard coded for 4 input 2 output NN control policies. Also initialises simulation properties given number of timesteps per learning epoch and toggles boolean for writing NN evaluations to file. 
SingleRover::SingleRover(vector<double> wLims, size_t nPOIs, size_t n, size_t nPop): worldLimits(wLims), numPOIs(nPOIs), nSteps(n), outputEval(false), outputTraj(false), outputNNs(false){
  InitialiseNewLearningEpoch() ;
  numIn = 4 ; // hard coded for 4 element input (body frame quadrant decomposition)
  numOut = 2 ; // hard coded for 2 element output [dx,dy]
  numHidden = 8 ;
  RoverNE = new NeuroEvo(numIn, numOut, numHidden, nPop) ;
}

// Destructor: Deletes NE control policies and closes write file
SingleRover::~SingleRover(){
  if (outputEval)
    evalFile.close() ;
  if (outputTraj){
    trajFile.close() ;
    POIFile.close() ;
  }
  delete(RoverNE) ;
  RoverNE = 0 ;
}

// Main learning execution step, includes evolution competition and resetting the environment after the first learning epoch
void SingleRover::ExecuteLearning(size_t nEpochs){
  for (size_t i = 0; i < nEpochs; i++){
    std::cout << "Episode " << i << "..." ;
    if (i > 0){
      RoverNE->EvolvePopulation(epochEvals) ;
//      epochEvals.clear() ;
      InitialiseNewLearningEpoch() ;
    }
    
    RoverNE->MutatePopulation() ;
    if (i == nEpochs-1 && outputTraj)
      SimulateEpoch(true) ;
    else
      SimulateEpoch() ;
    
    std::cout << "complete!\n" ;
  }
    
  if (outputEval)
    evalFile << maxPossibleEval << "\n" ;
  
  if (outputNNs){
    size_t nPop = RoverNE->GetCurrentPopSize()/2 ; // only write in non-mutated (competitive) policies
    for (size_t i = 0; i < nPop; i++){
      NeuralNet * NN = RoverNE->GetNNIndex(i) ;
      MatrixXd NNA = NN->GetWeightsA() ;
      for (int j = 0; j < NNA.rows(); j++){
        for (int k = 0; k < NNA.cols(); k++)
          NNFile << NNA(j,k) << "," ;
        NNFile << "\n" ;
      }
      
      MatrixXd NNB = NN->GetWeightsB() ;
      for (int j = 0; j < NNB.rows(); j++){
        for (int k = 0; k < NNB.cols(); k++)
          NNFile << NNB(j,k) << "," ;
        NNFile << "\n" ;
      }
    }
    NNFile.close() ;
  }
}

// Initial simulation parameters, includes setting initial rover position, POI positions and values, and clearing the evaluation storage vector
void SingleRover::InitialiseNewLearningEpoch(){
  // Clear initial world properties
  initialXY.setZero(initialXY.size(),1) ;
  POIs.clear() ;
  epochEvals.clear() ;
  maxPossibleEval = 0.0 ;
  
  // Initial XY location and heading in global frame (restricted to within inner region of 9 grid)
  double rangeX = worldLimits[1] - worldLimits[0] ;
  double rangeY = worldLimits[3] - worldLimits[2] ;
  initialXY(0) = rand_interval(worldLimits[0]+rangeX/3.0,worldLimits[1]-rangeX/3.0) ;
  initialXY(1) = rand_interval(worldLimits[2]+rangeY/3.0,worldLimits[3]-rangeX/3.0) ;
  initialPsi = rand_interval(-PI,PI) ;
  
  // POI locations and values in global frame (restricted to within outer regions of 9 grid)
  for (size_t i = 0; i < numPOIs; i++){
    Vector2d xy ;
    double x, y ;
    bool accept = false ;
    while (!accept){
      x = rand_interval(worldLimits[0],worldLimits[1]) ;
      y = rand_interval(worldLimits[0],worldLimits[1]) ;
      if (x > worldLimits[0]+rangeX/3.0 && x < worldLimits[1]-rangeX/3.0 && y > worldLimits[2]+rangeY/3.0 && y < worldLimits[3]-rangeX/3.0) {}
      else accept = true ;
    }
    xy(0) = x ; // x location
    xy(1) = y ; // y location
    double v = rand_interval(1,10) ; // value
    POIs.push_back(Target(xy,v)) ;
    maxPossibleEval += v ; // compute maximum achievable performance
  }
}

// Simulation loop, tests each NN in the current population in the simulation world. Each simulation starts with the same configuration of rover location and POI locations and values.
void SingleRover::SimulateEpoch(bool write){
  // Write POI configuration to file
  if (write && outputTraj)
    for (size_t i = 0; i < numPOIs; i++)
      POIFile << POIs[i].GetLocation()(0) << "," << POIs[i].GetLocation()(1) << "," << POIs[i].GetValue() << "\n" ;
  
  double maxEval = 0.0 ;
  for (size_t i = 0; i < RoverNE->GetCurrentPopSize(); i++){
    Vector2d xy = initialXY ;
    double psi = initialPsi ;
    
    // Write current global state to file
    if (write)
      trajFile << xy(0) << "," << xy(1) << "," << psi << "\n" ;
    
    for (size_t t = 0; t < nSteps; t++){
      // Calculate body frame NN input state
      VectorXd s = ComputeNNInput(xy, psi) ;
      
      // Calculate body frame action
      VectorXd a = RoverNE->GetNNIndex(i)->EvaluateNN(s).normalized() ;
      
      // Transform to global frame
      Matrix2d Body2Global = RotationMatrix(psi) ;
      Vector2d deltaXY = Body2Global*a ;
      double deltaPsi = atan2(a(1),a(0)) ;
      
      // Move
      xy += deltaXY ;
      psi += deltaPsi ;
      psi = pi_2_pi(psi) ;
      
      // Compute observations
      for (size_t j = 0; j < numPOIs; j++)
        POIs[j].ObserveTarget(xy) ;
      
      // Write current global state to file
      if (write && outputTraj)
        trajFile << xy[0] << "," << xy[1] << "," << psi << "\n" ;
    }
    
    // Evaluate NN
    double eval = 0.0 ;
    for (size_t j = 0; j < numPOIs; j++){
      eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
      POIs[j].ResetTarget() ;
    }
    
    epochEvals.push_back(eval) ;
    
    if (eval > maxEval)
      maxEval = eval ;
    
  }
  std::cout << "max achieved value: " << maxEval << "..." ;
  if (outputEval)
    evalFile << maxEval << "," ; // output as fraction of maximum achievable performance
}

void SingleRover::ExecutePolicy(char * readFile, char * storeTraj, char * storePOI, size_t numIn, size_t numOut, size_t numHidden){
  // Filename to read NN control policy
	std::stringstream fileName ;
  fileName << readFile ;
  std::ifstream nnFile ;
  
  vector<NeuralNet *> loadedNN ;
  std::cout << "Reading out " <<  RoverNE->GetCurrentPopSize()/2 << " NN control policies to test...\n" ;
  nnFile.open(fileName.str().c_str(),std::ios::in) ;
  
  // Read in NN weight matrices
  std::string line ;
  NeuralNet * NN = RoverNE->GetNNIndex(0) ;
  MatrixXd NNA = NN->GetWeightsA() ;
  MatrixXd NNB = NN->GetWeightsB() ;
  int nnK = NNA.rows() + NNB.rows() ; // number of lines corresponding to a single control policy
  int k = 0 ; // track line number
  while (std::getline(nnFile,line)){
    std::stringstream lineStream(line) ;
    std::string cell ;
    if (k % nnK < NNA.rows()){
      int i = k % nnK ;
      int j = 0 ;
      while (std::getline(lineStream,cell,','))
        NNA(i,j++) = atof(cell.c_str()) ;
    }
    else {
      int i = (k % nnK)-NNA.rows() ;
      int j = 0 ;
      while (std::getline(lineStream,cell,','))
        NNB(i,j++) = atof(cell.c_str()) ;
    }
    if ((k+1) % nnK == 0){
      NeuralNet * newNN = new NeuralNet(numIn, numOut, numHidden) ;
      newNN->SetWeights(NNA, NNB) ;
      loadedNN.push_back(newNN) ;
    }
    k++ ;
  }
  nnFile.close() ;
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseNewLearningEpoch() ;
  
  // Initialise files to store policy execution results
  std::ofstream tFile ;
	std::stringstream tfileName ;
  tfileName << storeTraj ;
  tFile.open(tfileName.str().c_str(),std::ios::app) ;
  
  std::ofstream pFile ;
	std::stringstream pfileName ;
  pfileName << storePOI ;
  pFile.open(pfileName.str().c_str(),std::ios::app) ;
  
  // Write POI configuration to file
  for (size_t i = 0; i < numPOIs; i++)
    pFile << POIs[i].GetLocation()(0) << "," << POIs[i].GetLocation()(1) << "," << POIs[i].GetValue() << "\n" ;
  
  for (size_t i = 0; i < loadedNN.size(); i++){
    Vector2d xy = initialXY ;
    double psi = initialPsi ;
    
    // Write current global state to file
    tFile << xy(0) << "," << xy(1) << "," << psi << "\n" ;
    
    for (size_t t = 0; t < nSteps; t++){
      // Calculate body frame NN input state
      VectorXd s = ComputeNNInput(xy, psi) ;
      
      // Calculate body frame action
      VectorXd a = loadedNN[i]->EvaluateNN(s).normalized() ;
      
      // Transform to global frame
      Matrix2d Body2Global = RotationMatrix(psi) ;
      Vector2d deltaXY = Body2Global*a ;
      double deltaPsi = atan2(a(1),a(0)) ;
      
      // Move
      xy += deltaXY ;
      psi += deltaPsi ;
      psi = pi_2_pi(psi) ;
      
      // Compute observations
      for (size_t j = 0; j < numPOIs; j++)
        POIs[j].ObserveTarget(xy) ;
      
      // Write current global state to file
      tFile << xy[0] << "," << xy[1] << "," << psi << "\n" ;
    }
    
    // Evaluate NN
    double eval = 0.0 ;
    for (size_t j = 0; j < numPOIs; j++){
      eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
      POIs[j].ResetTarget() ;
    }
    
    std::cout << "Performance of executed policy: " << eval << std::endl ; // output as fraction of maximum achievable performance
  }
  
  tFile.close() ;
  pFile.close() ;
  nnFile.close() ;
  
  for (size_t i = 0; i < loadedNN.size(); i++){
    delete loadedNN[i] ;
    loadedNN[i] = 0 ;
  }
}

// Wrapper for writing epoch evaluations to specified files
void SingleRover::OutputPerformance(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  evalFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputEval = true ;
}

// Wrapper for writing final trajectories to specified files
void SingleRover::OutputTrajectories(char * A, char * B){
	// Filename to write trajectories to stored in A
	std::stringstream tfileName ;
  tfileName << A ;
  trajFile.open(tfileName.str().c_str(),std::ios::app) ;
  
  // Filename to write POIs to stored in B
	std::stringstream pfileName ;
  pfileName << B ;
  POIFile.open(pfileName.str().c_str(),std::ios::app) ;
  
  outputTraj = true ;
}

// Wrapper for writing final control policies to specified binary file
void SingleRover::OutputNNs(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  NNFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputNNs = true ;
}

// Compute the NN input state given the rover location and the POI locations and values in the world
VectorXd SingleRover::ComputeNNInput(Vector2d xy, double psi){
  VectorXd s ;
  s.setZero(4,1) ;
  MatrixXd Global2Body = RotationMatrix(-psi) ;
  Vector2d POIv ;
  POIv.setZero(2,1) ;
  for (size_t i = 0; i < numPOIs; i++){
    POIv = POIs[i].GetLocation() - xy ;
    Vector2d POIbody = Global2Body*POIv ;
    Vector2d diff = xy - POIbody ;
    double d = diff.norm() ;
    double theta = atan2(POIbody(1),POIbody(0)) ;
    size_t q ;
    if (theta >= PI/2.0)
      q = 3 ;
    else if (theta >= 0.0)
      q = 0 ;
    else if (theta >= -PI/2.0)
      q = 1 ;
    else
      q = 2 ;
    s(q) += POIs[i].GetValue()/max(d,1.0) ;
//    std::cout << "Rover global state: (" << xy[0] << "," << xy[1] << "," << psi*180.0/PI 
//    << "), POI global location: (" << POIs[i].GetLocation()[0] << "," << POIs[i].GetLocation()[1] 
//    << "), POI body location: (" << POIbody[0] << "," << POIbody[1]
//    << "), bearing: " << theta*180.0/PI << ", quadrant: " << q << std::endl ;
  }
//  std::cout << "State: [" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n" ;
  return s ;
}

Matrix2d SingleRover::RotationMatrix(double psi){
  Matrix2d R ;
  R(0,0) = cos(psi) ;
  R(0,1) = -sin(psi) ;
  R(1,0) = sin(psi) ;
  R(1,1) = cos(psi) ;
  return R ;
}
