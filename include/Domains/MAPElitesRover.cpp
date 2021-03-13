#include "MAPElitesRover.h"

MAPElitesRover::MAPElitesRover(vector<double> wLims, size_t nPOIs, size_t n, MatrixXd bins): worldLimits(wLims), numPOIs(nPOIs), nSteps(n), outputTraj(false){
  input_size = 4 ; // hard coded for 4 element input (body frame quadrant decomposition)
  output_size = 2 ; // hard coded for 2 element output [dx,dy]
  hidden_size = 2*input_size ; // hard coded for 2*input_size hidden nodes
  
  minPOIVal = 1.0 ;
  maxPOIVal = 10.0 ;
  bpMap = new MAPElites(bins, input_size, output_size, hidden_size) ;
  
  // Use same world every time to create behaviour performance map
  InitialiseSimulationWorld() ;
}

MAPElitesRover::~MAPElitesRover(){
  delete(bpMap) ;
  bpMap = 0 ;
}

void MAPElitesRover::InitialiseMap(size_t n){
  for (size_t i = 0; i < n; i++){
    if ( fmod(i,(n/10)) == 0 )
      std::cout << i << " controllers tested...\n" ;
    
    // Create random NN controller
    NeuralNet * curNN = new NeuralNet(input_size, output_size, hidden_size) ;
    
    // Create new simulation world
//    InitialiseSimulationWorld() ; // commented out so that behaviour performance map is generated using the same simulation world
    
    // Simulate controller in world
    VectorXd bVec ;
    bVec.setZero(bpMap->GetBDim(),1) ;
    double eval = SimulateController(curNN, bVec) ;
    
    // Update MAPElites behaviour-performance map
    bpMap->UpdateMap(curNN, bVec, eval) ;
    
    // Release memory
    delete(curNN) ;
  }
  std::cout << "Map initialisation complete!\n" ;
}

void MAPElitesRover::EvolveMap(size_t n){
  NeuralNet * curNN = new NeuralNet(input_size, output_size, hidden_size) ;
  
  for (size_t i = 0; i < n; i++){
    if ( fmod(i,(n/10)) == 0 )
      std::cout << i << " controllers tested...\n" ;
    
    // Select a random behaviour and find corresponding NN in bpMap
    VectorXd bVec ;
    bVec.setZero(bpMap->GetBDim(),1) ;
    for (int j = 0; j < bVec.size(); j++)
      bVec(j) = rand_interval(0.0,1.0) ;
    
    bool isTested = bpMap->IsVisited(bVec) ;
    while (!isTested){
      for (int j = 0; j < bVec.size(); j++)
        bVec(j) = rand_interval(0.0,1.0) ;
      isTested = bpMap->IsVisited(bVec) ;
    }
    
    NeuralNet * tempNN = bpMap->GetNeuralNet(bVec) ;
    
    // Copy weight matrices to current NN controller
    curNN->SetWeights(tempNN->GetWeightsA(),tempNN->GetWeightsB()) ;
    
    // Mutate weights of current controller
    curNN->MutateWeights() ;
    
    // Create a new simulation world
//    InitialiseSimulationWorld() ; // commented out so that behaviour performance map is generated using the same simulation world
    
    // Simulate controller in world
    double eval = SimulateController(curNN, bVec) ;
    
    // Update MAPElites behaviour-performance map
    bpMap->UpdateMap(curNN, bVec, eval) ;
  }
  
  // Release memory
  delete(curNN) ;
  
  std::cout << "Map evolution complete!\n" ;
}

double MAPElitesRover::PercentageFilled(){
  vector<bool> fLog = bpMap->GetFilledLog() ;
  double filled = 0.0 ;
  for (size_t i = 0; i < fLog.size(); i++)
    if (fLog[i])
      filled += 1.0 ;
  filled /= ((double)fLog.size()) ;
  return filled ;
}

double MAPElitesRover::BestPerformance(NeuralNet * bestNN, VectorXd & bVec){
  vector<double> pLog = bpMap->GetPerformanceLog() ;
  double maxEval = 0.0 ;
  size_t maxInd = 0 ;
  for (size_t i = 0; i < pLog.size(); i++){
    if (pLog[i] > maxEval){
      maxEval = pLog[i] ;
      maxInd = i ;
    }
  }
  bVec = bpMap->GetBehaviour(maxInd) ;
  if (maxEval > 0.0){
    bestNN->SetWeights(bpMap->GetNeuralNet(maxInd)->GetWeightsA(),bpMap->GetNeuralNet(maxInd)->GetWeightsB()) ;
    return maxEval ;
  }
  std::cout << "No controllers achieved positive performance!\n" ;
  return maxEval ;
}

// Initial simulation parameters, includes setting initial rover position, POI positions and values
void MAPElitesRover::InitialiseSimulationWorld(){
  // Clear initial world properties
  initialXY.setZero(initialXY.size(),1) ;
  POIs.clear() ;
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
    double v = rand_interval(minPOIVal,maxPOIVal) ; // value
    POIs.push_back(Target(xy,v)) ;
    maxPossibleEval += v ; // compute maximum achievable performance
  }
}

double MAPElitesRover::SimulateController(NeuralNet * NN, VectorXd & bVec, bool write){
  // Write POI configuration to file
  if (write && outputTraj)
    for (size_t i = 0; i < numPOIs; i++)
      POIFile << POIs[i].GetLocation()[0] << "," << POIs[i].GetLocation()[1] << "," << POIs[i].GetValue() << "\n" ;
  
  // Initialise behaviour vector
  for (int i = 0; i < bVec.size(); i++)
    bVec(i) = 0.0 ;
  
  Vector2d xy = initialXY ;
  double psi = initialPsi ;
  
  // Write current global state to file
  if (write && outputTraj)
    trajFile << xy(0) << "," << xy(1) << "," << psi << "\n" ;
  
  for (size_t t = 0; t < nSteps; t++){
    // Calculate body frame NN input state
    VectorXd s = ComputeNNInput(xy, psi) ;
    
    // Calculate body frame action
    VectorXd a = NN->EvaluateNN(s).normalized() ;
    
    // Transform to global frame
    Matrix2d Body2Global = RotationMatrix(psi) ;
    Vector2d deltaXY = Body2Global*a ;
    double deltaPsi = atan2(a(1),a(0)) ;
    
    // Compute action motion quadrant for behaviour vector
//    ComputeBehaviourActions(bVec, deltaPsi) ;
    
    // Compute observation thresholds for behaviour vector
    ComputeBehaviourObservations(bVec, s) ;
    
    // Move
    xy += deltaXY ;
    psi += deltaPsi ;
    psi = pi_2_pi(psi) ;
    
    // Compute observations
    for (size_t j = 0; j < numPOIs; j++)
      POIs[j].ObserveTarget(xy) ;
    
    // Write current global state to file
    if (write && outputTraj)
      trajFile << xy(0) << "," << xy(1) << "," << psi << "\n" ;
  }
  
  // Evaluate NN
  double eval = 0.0 ;
  for (size_t j = 0; j < numPOIs; j++){
    eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
    POIs[j].ResetTarget() ;
  }
  
  eval /= maxPossibleEval ;
  
  return eval ;
}

// Wrapper for writing final trajectories to specified files
void MAPElitesRover::OutputTrajectories(char * A, char * B){
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

// Wrapper for writing behaviour performance map to binary file
void MAPElitesRover::WriteToBinary(char * bpName, char * pName, char * fName){
  bpMap->WriteBPMapBinary(bpName) ;
  bpMap->WritePerformanceBinary(pName) ;
  bpMap->WriteVisitedBinary(fName) ;
}

// Wrapper for reading behaviour performance map from binary file
void MAPElitesRover::ReadFromBinary(char * bpName, char * pName, char * fName){
  bpMap->ReadBPMapBinary(bpName) ;
  bpMap->ReadPerformanceBinary(pName) ;
  bpMap->ReadVisitedBinary(fName) ;
}

// Compute the NN input state given the rover location and the POI locations and values in the world
VectorXd MAPElitesRover::ComputeNNInput(Vector2d xy, double psi){
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

Matrix2d MAPElitesRover::RotationMatrix(double psi){
  Matrix2d R ;
  R(0,0) = cos(psi) ;
  R(0,1) = -sin(psi) ;
  R(1,0) = sin(psi) ;
  R(1,1) = cos(psi) ;
  return R ;
}

void MAPElitesRover::ComputeBehaviourActions(VectorXd & bVec, double delta){
  // Compute action motion quadrant for behaviour vector
  size_t q ;
  if (delta >= PI/2.0)
    q = 3 ;
  else if (delta >= 0.0)
    q = 0 ;
  else if (delta >= -PI/2.0)
    q = 1 ;
  else
    q = 2 ;
  bVec(q) += 1.0/((double) nSteps) ;
}

void MAPElitesRover::ComputeBehaviourObservations(VectorXd & bVec, VectorXd s){
  double thresholdObsVal = (maxPOIVal+minPOIVal)/2.0 ; // average POI value
  thresholdObsVal *= numPOIs ; // scaled by number of POIs
  thresholdObsVal /= ((double)s.size()) ; // divided by number of observation quadrants
  thresholdObsVal /= max(worldLimits[1]/2.0,worldLimits[3]/2.0) ; // divided by half the longest length scale of world
  for (int i = 0; i < bVec.size(); i++)
    if (s(i) >= thresholdObsVal)
      bVec(i) += 1.0/((double) nSteps) ;
}
