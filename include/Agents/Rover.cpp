#include "Rover.h"

Rover::Rover(size_t n, size_t nPop, string evalFunc): nSteps(n), popSize(nPop){
  numIn = 8 ; // hard coded for 4 element input (body frame quadrant decomposition)
  numOut = 2 ; // hard coded for 2 element output [dx,dy]
  numHidden = 16 ;
  RoverNE = new NeuroEvo(numIn, numOut, numHidden, nPop) ;
  
  if (evalFunc.compare("D") == 0)
    isD = true ;
  else if (evalFunc.compare("G") == 0)
    isD = false ;
  else{
    std::cout << "ERROR: Unknown evaluation function type [" << evalFunc << "], setting to global evaluation!\n" ;
    isD = false ;
  }
  windowSize = nSteps/10 ; // hardcoded running average window size to be 1/10 of full experimental run
  rThreshold.push_back(0.01) ;
  rThreshold.push_back(0.3) ; // hardcoded reward threshold, D logs for 1000 step executions suggest this is a good value
  pomdpAction = 0 ; // initial action is always to not ask for help
  stateObsUpdate = false ; // true if human assistance has redefined NN control policy state calculation
}

Rover::~Rover(){
  delete(RoverNE) ;
  RoverNE = 0 ;
}
  
void Rover::ResetEpochEvals(){
  // Re-initialise size of evaluations vector
  vector<double> evals(2*popSize,0) ;
  epochEvals = evals ;
  epochG = evals ;
}

// Initial simulation parameters, includes setting initial rover position, POI positions and values, and clearing the evaluation storage vector
void Rover::InitialiseNewLearningEpoch(vector<Target> pois, Vector2d xy, double psi){
  // Clear initial world properties
  initialXY.setZero(initialXY.size(),1) ;
  POIs.clear() ;
  
  ResetStepwiseEval() ;
  
  for (size_t i = 0; i < pois.size(); i++){
    POIs.push_back(pois[i]) ;
//    maxPossibleEval += POIs[i].GetValue() ;
  }
  
  initialXY(0) = xy(0) ;
  initialXY(1) = xy(1) ;
  initialPsi = psi ;
  
  currentXY = initialXY ;
  currentPsi = initialPsi ;
  
  // Reinitialise expertise POMDP properties
  pomdpAction = 0 ;
  stateObsUpdate = false ;
}

void Rover::ResetStepwiseEval(){
  stepwiseD = 0.0 ;
  runningAvgR.clear() ;
}

Vector2d Rover::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState){
  // Calculate body frame NN input state
  VectorXd s ;
  if (!stateObsUpdate)
    s = ComputeNNInput(jointState) ;
  else{
    vector<Vector2d> tempState ;
    tempState.clear() ;
    s = ComputeNNInput(tempState) ;
  }
  
  // Calculate body frame action
  VectorXd a = RoverNE->GetNNIndex(i)->EvaluateNN(s).normalized() ;
  
  // Transform to global frame
  Matrix2d Body2Global = RotationMatrix(currentPsi) ;
  Vector2d deltaXY = Body2Global*a ;
  double deltaPsi = atan2(a(1),a(0)) ;
  
  // Move
  currentXY += deltaXY ;
  currentPsi += deltaPsi ;
  currentPsi = pi_2_pi(currentPsi) ;
  
  return currentXY ;
}

void Rover::ComputeStepwiseEval(vector<Vector2d> jointState, double G){
  if (!stateObsUpdate)
    DifferenceEvaluationFunction(jointState, G) ;
  else
    UpdatedStateEvaluationFunction(jointState, G) ;
}

void Rover::SetEpochPerformance(double G, size_t i){
  if (isD)
    epochEvals[i] = stepwiseD ;
  else
    epochEvals[i] = G ;
  epochG[i] = G ;
}

void Rover::EvolvePolicies(bool init){
  // Determine whether or not to evolve
  if (evalLearning){
    if (init){
      RoverNE->MutatePopulation() ;
      isLearn = true ;
    }
    else{
      deltaPi = RoverNE->GetMutationNorm() ;
      deltaR.clear() ;
      dRdPi.clear() ;
      sumdRdPi = 0.0 ;
      for (size_t i = 0; i < popSize; i++){
//        deltaR.push_back(fabs(epochEvals[i]-epochEvals[i+popSize])) ;
        deltaR.push_back(fabs(epochG[i]-epochG[i+popSize])) ;
        dRdPi.push_back(deltaR[i]/deltaPi[i]) ;
//        sumdRdPi += dRdPi[i] ;
        if (dRdPi[i] > sumdRdPi)
          sumdRdPi = dRdPi[i] ;
      }
      double p = rand_interval(0.0, 1.0) ;
      double pLearn = 1.0 - exp(-sumdRdPi/tau) ;
      if (p < pLearn){
        isLearn = true ;
        RoverNE->EvolvePopulation(epochEvals) ;
        RoverNE->MutatePopulation() ;
      }
      else{
        isLearn = false ;
      }
    }
  }
  else{
    if (!init)
      RoverNE->EvolvePopulation(epochEvals) ;
    RoverNE->MutatePopulation() ;
  }
}

void Rover::OutputNNs(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  std::ofstream NNFile ;
  NNFile.open(fileName.str().c_str(),std::ios::app) ;
  
  // Only write in non-mutated (competitive) policies
  for (size_t i = 0; i < popSize; i++){
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

void Rover::SetPOMDPPolicy(POMDP * pomdp){
  expertisePOMDP = pomdp ;
  belief = expertisePOMDP->GetBelief() ;
}

size_t Rover::ComputePOMDPAction(){
  // Wait for sufficient reward observations
  if (runningAvgR.size() == windowSize){
    // Calculate reward observation from running average
    double avgSum = GetAverageR() ;
    
    // Convert to discrete observation for POMDP interface
    size_t obs ;
    if (avgSum <= rThreshold[0])
      obs = 0 ;
    else if (avgSum < rThreshold[1])
      obs = 1 ;
    else
      obs = 2 ;
    
    // Updated POMDP with latest observation
    expertisePOMDP->UpdateBelief(pomdpAction, obs) ;
    belief = expertisePOMDP->GetBelief() ;
    
    // Compute next action
    pomdpAction = expertisePOMDP->GetBestAction() ;
  }
  return pomdpAction ;
}

double Rover::GetAverageR(){
  // Calculate reward observation from running average
  double avgSum = 0.0 ;
  for (list<double>::iterator it=runningAvgR.begin(); it!=runningAvgR.end(); ++it)
    avgSum += *it ;
  
  avgSum /= windowSize ;
  return avgSum ;
}

void Rover::SetLearningEvaluation(double t, bool b){
  evalLearning = b ;
  if (evalLearning){
    RoverNE->SetMutationNormLog() ; // log the Frobenius norm of mutated weight matrices
    tau = t ;
  }
}

void Rover::OutputImpact(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  std::ofstream IFile ;
  IFile.open(fileName.str().c_str(),std::ios::app) ;
  
  // Write in change in reward
  for (size_t i = 0; i < deltaR.size(); i++){
    IFile << deltaR[i] << "," ;
  }
  IFile << "\n" ;
  // Write in change in policy
  for (size_t i = 0; i < deltaPi.size(); i++){
    IFile << deltaPi[i] << "," ;
  }
  IFile << "\n" ;
  // Write in ratio of change
  for (size_t i = 0; i < dRdPi.size(); i++){
    IFile << dRdPi[i] << "," ;
  }
  IFile << "\n" ;
  IFile << sumdRdPi << "," << isLearn << "\n" ;
  IFile.close() ;
}

void Rover::UpdateNNStateInputCalculation(bool update, size_t gID){
  stateObsUpdate = update ;
  goalPOI = gID ;
  vector<Target> newPOIs ;
  newPOIs.push_back(POIs[gID]) ;
  POIs.clear() ;
  POIs.push_back(newPOIs[0]) ; // remove all other POIs from consideration in the state
  runningAvgR.clear() ; // restart running average calculation window
  pomdpAction = 0 ; // reset pomdp action
}

// Compute the NN input state given the rover locations and the POI locations and values in the world
VectorXd Rover::ComputeNNInput(vector<Vector2d> jointState){
  VectorXd s ;
  s.setZero(numIn,1) ;
  MatrixXd Global2Body = RotationMatrix(-currentPsi) ;
  
  // Compute POI observation states
  Vector2d POIv ;
  POIv.setZero(2,1) ;
  for (size_t i = 0; i < POIs.size(); i++){
    POIv = POIs[i].GetLocation() - currentXY ;
    Vector2d POIbody = Global2Body*POIv ;
    Vector2d diff = currentXY - POIbody ;
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

  // Compute rover observation states
  size_t ind = 0 ; // stores agent's index in the joint state
  double minDiff = DBL_MAX ;
  for (size_t i = 0; i < jointState.size(); i++){
    double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
    if (diff < minDiff){
      minDiff = diff ;
      ind = i ;
    }
  }
  
  Vector2d rovV ;
  rovV.setZero(2,1) ;
  for (size_t i = 0; i < jointState.size(); i++){
    if (i != ind){
      rovV = jointState[i] - currentXY ;
      Vector2d rovBody = Global2Body*rovV ;
      Vector2d diff = currentXY - rovBody ;
      double d = diff.norm() ;
      double theta = atan2(rovBody(1),rovBody(0)) ;
      size_t q ;
      if (theta >= PI/2.0)
        q = 7 ;
      else if (theta >= 0.0)
        q = 4 ;
      else if (theta >= -PI/2.0)
        q = 5 ;
      else
        q = 6 ;
      s(q) += 1.0/max(d,1.0) ;
    }
  }
  
  return s ;
}

Matrix2d Rover::RotationMatrix(double psi){
  Matrix2d R ;
  R(0,0) = cos(psi) ;
  R(0,1) = -sin(psi) ;
  R(1,0) = sin(psi) ;
  R(1,1) = cos(psi) ;
  return R ;
}

void Rover::DifferenceEvaluationFunction(vector<Vector2d> jointState, double G){
  double G_hat = 0.0 ;
  size_t ind = 0 ; // stores agent's index in the joint state
  double minDiff = DBL_MAX ;
  for (size_t i = 0; i < jointState.size(); i++){
    double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
    if (diff < minDiff){
      minDiff = diff ;
      ind = i ;
    }
  }
  
  // Replace agent state with counterfactual
  jointState[ind](0) = initialXY(0) ;
  jointState[ind](1) = initialXY(1) ;
  for (size_t i = 0; i < jointState.size(); i++)
    for (size_t j = 0; j < POIs.size(); j++)
      POIs[j].ObserveTarget(jointState[i]) ;
       
  for (size_t j = 0; j < POIs.size(); j++){
    G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
    POIs[j].ResetTarget() ;
  }
  
  stepwiseD += (G-G_hat) ;
  if (runningAvgR.size() == windowSize)
    runningAvgR.pop_front() ;
  runningAvgR.push_back(G) ;
}

void Rover::UpdatedStateEvaluationFunction(vector<Vector2d> jointState, double G){
  double G_hat = 0.0 ;
  size_t ind = 0 ; // stores agent's index in the joint state
  double minDiff = DBL_MAX ;
  for (size_t i = 0; i < jointState.size(); i++){
    double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
    if (diff < minDiff){
      minDiff = diff ;
      ind = i ;
    }
  }
  vector<Vector2d> newState ; // ignore effect of all other agents in state
  newState.push_back(jointState[ind]) ;
  
  // Replace agent state with counterfactual
  newState[0](0) = initialXY(0) ;
  newState[0](1) = initialXY(1) ;
  for (size_t i = 0; i < newState.size(); i++)
    for (size_t j = 0; j < POIs.size(); j++)
      POIs[j].ObserveTarget(newState[i]) ;
       
  for (size_t j = 0; j < POIs.size(); j++){
    G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
    POIs[j].ResetTarget() ;
  }
  
  stepwiseD += (G-G_hat) ;
  if (runningAvgR.size() == windowSize)
    runningAvgR.pop_front() ;
  runningAvgR.push_back(G) ;
}
