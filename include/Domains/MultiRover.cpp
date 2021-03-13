#include "MultiRover.h"

MultiRover::MultiRover(vector<double> wLims, size_t numSteps, size_t numPop, size_t numPOIs, string evalFunc, size_t rovs, int c): world(wLims), nSteps(numSteps), nPop(numPop), nPOIs(numPOIs), evaluationFunction(evalFunc), nRovers(rovs), coupling(c){
  for (size_t i = 0; i < nRovers; i++){
    Rover * newRover = new Rover(nSteps, nPop, evaluationFunction) ;
    roverTeam.push_back(newRover) ;
  }
  outputEvals = false ;
  outputTrajs = false ;
  outputQury = false ;
  outputBlf = false ;
  gPOIObs = false ; // true if goal POI is observed
}

MultiRover::~MultiRover(){
  for (size_t i = 0; i < nRovers; i++){
    delete roverTeam[i] ;
    roverTeam[i] = 0 ;
  }
  if (outputEvals)
    evalFile.close() ;
  if (outputTrajs){
    trajFile.close() ;
    POIFile.close() ;
  }
}

void MultiRover::InitialiseEpoch(){
  double rangeX = world[1] - world[0] ;
  double rangeY = world[3] - world[2] ;
  
  initialXYs.clear() ;
  initialPsis.clear() ;
  for (size_t i = 0; i < nRovers; i++){
    Vector2d initialXY ;
    initialXY(0) = rand_interval(world[0]+rangeX/3.0,world[1]-rangeX/3.0) ;
    initialXY(1) = rand_interval(world[2]+rangeY/3.0,world[3]-rangeX/3.0) ;
    double initialPsi = rand_interval(-PI,PI) ;
    initialXYs.push_back(initialXY) ;
    initialPsis.push_back(initialPsi) ;
  }
  
  // POI locations and values in global frame (restricted to within outer regions of 9 grid)
  POIs.clear() ;
  for (size_t p = 0; p < nPOIs; p++){
    Vector2d xy ;
    double x, y ;
    bool accept = false ;
    while (!accept){
      x = rand_interval(world[0],world[1]) ;
      y = rand_interval(world[2],world[3]) ;
      if (x > world[0]+rangeX/3.0 && x < world[1]-rangeX/3.0 && y > world[2]+rangeY/3.0 && y < world[3]-rangeX/3.0) {}
      else accept = true ;
    }
    xy(0) = x ; // x location
    xy(1) = y ; // y location
    double v = rand_interval(1,10) ; // value
    if (coupling > 1)
      POIs.push_back(Target(xy,v,coupling)) ;
    else
      POIs.push_back(Target(xy,v)) ;
  }
}

vector< vector<size_t> > MultiRover::RandomiseTeams(size_t n){
  vector< vector<size_t> > teams ;
  vector<size_t> order ;
  for (size_t i = 0; i < n; i++)
    order.push_back(i) ;
  
  // TODO: UNCOMMENT LINES 72, 75 TO REINTRODUCE TEAM SHUFFLING
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;

  for (size_t j = 0; j < nRovers; j++){
    shuffle (order.begin(), order.end(), std::default_random_engine(seed)) ;
    teams.push_back(order) ;
  }
  
  return teams ;
}

void MultiRover::SimulateEpoch(bool train){
  size_t teamSize ;
  if (train)
    teamSize = 2*nPop ;
  else
    teamSize = nPop ;
  
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; // each row is the population for a single agent
  
  if (outputTrajs)
    for (size_t i = 0; i < nPOIs; i++)
      POIFile << POIs[i].GetLocation()(0) << "," << POIs[i].GetLocation()(1) << "," << POIs[i].GetValue() << "\n" ;
  
  double maxEval = 0.0 ;
  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
    // Initialise world and reset rovers and POIs
    vector<Vector2d> jointState ;
    for (size_t j = 0; j < nRovers; j++){
      roverTeam[j]->InitialiseNewLearningEpoch(POIs,initialXYs[j],initialPsis[j]) ;
      jointState.push_back(initialXYs[j]) ;
    }
    if (outputTrajs){
      for (size_t i = 0; i < jointState.size(); i++)
        trajFile << jointState[i](0) << "," << jointState[i](1) << "," ;
      trajFile << "\n" ;
    }
    
    // Create tempPOIs variable to compute stepwise G
    vector<Target> tempPOIs ;
    tempPOIs = POIs ;
    
    // Store summation of stepwise G
    double sumG = 0.0 ;
    for (size_t t = 0; t < nSteps; t++){
      vector<Vector2d> newJointState ;
      double G = 0.0 ;
      for (size_t j = 0; j < nRovers; j++){ // looping down the rows of 'teams'
        // Step forward rover j based on team XY state and POI locations, store new xy location
        Vector2d xy = roverTeam[j]->ExecuteNNControlPolicy(teams[j][i],jointState) ;
        newJointState.push_back(xy) ;
        
        // Update recorded POI observations
        for (size_t k = 0; k < POIs.size(); k++){
          if (coupling > 1){
//            std:: cout << "Timestep: " << t << ", POI number: " << k << "\n" ;
            POIs[k].ObserveTarget(xy,t) ;
            tempPOIs[k].ObserveTarget(xy,t) ;
          }
          else{
            POIs[k].ObserveTarget(xy) ;
            tempPOIs[k].ObserveTarget(xy) ;
          }
        }
      }
      // Compute stepwise G
      for (size_t k = 0; k < POIs.size(); k++){
        G += tempPOIs[k].IsObserved() ? (tempPOIs[k].GetValue()/max(tempPOIs[k].GetNearestObs(),1.0)) : 0.0 ;
        tempPOIs[k].ResetTarget() ;
      }
      sumG += G ;

      // Compute stepwiseD
      for (size_t j = 0; j < nRovers; j++)
        roverTeam[j]->ComputeStepwiseEval(newJointState, G) ;
      
      // Increment stored joint state
      jointState.clear() ;
      for (size_t j = 0; j < nRovers; j++)
        jointState.push_back(newJointState[j]) ;
      
      if (outputTrajs){
        for (size_t j = 0; j < jointState.size(); j++)
          trajFile << jointState[j](0) << "," << jointState[j](1) << "," ;
        trajFile << "\n" ;
      }
      
      if (outputAvgStepR){
        for (size_t j = 0; j < nRovers; j++)
          avgStepRFile << roverTeam[j]->GetAverageR() << "," ;
        avgStepRFile << "\n" ;
      }
    }
    // Compute overall team performance
    double eval = 0.0 ;
    for (size_t j = 0; j < nPOIs; j++){
      eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
      POIs[j].ResetTarget() ;
    }
    
    // Compute overall team performance as sum over time
//    eval = sumG ;
    
    // Store maximum team performance
    if (eval > maxEval)
      maxEval = eval ;
    
    // Assign fitness
    for (size_t j = 0; j < nRovers; j++)
      roverTeam[j]->SetEpochPerformance(eval, teams[j][i]) ;
    
    // Output to file
    if (outputEvals)
      evalFile << eval << "," ;
  }
  
  if (outputEvals)
    evalFile << "\n" ;
  
  // Print best team performance
  std::cout << "max achieved value: " << maxEval << "...\n" ;
}

void MultiRover::SimulateEpoch(size_t goalPOI, char * env, char * policy, VectorXd prior){
  size_t teamSize ;
  teamSize = nPop ; // inquiry simulation is always in test mode
  
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; // each row is the population for a single agent
  
  if (outputTrajs)
    for (size_t i = 0; i < nPOIs; i++)
      POIFile << POIs[i].GetLocation()(0) << "," << POIs[i].GetLocation()(1) << "," << POIs[i].GetValue() << "\n" ;
  
  double maxEval = 0.0 ;
  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
    // Initialise world and reset rovers
    gPOIObs = false ; // reset goal POI observation flag
    vector<Vector2d> jointState ;
    for (size_t j = 0; j < nRovers; j++){
      roverTeam[j]->InitialiseNewLearningEpoch(POIs,initialXYs[j],initialPsis[j]) ;
      jointState.push_back(initialXYs[j]) ;
      
      // Create new expertise POMDP
      roverTeam[j]->SetPOMDPPolicy(new POMDP(env, policy, prior)) ;
    }
    if (outputTrajs){
      for (size_t i = 0; i < jointState.size(); i++)
        trajFile << jointState[i](0) << "," << jointState[i](1) << "," ;
      trajFile << "\n" ;
    }
    
    // Create tempPOIs variable to compute stepwise G
    vector<Target> tempPOIs ;
    tempPOIs = POIs ;
    
    for (size_t t = 0; t < nSteps; t++){
      vector<Vector2d> newJointState ;
      double G = 0.0 ;
      for (size_t j = 0; j < nRovers; j++){ // looping down the rows of 'teams'
        // Step forward rover j based on team XY state and POI locations, store new xy location
        Vector2d xy = roverTeam[j]->ExecuteNNControlPolicy(teams[j][i],jointState) ;
        newJointState.push_back(xy) ;
        
        // Update recorded POI observations
        for (size_t k = 0; k < POIs.size(); k++){
          if (coupling > 1){
            POIs[k].ObserveTarget(xy,t) ;
            tempPOIs[k].ObserveTarget(xy,t) ;
          }
          else{
            POIs[k].ObserveTarget(xy) ;
            tempPOIs[k].ObserveTarget(xy) ;
          }
        }
      }
      // Compute stepwise G
      if (!gPOIObs){ // goal POI has not been observed
        for (size_t k = 0; k < POIs.size(); k++){
          G += tempPOIs[k].IsObserved() ? (tempPOIs[k].GetValue()/max(tempPOIs[k].GetNearestObs(),1.0)) : 0.0 ;
          tempPOIs[k].ResetTarget() ;
        }
      }
      else{ // goal POI has been observed, all other POIs have no value
        G = tempPOIs[goalPOI].IsObserved() ? (tempPOIs[goalPOI].GetValue()/max(tempPOIs[goalPOI].GetNearestObs(),1.0)) : 0.0 ;
        tempPOIs[goalPOI].ResetTarget() ;
      }
      
      // Compute stepwiseD and output expertise POMDP action
      for (size_t j = 0; j < nRovers; j++){
        roverTeam[j]->ComputeStepwiseEval(newJointState, G) ;
        size_t pomdpAct = roverTeam[j]->ComputePOMDPAction() ; // TODO: commented for testing changes in stepwise G
//        size_t pomdpAct = 0 ; // TODO: hard-coded for testing changes in stepwise G, use for comparative results
        
        if (outputQury)
          quryFile << pomdpAct << "," ;
        
        if (outputBlf)
          blfFile << roverTeam[j]->GetPOMDPBelief()(0) << "," << roverTeam[j]->GetPOMDPBelief()(1) << "," ;
        
        if (pomdpAct == 1) // help requested
          if (gPOIObs && !(roverTeam[j]->IsStateObsUpdated())) // goal POI has been observed by team and current rover has not updated state
            roverTeam[j]->UpdateNNStateInputCalculation(gPOIObs,goalPOI) ;
      }
      
      if (outputQury)
        quryFile << "\n" ;
      
      if (outputBlf)
        blfFile << "\n" ;
      
      // Increment stored joint state
      jointState.clear() ;
      for (size_t j = 0; j < nRovers; j++)
        jointState.push_back(newJointState[j]) ;
      
      if (outputTrajs){
        for (size_t i = 0; i < jointState.size(); i++)
          trajFile << jointState[i](0) << "," << jointState[i](1) << "," ;
        trajFile << "\n" ;
      }
      
      if (outputAvgStepR){
        for (size_t j = 0; j < nRovers; j++)
          avgStepRFile << roverTeam[j]->GetAverageR() << "," ;
        avgStepRFile << "\n" ;
      }
      
      // Check if goal POI has been discovered
      if (!gPOIObs && POIs[goalPOI].IsObserved())
        gPOIObs = true ;
    }
    // Compute overall team performance. TODO: update such that once goal POI is observed, reward for observing goal POI is cumulative
    double eval = 0.0 ;
    for (size_t j = 0; j < nPOIs; j++){
      eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
      POIs[j].ResetTarget() ;
    }
    
    // Store maximum team performance
    if (eval > maxEval)
      maxEval = eval ;
    
    // Assign fitness and reset POMDPs
    for (size_t j = 0; j < nRovers; j++){
      roverTeam[j]->SetEpochPerformance(eval, teams[j][i]) ;
      delete roverTeam[j]->GetPOMDPPolicy() ;
    }
    
    // Output to file
    if (outputEvals)
      evalFile << eval << "," ;
  }
  
  if (outputEvals)
    evalFile << "\n" ;
  
  if (outputQury)
    quryFile.close() ;
  
  if (outputBlf)
    blfFile.close() ;
  
  // Print best team performance
  std::cout << "max achieved value: " << maxEval << "...\n" ;
}

void MultiRover::EvolvePolicies(bool init){
  for (size_t i = 0; i < nRovers; i++)
    roverTeam[i]->EvolvePolicies(init) ;
}

void MultiRover::ResetEpochEvals(){
  for (size_t i = 0; i < nRovers; i++)
    roverTeam[i]->ResetEpochEvals() ;
}

void MultiRover::SetLearningEvaluation(double tau){
  for (size_t i = 0; i < nRovers; i++)
    roverTeam[i]->SetLearningEvaluation(tau) ;
}

// Wrapper for writing epoch evaluations to specified files
void MultiRover::OutputPerformance(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (evalFile.is_open())
    evalFile.close() ;
  evalFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputEvals = true ;
}

// Wrapper for writing final trajectories to specified files
void MultiRover::OutputTrajectories(char * A, char * B){
	// Filename to write trajectories to stored in A
	std::stringstream tfileName ;
  tfileName << A ;
  if (trajFile.is_open())
    trajFile.close() ;
  trajFile.open(tfileName.str().c_str(),std::ios::app) ;
  
  // Filename to write POIs to stored in B
	std::stringstream pfileName ;
  pfileName << B ;
  if (POIFile.is_open())
    POIFile.close() ;
  POIFile.open(pfileName.str().c_str(),std::ios::app) ;
  
  outputTrajs = true ;
}

// Wrapper for writing final control policies to specified file
void MultiRover::OutputControlPolicies(char * A){
  for (size_t i = 0; i < nRovers; i++)
    roverTeam[i]->OutputNNs(A) ;
}

// Wrapper for writing POMDP actions to specified file
void MultiRover::OutputQueries(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (quryFile.is_open())
    quryFile.close() ;
  quryFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputQury = true ;
}

// Wrapper for writing POMDP beliefs to specified file
void MultiRover::OutputBeliefs(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (blfFile.is_open())
    blfFile.close() ;
  blfFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputBlf = true ;
}

// Wrapper for writing running average of stepwiseD to specified file
void MultiRover::OutputAverageStepwise(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (avgStepRFile.is_open())
    avgStepRFile.close() ;
  avgStepRFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputAvgStepR = true ;
}

// Wrapper for writing agent impact values to specified file
void MultiRover::OutputImpacts(char * A){
  int buffSize = 100 ;
  char iFile[buffSize] ;
  for (size_t i = 0; i < nRovers; i++){
    sprintf(iFile,"%s%i.txt",A,i) ; // each rover agent writes to unique impact logger
    roverTeam[i]->OutputImpact(iFile) ;
  }
}

// Wrapper for writing IDs of learning agents to specified file
void MultiRover::OutputLearners(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  std::ofstream LFile ;
  LFile.open(fileName.str().c_str(),std::ios::app) ;
  
  for (size_t i = 0; i < nRovers; i++){
    LFile << roverTeam[i]->GetIsLearn() << "," ;
  }
  LFile << "\n" ;
}

void MultiRover::ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char* storeEval, size_t numIn, size_t numOut, size_t numHidden){
  // Filename to read NN control policy
	std::stringstream fileName ;
  fileName << readFile ;
  std::ifstream nnFile ;
  
  vector<NeuralNet *> loadedNN ;
  std::cout << "Reading out " << nPop << " NN control policies for each rover to test...\n" ;
  nnFile.open(fileName.str().c_str(),std::ios::in) ;
  
  // Read in all NN weight matrices
  std::string line ;
  MatrixXd NNA ;
  MatrixXd NNB ;
  NNA.setZero(numIn,numHidden) ;
  NNB.setZero(numHidden+1,numOut) ;
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
  
  // Assign control policies to rovers ;
  k = 0 ;
  for (size_t i = 0; i < nRovers; i++){
    NeuroEvo * rovNE = roverTeam[i]->GetNEPopulation() ;
    for (size_t j = 0; j < nPop; j++){
      rovNE->GetNNIndex(j)->SetWeights(loadedNN[k]->GetWeightsA(),loadedNN[k]->GetWeightsB()) ;
      k++ ;
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI) ;
  ResetEpochEvals() ;
  SimulateEpoch(false) ; // simulate in test mode
  
  for (size_t i = 0; i < loadedNN.size(); i++){
    delete loadedNN[i] ;
    loadedNN[i] = 0 ;
  }
}

void MultiRover::ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char * storeEval, char * storeQury, char * storeBlf, size_t numIn, size_t numOut, size_t numHidden, size_t goalPOI, char * pomdpEnv, char * pomdpPolicy, VectorXd prior){
  // Filename to read NN control policy
	std::stringstream fileName ;
  fileName << readFile ;
  std::ifstream nnFile ;
  
  vector<NeuralNet *> loadedNN ;
  std::cout << "Reading out " << nPop << " NN control policies for each rover to test...\n" ;
  nnFile.open(fileName.str().c_str(),std::ios::in) ;
  
  // Read in all NN weight matrices
  std::string line ;
  MatrixXd NNA ;
  MatrixXd NNB ;
  NNA.setZero(numIn,numHidden) ;
  NNB.setZero(numHidden+1,numOut) ;
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
  
  // Assign control policies to rovers ;
  k = 0 ;
  for (size_t i = 0; i < nRovers; i++){
    NeuroEvo * rovNE = roverTeam[i]->GetNEPopulation() ;
    for (size_t j = 0; j < nPop; j++){
      rovNE->GetNNIndex(j)->SetWeights(loadedNN[k]->GetWeightsA(),loadedNN[k]->GetWeightsB()) ;
      k++ ;
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI) ;
  OutputQueries(storeQury) ;
  OutputBeliefs(storeBlf) ;
  ResetEpochEvals() ;
  std::cout << "Simulating rover team...\n" ;
  SimulateEpoch(goalPOI, pomdpEnv, pomdpPolicy, prior) ; // simulate in test mode
  
  for (size_t i = 0; i < loadedNN.size(); i++){
    delete loadedNN[i] ;
    loadedNN[i] = 0 ;
  }
}

void MultiRover::ExecutePolicies(char * expFile, char * novFile, char * storeTraj, char * storePOI, char* storeEval, size_t numIn, size_t numOut, size_t numHidden){
  // Filename to read expert NN control policies
	std::stringstream expFileName ;
  expFileName << expFile ;
  std::ifstream expNNFile ;
  
  vector<NeuralNet *> expLoadedNN ;
  std::cout << "Reading out " << nPop << " expert NN control policies for each rover to test...\n" ;
  expNNFile.open(expFileName.str().c_str(),std::ios::in) ;
  
  // Read in all NN weight matrices
  std::string eline ;
  MatrixXd NNA ;
  MatrixXd NNB ;
  NNA.setZero(numIn,numHidden) ;
  NNB.setZero(numHidden+1,numOut) ;
  int nnK = NNA.rows() + NNB.rows() ; // number of lines corresponding to a single control policy
  int k = 0 ; // track line number
  while (std::getline(expNNFile,eline)){
    std::stringstream lineStream(eline) ;
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
      expLoadedNN.push_back(newNN) ;
    }
    k++ ;
  }
  expNNFile.close() ;
  
  // Filename to read novive NN control policies
	std::stringstream novFileName ;
  novFileName << novFile ;
  std::ifstream novNNFile ;
  
  vector<NeuralNet *> novLoadedNN ;
  std::cout << "Reading out " << nPop << " novice NN control policies for each rover to test...\n" ;
  novNNFile.open(novFileName.str().c_str(),std::ios::in) ;
  
  // Read in all NN weight matrices
  std::string nline ;
  NNA.setZero(numIn,numHidden) ;
  NNB.setZero(numHidden+1,numOut) ;
  nnK = NNA.rows() + NNB.rows() ; // number of lines corresponding to a single control policy
  k = 0 ; // track line number
  while (std::getline(novNNFile,nline)){
    std::stringstream lineStream(nline) ;
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
      novLoadedNN.push_back(newNN) ;
    }
    k++ ;
  }
  novNNFile.close() ;
  
  // Assign control policies to rovers ;
  k = 0 ;
  for (size_t i = 0; i < nRovers; i++){
    NeuroEvo * rovNE = roverTeam[i]->GetNEPopulation() ;
    if (k == 0){
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(novLoadedNN[k]->GetWeightsA(),novLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
    else {
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(expLoadedNN[k]->GetWeightsA(),expLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI) ;
  ResetEpochEvals() ;
  SimulateEpoch(false) ; // simulate in test mode
  
  for (size_t i = 0; i < expLoadedNN.size(); i++){
    delete expLoadedNN[i] ;
    expLoadedNN[i] = 0 ;
  }
  for (size_t i = 0; i < novLoadedNN.size(); i++){
    delete novLoadedNN[i] ;
    novLoadedNN[i] = 0 ;
  }
}
