#include "MultiNightBar.h"

MultiNightBar::MultiNightBar(size_t numNights, size_t c, size_t numPop, string evalFunc, size_t agents): nNights(numNights), capacity(c), nPop(numPop), evaluationFunction(evalFunc), nAgents(agents){
  for (size_t i = 0; i < nAgents; i++){
    BarAgent * newAgent = new BarAgent(nPop, evaluationFunction, nNights) ;
    agentTeam.push_back(newAgent) ;
  }
  outputEvals = false ;
  outputActs = false ;
}

MultiNightBar::~MultiNightBar(){
  for (size_t i = 0; i < nAgents; i++){
    delete agentTeam[i] ;
    agentTeam[i] = 0 ;
  }
  if (outputEvals)
    evalFile.close() ;
  if (outputActs){
    actFile.close() ;
    barFile.close() ;
  }
}

void MultiNightBar::InitialiseEpoch(){
  // Initialise each night as a separate Bar object
  barNights.clear() ;
  for (size_t i = 0; i < nNights; i++){
    barNights.push_back(Bar(capacity)) ;
  }
}

void MultiNightBar::SimulateEpoch(bool train){
  size_t teamSize ;
  if (train)
    teamSize = 2*nPop ;
  else
    teamSize = nPop ;
  
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; // each row is the population for a single agent
  
  if (outputActs) // record the domain
    for (size_t i = 0; i < nNights; i++)
      barFile << barNights[i].GetCapacity() << "\n" ;
  
  double maxEval = 0.0 ;
  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
    // Initialise bar nights
    vector<int> jointState ;
    for (size_t j = 0; j < nAgents; j++){
      agentTeam[j]->InitialiseNewLearningEpoch(barNights) ;
    }
    
    for (size_t j = 0; j < nAgents; j++){ // looping down the rows of 'teams'
      // Compute action for agent j
      int a = agentTeam[j]->ExecuteNNControlPolicy(teams[j][i]) ;
      jointState.push_back(a) ;
    }
    
    // Compute G
    double G = 0.0 ;
    vector<size_t> barState(nNights, 0) ;
    for (size_t j = 0; j < jointState.size(); j++){
      // Count number of attendees per night
      barState[jointState[j]]++ ;
    }
    for (size_t j = 0; j < barNights.size(); j++){ // compute reward for each night and sum
      G += barNights[j].GetReward(barState[j]) ;
//      std::cout << "Night number: " << j << ", attendance: " << barState[j] << ", enjoyment: " << barNights[j].GetReward(barState[j]) << "\n" ;
    }
    
    // Compute D
    for (size_t j = 0; j < nAgents; j++)
      agentTeam[j]->ComputeEval(jointState, j, G) ;
    
    // Output actions
    if (outputActs){
      for (size_t j = 0; j < jointState.size(); j++)
        actFile << jointState[j] << "," ;
      actFile << "\n" ;
    }
    
    // Store maximum team performance
    if (G > maxEval)
      maxEval = G ;
    
    // Assign fitness
    for (size_t j = 0; j < nAgents; j++)
      agentTeam[j]->SetEpochPerformance(G, teams[j][i]) ;
    
    // Output to file
    if (outputEvals)
      evalFile << G << "," ;
  }
  
  if (outputEvals)
    evalFile << "\n" ;
  
  // Compute delta Pis - Only do this for training
  // For every agent, for every neural network in the first k agents, compute delta D / delta Pi
  if (train){

    for (size_t i = 0; i < nAgents; i++){
      
      BarAgent* currentAgent = agentTeam[i];
      // Pointer to currentAgent's Neural Networks
      NeuroEvo* currentAgentNNs = currentAgent->GetNEPopulation();

      for (size_t j = 0; j < nPop; j++){

        // Original NN indices go from 0 to nPop-1, Mutated NN indices go from nPop to 2*NPop
        // where 0's mutated Neural Network is at nPop, 1's at 1+nPoP etc.
        NeuralNet* originalNN = currentAgentNNs->GetNNIndex(j);
        NeuralNet* mutatedNN = currentAgentNNs->GetNNIndex(j+nPop);

        // Determine Pi for each of the neural networks
        VectorXd oneInput(1);
        oneInput(0) = 1;
        VectorXd originalOutput = originalNN->EvaluateNN(oneInput);
        VectorXd mutatedOutput = mutatedNN->EvaluateNN(oneInput);

        VectorXd diffVector = mutatedOutput - originalOutput;

        double deltaPi = diffVector.norm();
        // std::cout << "Delta Pi: " << deltaPi << std::endl;
      }
    }
  }

  // Print best team performance
  std::cout << "max achieved value: " << maxEval << "...\n" ;
}

void MultiNightBar::EvolvePolicies(bool init){
  for (size_t i = 0; i < nAgents; i++)
    agentTeam[i]->EvolvePolicies(init) ;
}

void MultiNightBar::ResetEpochEvals(){
  for (size_t i = 0; i < nAgents; i++)
    agentTeam[i]->ResetEpochEvals() ;
}

// Wrapper for writing epoch evaluations to specified files
void MultiNightBar::OutputPerformance(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (evalFile.is_open())
    evalFile.close() ;
  evalFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputEvals = true ;
}

// Wrapper for writing agent actions to specified files
void MultiNightBar::OutputActions(char * A, char * B){
	// Filename to write trajectories to stored in A
	std::stringstream afileName ;
  afileName << A ;
  if (actFile.is_open())
    actFile.close() ;
  actFile.open(afileName.str().c_str(),std::ios::app) ;
  
  // Filename to write bar configurations to stored in B
	std::stringstream bfileName ;
  bfileName << B ;
  if (barFile.is_open())
    barFile.close() ;
  barFile.open(bfileName.str().c_str(),std::ios::app) ;
  
  outputActs = true ;
}

// Wrapper for writing final control policies to specified file
void MultiNightBar::OutputControlPolicies(char * A){
  for (size_t i = 0; i < nAgents; i++)
    agentTeam[i]->OutputNNs(A) ;
}

void MultiNightBar::ExecutePolicies(char * readFile, char * storeActs, char * storeBar, char* storeEval, size_t numIn, size_t numOut, size_t numHidden){
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
  
  // Assign control policies to agents ;
  k = 0 ;
  for (size_t i = 0; i < nAgents; i++){
    NeuroEvo * agentNE = agentTeam[i]->GetNEPopulation() ;
    for (size_t j = 0; j < nPop; j++){
      agentNE->GetNNIndex(j)->SetWeights(loadedNN[k]->GetWeightsA(),loadedNN[k]->GetWeightsB()) ;
      k++ ;
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputActions(storeActs, storeBar) ;
  ResetEpochEvals() ;
  SimulateEpoch(false) ; // simulate in test mode
  
  for (size_t i = 0; i < loadedNN.size(); i++){
    delete loadedNN[i] ;
    loadedNN[i] = 0 ;
  }
}

vector< vector<size_t> > MultiNightBar::RandomiseTeams(size_t n){
  vector< vector<size_t> > teams ;
  vector<size_t> order ;
  for (size_t i = 0; i < n; i++)
    order.push_back(i) ;
  
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;

  for (size_t j = 0; j < nAgents; j++){
    shuffle (order.begin(), order.end(), std::default_random_engine(seed)) ;
    teams.push_back(order) ;
  }
  
  return teams ;
}
