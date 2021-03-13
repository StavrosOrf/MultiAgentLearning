#include "Warehouse.h"

Warehouse::Warehouse(YAML::Node configs){
  // Read in graph definition from vertices and edges files (directories stored in configs)
  string domainDir = configs["domain"]["folder"].as<string>() ;
  string vFile = domainDir + configs["graph"]["vertices"].as<string>() ;
  string eFile = domainDir + configs["graph"]["edges"].as<string>() ;
  string cFile = domainDir + configs["graph"]["capacities"].as<string>() ;
  nPop = configs["neuroevo"]["population_size"].as<size_t>() ;
  nSteps = configs["simulation"]["steps"].as<size_t>() ;
  neLearn = configs["neuroevo"]["learn"].as<bool>() ;
  
  InitialiseGraph(vFile, eFile, cFile, configs) ;
  outputEvals = false ;
}

Warehouse::~Warehouse(void){
  delete whGraph ;
  whGraph = 0 ;
  for (size_t i = 0; i < whAGVs.size(); i++){
    delete whAGVs[i] ;
    whAGVs[i] = 0 ;
  }
  for (size_t i = 0; i < whAgents.size(); i++){
    delete whAgents[i] ;
    whAgents[i] = 0 ;
    delete maTeam[i] ;
    maTeam[i] = 0 ;
  }
  if (outputEvals){
    evalFile.close() ;
  }
  if (outputEpReplay){
    agvStateFile.close() ;
    agvEdgeFile.close() ;
    agentStateFile.close() ;
    agentActionFile.close() ;
  }
}

void Warehouse::EvolvePolicies(bool init){
  for (size_t i = 0; i < nAgents; i++)
    maTeam[i]->EvolvePolicies(init) ;
}

void Warehouse::ResetEpochEvals(){
  for (size_t i = 0; i < nAgents; i++)
    maTeam[i]->ResetEpochEvals() ;
}

void Warehouse::OutputPerformance(string eval_str){
  if (evalFile.is_open())
    evalFile.close() ;
  evalFile.open(eval_str.c_str(),std::ios::app) ;
  
  outputEvals = true ;
  
  std::cout << "Writing evaluation outputs to file: " << eval_str << "\n" ;
}

void Warehouse::OutputControlPolicies(string nn_str){
  for (size_t i = 0; i < nAgents; i++){
    maTeam[i]->OutputNNs(nn_str) ;
  }
}

void Warehouse::OutputEpisodeReplay(string agv_s_str, string agv_e_str, string a_s_str, string a_a_str){
  if (agvStateFile.is_open())
    agvStateFile.close() ;
  agvStateFile.open(agv_s_str.c_str(),std::ios::app) ;
  
  if (agvEdgeFile.is_open())
    agvEdgeFile.close() ;
  agvEdgeFile.open(agv_e_str.c_str(),std::ios::app) ;
  
  if (agentStateFile.is_open())
    agentStateFile.close() ;
  agentStateFile.open(a_s_str.c_str(),std::ios::app) ;
  
  if (agentActionFile.is_open())
    agentActionFile.close() ;
  agentActionFile.open(a_a_str.c_str(),std::ios::app) ;
  
  outputEpReplay = true ;
  
  std::cout << "Writing AGV logs to files: " << "\n" ;
  std::cout << "\t" << agv_s_str << "\n" ;
  std::cout << "\t" << agv_e_str << "\n" ;
  
  std::cout << "Writing agent logs to files: " << "\n" ;
  std::cout << "\t" << a_s_str << "\n" ;
  std::cout << "\t" << a_a_str << "\n" ;
}

void Warehouse::LoadPolicies(YAML::Node configs){
  // Filename to read NN control policy
  string nn_str = configs["mode"]["agent_policies"].as<string>() ;
  std::ifstream nnFile ;
  
  vector<NeuralNet *> loadedNN ;
  size_t fPop = 2*nPop ;
  std::cout << "Reading out " << fPop << " NN control policies for each agent to test...\n" ;
  
  // Read in NN weight matrices for each agent
  int kStart = 0 ;
  int kEnd ;
  for (size_t n = 0; n < nAgents; n++){
    // Double population through mutation
    maTeam[n]->GetNEPopulation()->MutatePopulation() ;
  
    nnFile.open(nn_str.c_str(),std::ios::in) ;
    // Get NN matrix parameters for current agent
    size_t nIn = maTeam[n]->GetNumIn() ;
    size_t nOut = maTeam[n]->GetNumOut() ;
    size_t nHid = maTeam[n]->GetNumHidden() ;
    std::string line ;
    MatrixXd NNA ;
    MatrixXd NNB ;
    NNA.setZero(nIn,nHid) ;
    NNB.setZero(nHid+1,nOut) ;
    int nnK = NNA.rows() + NNB.rows() ; // number of lines corresponding to a single NN
    int popK = nnK * fPop ; // number of lines corresponding to the entire population
    kEnd = kStart + popK ; // rows corresponding to agent i's NN population
    size_t m = 0 ; // track member number
    int k = 0 ; // track line number
    while (std::getline(nnFile,line)){
      std::stringstream lineStream(line) ;
      std::string cell ;
      if (k >= kStart){
        if ((k-kStart) % nnK < NNA.rows()){
          int i = (k-kStart) % nnK ;
          int j = 0 ;
          while (std::getline(lineStream,cell,',')){
            NNA(i,j++) = atof(cell.c_str()) ;
          }
        }
        else {
          int i = ((k-kStart) % nnK)-NNA.rows() ;
          int j = 0 ;
          while (std::getline(lineStream,cell,',')){
            NNB(i,j++) = atof(cell.c_str()) ;
          }
        }
        if ((k-kStart+1) % nnK == 0){
          maTeam[n]->GetNEPopulation()->GetNNIndex(m)->SetWeights(NNA,NNB) ;
          m++ ;
        }
      }
      k++ ;
      if (k >= kEnd){
        if (m != fPop){
          std::cout << "Error: insufficient NN's to fill all population members. Exiting.\n" ;
          exit(1) ;
        }
        kStart = kEnd ;
        break ;
      }
    }
    nnFile.close() ;
  }
}

void Warehouse::InitialiseGraph(string v_str, string e_str, string c_str, YAML::Node configs){
  vector<int> vertices ;
  vector< vector<int> > edges ;
  vector< double > costs ;
  
  // Read in data from files
  cout << "Reading vertices from file: " ;
  ifstream verticesFile(v_str.c_str()) ;
  cout << v_str.c_str() << "..." ;
  if (!verticesFile.is_open()){
    cout << "\nFile: " << v_str.c_str() << " not found, exiting.\n" ;
    exit(1) ;
  }
  string line ;
  while (getline(verticesFile,line))
  {
    vertices.push_back(atoi(line.c_str())) ;
  }
  cout << "complete. " << vertices.size() << " vertices in graph.\n" ;
  
  cout << "Reading edges from file: " ;
  ifstream edgesFile(e_str.c_str()) ;
  cout << e_str.c_str() << "..." ;
  if (!edgesFile.is_open()){
    cout << "\nFile: " << e_str.c_str() << " not found, exiting.\n" ;
    exit(1) ;
  }
  while (getline(edgesFile,line))
  {
    stringstream lineStream(line) ;
    string cell ;
    vector<double> ec ;
    while (getline(lineStream,cell,','))
    {
	    ec.push_back(atof(cell.c_str())) ;
    }
    vector<int> e ;
    e.push_back((int)ec[0]) ;
    e.push_back((int)ec[1]) ;
    costs.push_back(ec[2]) ;
    edges.push_back(e) ;
  }
  cout << "complete. " << edges.size() << " edges in graph.\n" ;
  
  baseCosts = costs ;
  
  // Read in data from files
  cout << "Reading capacities from file: " ;
  ifstream capacitiesFile(c_str.c_str()) ;
  cout << c_str.c_str() << "..." ;
  if (!capacitiesFile.is_open()){
    cout << "\nFile: " << c_str.c_str() << " not found, exiting.\n" ;
    exit(1) ;
  }
  while (getline(capacitiesFile,line))
  {
    capacities.push_back((size_t)atoi(line.c_str())) ;
  }
  cout << "complete.\n" ;
  
  whGraph = new Graph(vertices, edges, costs) ;
  
//  std::cout << "Number of graph vertices: " << whGraph->GetNumVertices() << "\n" ;
//  std::cout << "Number of graph edges: " << whGraph->GetNumEdges() << "\n" ;
  
//  InitialiseMATeam() ;
  InitialiseAGVs(configs) ;
}

void Warehouse::InitialiseAGVs(YAML::Node configs){
  string domainDir = configs["domain"]["folder"].as<string>() ;
  // Initialise AGV objects
  string agv_str = domainDir + configs["simulation"]["agvs"].as<string>() ;
  
  // Read in data from files
  cout << "Reading AGVs from file: " ;
  ifstream AGVFile(agv_str.c_str()) ;
  cout << agv_str.c_str() << "..." ;
  if (!AGVFile.is_open()){
    cout << "\nFile: " << agv_str.c_str() << " not found, exiting.\n" ;
    exit(1) ;
  }
  nAGVs = 0 ;
  vector<size_t> agvOrigins ;
  string line ;
  while (getline(AGVFile,line))
  {
    agvOrigins.push_back((size_t)atoi(line.c_str())) ;
    nAGVs++ ;
  }
  cout << "complete. Created " << nAGVs << " AGVs.\n" ;
  
//  std::cout << "Number of AGVs: " << nAGVs << "\n" ;
  
  // Store valid destination vertex IDs
  string goal_str = domainDir + configs["simulation"]["goals"].as<string>() ;
  
  // Read in data from files
  cout << "Reading goal vertex IDs from file: " ;
  ifstream goalFile(goal_str.c_str()) ;
  cout << goal_str.c_str() << "..." ;
  if (!goalFile.is_open()){
    cout << "\nFile: " << goal_str.c_str() << " not found, exiting.\n" ;
    exit(1) ;
  }
  vector<int> agvGoals ;
  while (getline(goalFile,line))
  {
    agvGoals.push_back(atoi(line.c_str())) ;
  }
  cout << "complete. " << agvGoals.size() << " goal vertices.\n" ;
  
//  std::cout << "Number of possible goals: " << agvGoals.size() << "\n" ;
  
  for (size_t i = 0; i < nAGVs; i++){
    AGV * agv = new AGV(agvOrigins[i], agvGoals, whGraph) ;
    whAGVs.push_back(agv) ;
  }
}

void Warehouse::InitialiseNewEpoch(){
  // Reset all AGVs
  for (size_t i = 0; i < nAGVs; i++){
    whAGVs[i]->ResetAGV() ;
  }
  for (size_t i = 0; i < nAgents; i++){
    whAgents[i]->agvIDs.clear() ;
  }
}

vector< vector<size_t> > Warehouse::RandomiseTeams(size_t n){ // n = number of agents in each team
  vector< vector<size_t> > teams ;
  vector<size_t> order ;
  for (size_t i = 0; i < n; i++){
    order.push_back(i) ;
  }
  
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;

  for (size_t j = 0; j < nAgents; j++){
    shuffle (order.begin(), order.end(), std::default_random_engine(seed)) ;
    teams.push_back(order) ;
  }
  
  return teams ;
}

void Warehouse::UpdateGraphCosts(vector<double> costs){
  vector<Edge *> e = whGraph->GetEdges() ;
  for (size_t i = 0; i < e.size(); i++){
    e[i]->SetCost(costs[i]) ;
  }
}

//void Warehouse::SimulateEpoch(bool train){
//  size_t teamSize ;
//  if (train)
//    teamSize = 2*nPop ;
//  else
//    teamSize = nPop ;
//  
//  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; // each row is the population for a single agent
//  
//  double maxEval = 0.0 ;
//  size_t maxTeamID = 0 ;
//  vector<size_t> travelStats ;
//  vector<size_t> championIDs ;
//  
//  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
//    InitialiseNewEpoch() ;
//  
//    if (outputEpReplay){
//      for (size_t k = 0; k < nAGVs; k++){
//        if (whAGVs[k]->GetNextVertex() < 0){
//          agvStateFile << "1," ;
//          agvEdgeFile << "-1," ;
//        }
//        else{
//          agvStateFile << "0," ;
//          agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
//        }
//      }
//      agvStateFile << "\n" ;
//      agvEdgeFile << "\n" ;
//      
//      for(size_t n = 0; n < whGraph->GetNumEdges(); n++){
//        agentStateFile << "0," ;
//        agentActionFile << baseCosts[n] << "," ;
//      }
//      agentStateFile << "\n" ;
//      agentActionFile << "\n" ;
//    }
//    
//    vector<size_t> memberIDs ;
//    for (size_t j = 0; j < nAgents; j++){ // extract agent member IDs for this team
//      memberIDs.push_back(teams[j][i]) ;
//    }
//    
//    for (size_t t = 0; t < nSteps; t++){ // each timestep
//      // Get agent actions and update graph costs
//      vector<double> a = baseCosts ;
//      vector<size_t> s(whGraph->GetNumEdges(),0) ;
//      QueryMATeam(memberIDs, a, s) ;
//      UpdateGraphCosts(a) ;
//      
//      // Replan AGVs as necessary
//      for (size_t k = 0; k < nAGVs; k++){
//        whAGVs[k]->CompareCosts(a) ; // set replanning flags
//      
//        if (whAGVs[k]->GetIsReplan()){ // replanning needed
//          whAGVs[k]->PlanAGV(a) ;
//        }
//        
//        // Identify any new AGVs that need to cross an intersection
//        if (whAGVs[k]->GetT2V() == 0){
//          size_t agentID ;
//          if (whAGVs[k]->GetNextVertex() < 0){ // AGV waiting to enter graph
//            agentID = GetAgentID(whAGVs[k]->GetNextEdge()->GetVertex1()) ;
//          }
//          else{ // AGV enroute
//            agentID = GetAgentID(whAGVs[k]->GetNextVertex()) ;
//          }
//          bool onWaitList = false ;
//          for (list<size_t>::iterator it = whAgents[agentID]->agvIDs.begin(); it!=whAgents[agentID]->agvIDs.end(); ++it){
//            if (k == *it){
//              onWaitList = true ;
//              break ;
//            }
//          }
//          if (!onWaitList){ // only add if not already on wait list
//            whAgents[agentID]->agvIDs.push_back(k) ;
//          }
//        }
//      }
//      
//      // Attempt to move any transitioning AGVs on to new edges (according to wait list order)
//      for (size_t k = 0; k < nAgents; k++){
//        vector<size_t> toRemove ;
//        for (list<size_t>::iterator it = whAgents[k]->agvIDs.begin(); it!=whAgents[k]->agvIDs.end(); ++it){
//          size_t curAGV = *it ;
//          size_t nextID = whGraph->GetEdgeID(whAGVs[curAGV]->GetNextEdge()) ; // next edge ID
//          
//          bool edgeFull = false ;
//          if (nextID < 0 || nextID >= s.size()){
//            std::cout << "AGV #" << curAGV << ", nextID: " << nextID << "\n" ;
//            std::cout << "  t2v: " << whAGVs[curAGV]->GetT2V() << "\n" ;
//            std::cout << "  itsQueue: " << (whAGVs[curAGV]->GetAGVPlanner()->GetQueue() != 0) << "\n" ;
//          }
//          if (s[nextID] >= capacities[nextID]){ // check if next edge is full
//            edgeFull = true ;
//          }
//          if (!edgeFull){ // transfer to new edge and update agv counters
//            size_t curID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
//            whAGVs[curAGV]->EnterNewEdge() ;
//            size_t newID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
//            if (curID < s.size()){ // if moving off an edge
//              s[curID]-- ; // remove from old edge
//            }
//            s[newID]++ ; // add to new edge (note that nextID and newID should be equal!)
//            if (nextID != newID){
//              std::cout << "Warning: nextID [" << nextID << "] != newID [" << newID << "]\n" ;
//            }
//            toRemove.push_back(curAGV) ; // remember to remove from agent wait list
//          }
//        }
//        for (size_t w = 0; w < toRemove.size(); w++){
//          whAgents[k]->agvIDs.remove(toRemove[w]) ;
//        }
//      }
//      
//      // Traverse
//      for (size_t k = 0; k < nAGVs; k++){
//        whAGVs[k]->Traverse() ;
//      }
//  
//      if (outputEpReplay){
//        for (size_t k = 0; k < nAGVs; k++){
//          if (whAGVs[k]->GetNextVertex() < 0){
//            agvStateFile << "1," ;
//            agvEdgeFile << "-1," ;
//          }
//          else{
//            agvStateFile << "0," ;
//            agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
//          }
//        }
//        agvStateFile << "\n" ;
//        agvEdgeFile << "\n" ;
//        
//        for(size_t k = 0; k < s.size(); k++){
//          agentStateFile << s[k] << "," ;
//          agentActionFile << a[k] << "," ;
//        }
//        agentStateFile << "\n" ;
//        agentActionFile << "\n" ;
//      }
//      
//    } // end simulation timesteps
//    
//    // Log data
//    size_t totalMove = 0 ;
//    size_t totalEnter = 0 ;
//    size_t totalWait = 0 ;
//    size_t totalSuccess = 0 ;
//    size_t totalCommand = 0 ;
//    for (size_t k = 0; k < nAGVs; k++){
//      totalMove += whAGVs[k]->GetMoveTime() ;
//      totalEnter += whAGVs[k]->GetEnterTime() ;
//      totalWait += whAGVs[k]->GetWaitTime() ;
//      totalSuccess += whAGVs[k]->GetNumCompleted() ;
//      totalCommand += whAGVs[k]->GetNumCommanded() ;
//    }
//    double G = (double)(totalSuccess) ; // if number of AGVs is constant then AGV time is constant over runs and only number of successful deliveries counts
//    
//    for (size_t j = 0; j < nAgents; j++){ // assign reward to each agent
//      maTeam[j]->SetEpochPerformance(G, memberIDs[j]) ;
//    }
//    
//    if (G > maxEval){
//      maxEval = G ;
//      maxTeamID = i ;
//      travelStats.clear() ;
//      travelStats.push_back(totalMove) ;
//      travelStats.push_back(totalEnter) ;
//      travelStats.push_back(totalWait) ;
//      travelStats.push_back(totalSuccess) ;
//      travelStats.push_back(totalCommand) ;
//    }
//  } // end evaluation of one team
//  
//  // Champion team members
//  for (size_t j = 0; j < nAgents; j++){ // extract agent member IDs for this team
//    championIDs.push_back(teams[j][maxTeamID]) ;
//  }
//  
//  // Print out best team for this learning epoch
//  std::cout << "Best team: #" << maxTeamID << ", G: " << maxEval << "\n" ;
//  
//  if (outputEvals){
//    evalFile << maxTeamID << "," ;
//    evalFile << maxEval << "," ;
//    for (size_t i = 0; i < travelStats.size(); i++){
//      evalFile << travelStats[i] << "," ;
//    }
//    for (size_t i = 0; i < championIDs.size(); i++){
//      evalFile << championIDs[i] << "," ;
//    }
//    evalFile << "\n" ;
//  }
//}

//void Warehouse::InitialiseMATeam(){
//  // Initialise NE components and domain housekeeping components of the intersection agents
//  vector<int> v = whGraph->GetVertices() ;
//  vector<Edge *> e = whGraph->GetEdges() ;
//  for (size_t i = 0; i < v.size(); i++){
//    vector<size_t> eIDs ;
//    for (size_t j = 0; j < e.size(); j++){
//      if (e[j]->GetVertex2() == (int)i){
//        eIDs.push_back(j) ;
//      }
//    }
//    iAgent * agent = new iAgent{i, eIDs} ;
//    size_t nOut = eIDs.size() ; // NN output is additional cost applied to each edge
//    size_t nIn = nOut*2 ; // NN input is current #AGVs on edge and time to arrival at intersection
//    size_t nHid = nOut*4 ; // set number of hidden nodes to twice the number of input neurons
//    Agent * neAgent ;
//    if (isAgent == INTERSECTION)
//      neAgent = new Intersection(nPop, nIn, nOut, nHid) ;
//    else if (isAgent == LINK)
//      neAgent = new Link(nPop, nIn, nOut, nHid) ;
//    else
//      neAgent = new Agent(nPop, nIn, nOut, nHid) ;
//    
//    whAgents.push_back(agent) ;
//    maTeam.push_back(neAgent) ;
//  }
//  nAgents = whAgents.size() ;
//}

//void Warehouse::QueryMATeam(vector<size_t> memberIDs, vector<double> &a, vector<size_t> &s){
//  vector<Edge *> e = whGraph->GetEdges() ;
//  vector<double> eTime = baseCosts ;
//  GetJointState(e, s, eTime) ;
//  
//  for (size_t i = 0; i < nAgents; i++){
//    VectorXd input(whAgents[i]->eIDs.size()*2) ;
//    for (size_t j = 0; j < whAgents[i]->eIDs.size(); j++){
//      input(2*j) = s[whAgents[i]->eIDs[j]] ;
//      input(2*j+1) = eTime[whAgents[i]->eIDs[j]] ;
//    }
//    VectorXd output = maTeam[i]->ExecuteNNControlPolicy(memberIDs[i], input) ;
//    
//    double maxBaseCost ;
//    if (neLearn){
//      maxBaseCost = * std::max_element(baseCosts.begin(), baseCosts.end()) ;
//    }
//    else{
//      maxBaseCost = 0.0 ;
//    }
//    for (size_t j = 0; j < (size_t)output.size(); j++){ // output [0,1] scaled to max base cost
//      a[whAgents[i]->eIDs[j]] += output(j)*maxBaseCost ;
//    }
//  }
//}

//void Warehouse::GetJointState(vector<Edge *> e, vector<size_t> &s, vector<double> &eTime){
//  for (size_t i = 0; i < nAGVs; i++){
//    Edge * curEdge = whAGVs[i]->GetCurEdge() ;
//    size_t j = whGraph->GetEdgeID(curEdge) ;
//    if (j < s.size()){
//      s[j]++ ;
//      if (eTime[j] > whAGVs[i]->GetT2V()){ // update next time of arrival
//        eTime[j] = whAGVs[i]->GetT2V() ;
//      }
//    }
//  }
//}

//size_t Warehouse::GetAgentID(int v){
//  size_t vID ;
//  for (size_t i = 0; i < whGraph->GetNumVertices(); i++){
//    if (whGraph->GetVertices()[i] == v){
//      vID = i ;
//      break ;
//    }
//  }
//  for (size_t i = 0; i < whAgents.size(); i++){
//    if (vID == whAgents[i]->vID){
//      return i ;
//    }
//  }
//  std::cout << "Error: managing agent not found in graph!\n" ;
//  return whAgents.size() ;
//}
