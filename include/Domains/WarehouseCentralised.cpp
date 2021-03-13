#include "WarehouseCentralised.h"

WarehouseCentralised::~WarehouseCentralised(void){
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

void WarehouseCentralised::SimulateEpoch(bool train){
  size_t teamSize ;
  if (train)
    teamSize = 2*nPop ;
  else
    teamSize = nPop ;
  
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; // each row is the population for a single agent
  
  double maxEval = 0.0 ;
  size_t maxTeamID = 0 ;
  vector<size_t> travelStats ;
  vector<size_t> championIDs ;
  
  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
    InitialiseNewEpoch() ;
  
    if (outputEpReplay){
      for (size_t k = 0; k < nAGVs; k++){
        if (whAGVs[k]->GetNextVertex() < 0){
          agvStateFile << "1," ;
          agvEdgeFile << "-1," ;
        }
        else{
          agvStateFile << "0," ;
          agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
        }
      }
      agvStateFile << "\n" ;
      agvEdgeFile << "\n" ;
      
      for(size_t n = 0; n < whGraph->GetNumEdges(); n++){
        agentStateFile << "0," ;
        agentActionFile << baseCosts[n] << "," ;
      }
      agentStateFile << "\n" ;
      agentActionFile << "\n" ;
    }
    
    vector<size_t> memberIDs ;
    for (size_t j = 0; j < nAgents; j++){ // extract agent member IDs for this team
      memberIDs.push_back(teams[j][i]) ;
    }
    
    for (size_t t = 0; t < nSteps; t++){ // each timestep
      // Get agent actions and update graph costs
      vector<double> a = baseCosts ;
      vector<size_t> s(whGraph->GetNumEdges(),0) ;
      QueryMATeam(memberIDs, a, s) ;
      UpdateGraphCosts(a) ;
      
      // Replan AGVs as necessary
      for (size_t k = 0; k < nAGVs; k++){
        whAGVs[k]->CompareCosts(a) ; // set replanning flags
      
        if (whAGVs[k]->GetIsReplan()){ // replanning needed
          whAGVs[k]->PlanAGV(a) ;
        }
        
        // Identify any new AGVs that need to cross an intersection
        if (whAGVs[k]->GetT2V() == 0){
          size_t agentID = 0 ; // only one agent
          bool onWaitList = false ;
          for (list<size_t>::iterator it = whAgents[agentID]->agvIDs.begin(); it!=whAgents[agentID]->agvIDs.end(); ++it){
            if (k == *it){
              onWaitList = true ;
              break ;
            }
          }
          if (!onWaitList){ // only add if not already on wait list
            whAgents[agentID]->agvIDs.push_back(k) ;
          }
        }
      }
      
      // Attempt to move any transitioning AGVs on to new edges (according to wait list order)
      for (size_t k = 0; k < nAgents; k++){
        vector<size_t> toRemove ;
        for (list<size_t>::iterator it = whAgents[k]->agvIDs.begin(); it!=whAgents[k]->agvIDs.end(); ++it){
          size_t curAGV = *it ;
          size_t nextID = whGraph->GetEdgeID(whAGVs[curAGV]->GetNextEdge()) ; // next edge ID
          
          bool edgeFull = false ;
          if (nextID < 0 || nextID >= s.size()){
            std::cout << "AGV #" << curAGV << ", nextID: " << nextID << "\n" ;
            std::cout << "  t2v: " << whAGVs[curAGV]->GetT2V() << "\n" ;
            std::cout << "  itsQueue: " << (whAGVs[curAGV]->GetAGVPlanner()->GetQueue() != 0) << "\n" ;
          }
          if (s[nextID] >= capacities[nextID]){ // check if next edge is full
            edgeFull = true ;
          }
          if (!edgeFull){ // transfer to new edge and update agv counters
            size_t curID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
            whAGVs[curAGV]->EnterNewEdge() ;
            size_t newID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
            if (curID < s.size()){ // if moving off an edge
              s[curID]-- ; // remove from old edge
            }
            s[newID]++ ; // add to new edge (note that nextID and newID should be equal!)
            if (nextID != newID){
              std::cout << "Warning: nextID [" << nextID << "] != newID [" << newID << "]\n" ;
            }
            toRemove.push_back(curAGV) ; // remember to remove from agent wait list
          }
        }
        for (size_t w = 0; w < toRemove.size(); w++){
          whAgents[k]->agvIDs.remove(toRemove[w]) ;
        }
      }
      
      // Traverse
      for (size_t k = 0; k < nAGVs; k++){
        whAGVs[k]->Traverse() ;
      }
  
      if (outputEpReplay){
        for (size_t k = 0; k < nAGVs; k++){
          if (whAGVs[k]->GetNextVertex() < 0){
            agvStateFile << "1," ;
            agvEdgeFile << "-1," ;
          }
          else{
            agvStateFile << "0," ;
            agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
          }
        }
        agvStateFile << "\n" ;
        agvEdgeFile << "\n" ;
        
        for(size_t k = 0; k < s.size(); k++){
          agentStateFile << s[k] << "," ;
          agentActionFile << a[k] << "," ;
        }
        agentStateFile << "\n" ;
        agentActionFile << "\n" ;
      }
      
    } // end simulation timesteps
    
    // Log data
    size_t totalMove = 0 ;
    size_t totalEnter = 0 ;
    size_t totalWait = 0 ;
    size_t totalSuccess = 0 ;
    size_t totalCommand = 0 ;
    for (size_t k = 0; k < nAGVs; k++){
      totalMove += whAGVs[k]->GetMoveTime() ;
      totalEnter += whAGVs[k]->GetEnterTime() ;
      totalWait += whAGVs[k]->GetWaitTime() ;
      totalSuccess += whAGVs[k]->GetNumCompleted() ;
      totalCommand += whAGVs[k]->GetNumCommanded() ;
    }
    double G = (double)(totalSuccess) ; // if number of AGVs is constant then AGV time is constant over runs and only number of successful deliveries counts
    
    for (size_t j = 0; j < nAgents; j++){ // assign reward to each agent
      maTeam[j]->SetEpochPerformance(G, memberIDs[j]) ;
    }
    
    if (G > maxEval){
      maxEval = G ;
      maxTeamID = i ;
      travelStats.clear() ;
      travelStats.push_back(totalMove) ;
      travelStats.push_back(totalEnter) ;
      travelStats.push_back(totalWait) ;
      travelStats.push_back(totalSuccess) ;
      travelStats.push_back(totalCommand) ;
    }
  } // end evaluation of one team
  
  // Champion team members
  for (size_t j = 0; j < nAgents; j++){ // extract agent member IDs for this team
    championIDs.push_back(teams[j][maxTeamID]) ;
  }
  
  // Print out best team for this learning epoch
  std::cout << "Best policy: #" << maxTeamID << ", G: " << maxEval << "\n" ;
  
  if (outputEvals){
    evalFile << maxTeamID << "," ;
    evalFile << maxEval << "," ;
    for (size_t i = 0; i < travelStats.size(); i++){
      evalFile << travelStats[i] << "," ;
    }
    for (size_t i = 0; i < championIDs.size(); i++){
      evalFile << championIDs[i] << "," ;
    }
    evalFile << "\n" ;
  }
}

void WarehouseCentralised::SimulateEpoch(vector<size_t> memberIDs){
  double maxEval = 0.0 ;
  vector<size_t> travelStats ;
  
  InitialiseNewEpoch() ;

  if (outputEpReplay){
    for (size_t k = 0; k < nAGVs; k++){
      if (whAGVs[k]->GetNextVertex() < 0){
        agvStateFile << "1," ;
        agvEdgeFile << "-1," ;
      }
      else{
        agvStateFile << "0," ;
        agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
      }
    }
    agvStateFile << "\n" ;
    agvEdgeFile << "\n" ;
    
    for(size_t n = 0; n < whGraph->GetNumEdges(); n++){
      agentStateFile << "0," ;
      agentActionFile << baseCosts[n] << "," ;
    }
    agentStateFile << "\n" ;
    agentActionFile << "\n" ;
  }
  
  for (size_t t = 0; t < nSteps; t++){ // each timestep
    // Get agent actions and update graph costs
    vector<double> a = baseCosts ;
    vector<size_t> s(whGraph->GetNumEdges(),0) ;
    QueryMATeam(memberIDs, a, s) ;
    UpdateGraphCosts(a) ;
    
    // Replan AGVs as necessary
    for (size_t k = 0; k < nAGVs; k++){
      whAGVs[k]->CompareCosts(a) ; // set replanning flags
    
      if (whAGVs[k]->GetIsReplan()){ // replanning needed
        whAGVs[k]->PlanAGV(a) ;
      }
      
      // Identify any new AGVs that need to cross an intersection
      if (whAGVs[k]->GetT2V() == 0){
        size_t agentID = 0 ; // only one agent
        bool onWaitList = false ;
        for (list<size_t>::iterator it = whAgents[agentID]->agvIDs.begin(); it!=whAgents[agentID]->agvIDs.end(); ++it){
          if (k == *it){
            onWaitList = true ;
            break ;
          }
        }
        if (!onWaitList){ // only add if not already on wait list
          whAgents[agentID]->agvIDs.push_back(k) ;
        }
      }
    }
    
    // Attempt to move any transitioning AGVs on to new edges (according to wait list order)
    for (size_t k = 0; k < nAgents; k++){
      vector<size_t> toRemove ;
      for (list<size_t>::iterator it = whAgents[k]->agvIDs.begin(); it!=whAgents[k]->agvIDs.end(); ++it){
        size_t curAGV = *it ;
        size_t nextID = whGraph->GetEdgeID(whAGVs[curAGV]->GetNextEdge()) ; // next edge ID
        
        bool edgeFull = false ;
        if (nextID < 0 || nextID >= s.size()){
          std::cout << "AGV #" << curAGV << ", nextID: " << nextID << "\n" ;
          std::cout << "  t2v: " << whAGVs[curAGV]->GetT2V() << "\n" ;
          std::cout << "  itsQueue: " << (whAGVs[curAGV]->GetAGVPlanner()->GetQueue() != 0) << "\n" ;
        }
        if (s[nextID] >= capacities[nextID]){ // check if next edge is full
          edgeFull = true ;
        }
        if (!edgeFull){ // transfer to new edge and update agv counters
          size_t curID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
          whAGVs[curAGV]->EnterNewEdge() ;
          size_t newID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge()) ;
          if (curID < s.size()){ // if moving off an edge
            s[curID]-- ; // remove from old edge
          }
          s[newID]++ ; // add to new edge (note that nextID and newID should be equal!)
          if (nextID != newID){
            std::cout << "Warning: nextID [" << nextID << "] != newID [" << newID << "]\n" ;
          }
          toRemove.push_back(curAGV) ; // remember to remove from agent wait list
        }
      }
      for (size_t w = 0; w < toRemove.size(); w++){
        whAgents[k]->agvIDs.remove(toRemove[w]) ;
      }
    }
    
    // Traverse
    for (size_t k = 0; k < nAGVs; k++){
      whAGVs[k]->Traverse() ;
    }

    if (outputEpReplay){
      for (size_t k = 0; k < nAGVs; k++){
        if (whAGVs[k]->GetNextVertex() < 0){
          agvStateFile << "1," ;
          agvEdgeFile << "-1," ;
        }
        else{
          agvStateFile << "0," ;
          agvEdgeFile << whGraph->GetEdgeID(whAGVs[k]->GetCurEdge()) << "," ;
        }
      }
      agvStateFile << "\n" ;
      agvEdgeFile << "\n" ;
      
      for(size_t k = 0; k < s.size(); k++){
        agentStateFile << s[k] << "," ;
        agentActionFile << a[k] << "," ;
      }
      agentStateFile << "\n" ;
      agentActionFile << "\n" ;
    }
    
  } // end simulation timesteps
  
  // Log data
  size_t totalMove = 0 ;
  size_t totalEnter = 0 ;
  size_t totalWait = 0 ;
  size_t totalSuccess = 0 ;
  size_t totalCommand = 0 ;
  for (size_t k = 0; k < nAGVs; k++){
    totalMove += whAGVs[k]->GetMoveTime() ;
    totalEnter += whAGVs[k]->GetEnterTime() ;
    totalWait += whAGVs[k]->GetWaitTime() ;
    totalSuccess += whAGVs[k]->GetNumCompleted() ;
    totalCommand += whAGVs[k]->GetNumCommanded() ;
  }
  double G = (double)(totalSuccess) ; // if number of AGVs is constant then AGV time is constant over runs and only number of successful deliveries counts
  
  for (size_t j = 0; j < nAgents; j++){ // assign reward to each agent
    maTeam[j]->SetEpochPerformance(G, memberIDs[j]) ;
  }
  
  maxEval = G ;
  travelStats.clear() ;
  travelStats.push_back(totalMove) ;
  travelStats.push_back(totalEnter) ;
  travelStats.push_back(totalWait) ;
  travelStats.push_back(totalSuccess) ;
  travelStats.push_back(totalCommand) ;
  
  // Print out team performance
  std::cout << "G: " << maxEval << "\n" ;
  
  if (outputEvals){
    evalFile << maxEval << "," ;
    for (size_t i = 0; i < travelStats.size(); i++){
      evalFile << travelStats[i] << "," ;
    }
    evalFile << "\n" ;
  }
}

void WarehouseCentralised::InitialiseMATeam(){
  // Initialise NE component and domain housekeeping components of the centralised agent
  vector<Edge *> e = whGraph->GetEdges() ;
  vector<size_t> eIDs ;
  for (size_t j = 0; j < e.size(); j++){
    eIDs.push_back(j) ;
  }
  iAgent * agent = new iAgent{0, eIDs} ; // only one centralised agent controlling all traffic
  whAgents.push_back(agent) ;
  
  size_t nOut = eIDs.size() ; // NN output is additional cost applied to each edge
  size_t nIn = nOut ; // NN input is current #AGVs on all edges
//  size_t nHid = 16 ; // fixed to compare against link agent formulation
  size_t nHid = 4*nIn ; // control for relative representational capacity
  Agent * neAgent ;
  neAgent = new Intersection(nPop, nIn, nOut, nHid) ;// only one centralised agent
  maTeam.push_back(neAgent) ;
  nAgents = whAgents.size() ;
}

void WarehouseCentralised::QueryMATeam(vector<size_t> memberIDs, vector<double> &a, vector<size_t> &s){
  vector<Edge *> e = whGraph->GetEdges() ;
  GetJointState(e, s) ;
  
  for (size_t i = 0; i < nAgents; i++){
    VectorXd input(whAgents[i]->eIDs.size()) ;
    for (size_t j = 0; j < whAgents[i]->eIDs.size(); j++){
      input(j) = s[whAgents[i]->eIDs[j]] ;
    }
    VectorXd output = maTeam[i]->ExecuteNNControlPolicy(memberIDs[i], input) ;
    
    double maxBaseCost ;
    if (neLearn){
      maxBaseCost = * std::max_element(baseCosts.begin(), baseCosts.end()) ;
    }
    else{
      maxBaseCost = 0.0 ;
    }
    for (size_t j = 0; j < (size_t)output.size(); j++){ // output [0,1] scaled to max base cost
      a[whAgents[i]->eIDs[j]] += output(j)*maxBaseCost ;
    }
  }
}

void WarehouseCentralised::GetJointState(vector<Edge *> e, vector<size_t> &s){
  for (size_t i = 0; i < nAGVs; i++){
    Edge * curEdge = whAGVs[i]->GetCurEdge() ;
    size_t j = whGraph->GetEdgeID(curEdge) ;
    if (j < s.size()){
      s[j]++ ;
    }
  }
}
