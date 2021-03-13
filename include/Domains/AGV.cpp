#include "AGV.h"

AGV::AGV(int initV, vector<int> goalVs, Graph * graph){
  nextVertex = -1 ;
  t2v = 0 ;
  origin = initV ;
  goal = -1 ;
  nsDel = 0 ;
  ncDel = 0 ;
  tMove = 0 ;
  tEnter = 0 ;
  tWait = 0 ;
  agvGoals = goalVs ;
  isReplan = true ;
  agvPlanner = new Search(graph, origin, goal) ;
  SetNewGoal() ;
  ncDel++ ;
}

AGV::~AGV(){
  delete agvPlanner ;
  agvPlanner = 0 ;
}

void AGV::ResetAGV(){
  nextVertex = -1 ;
  t2v = 0 ;
  nsDel = 0 ;
  ncDel = 0 ;
  tMove = 0 ;
  tEnter = 0 ;
  tWait = 0 ;
  agvPlanner->SetSource(origin) ;
  SetNewGoal() ;
  ncDel++ ;
  agvPlanner->ResetSearch() ;
  curEdge = 0 ;
  isReplan = true ;
}

void AGV::Traverse(){
  if (t2v > 0){ // move along edge
    t2v-- ;
    tMove++ ;
    // manage end of path transitions here
    if (t2v == 0){
      if (nextVertex == goal){ // end of path
        nsDel++ ; // increment number of sucessful deliveries
        agvPlanner->SetSource(goal) ; // set current vertex as new source
        SetNewGoal() ; // randomly choose and set new goal vertex from available set
        nextVertex = -1 ; // agv must re-enter graph
        ncDel++ ; // increment number of commanded deliveries
        agvPlanner->ResetSearch() ;
        curEdge = 0 ; // remove it from its final edge
        isReplan = true ;
      }
    }
  }
  else{
    if (nextVertex < 0){ // waiting to enter
      tEnter++ ;
    }
    else{ // waiting to cross intersection
      tWait++ ;
    }
  }
}

void AGV::EnterNewEdge(){
  // update current edge, next vertex, etc.
  curEdge = path.front() ;
  path.pop_front() ;
  nextVertex = curEdge->GetVertex2() ;
  t2v = curEdge->GetLength() ;
  isReplan = false ;
}

void AGV::CompareCosts(vector<double> c){
  if (t2v == 0 && !isReplan){ // compare if waiting to change edges and flag is not already set
    // Compare current graph costs to prior costs used to generate existing plan
    for (size_t i = 0; i < c.size(); i++){
      if (c[i] != costs[i]){
        agvPlanner->ResetSearch() ;
        isReplan = true ;
        if (nextVertex >= 0){ // only reset search source if AGV is enroute
          agvPlanner->SetSource(nextVertex) ;
        }
        break ;
      }
    }
  }
}

void AGV::PlanAGV(vector<double> c){
  size_t nEdges = agvPlanner->GetGraph()->GetNumEdges() ;
  vector<Edge *> edges = agvPlanner->GetGraph()->GetEdges() ;
  
  // Assuming agvPlanner is set up with correct start, goal
  Node * bestPath = agvPlanner->PathSearch() ;
  int v2 = bestPath->GetVertex() ;
  
  path.clear() ;
  while (bestPath->GetParent()){
    Node * curNode = bestPath->GetParent() ;
    int v1 = curNode->GetVertex() ;
    
    bool edgeFound = false ;
    for (size_t i = 0; i < nEdges; i++){
      if (edges[i]->GetVertex1() == v1 && edges[i]->GetVertex2() == v2){
        path.push_front(edges[i]) ;
        edgeFound = true ;
        break ;
      }
    }
    if (!edgeFound){
      std::cout << "Error: cannot find path through graph edges! Exiting.\n" ;
      exit(1) ;
    }
    bestPath = curNode ;
    v2 = v1 ;
  }
  
  costs = c ; // record the set of costs used for most recent plan
  isReplan = false ;
}

void AGV::DisplayPath(){
  for (list<Edge *>::iterator it = path.begin(); it != path.end(); it++){
    std::cout << "(" << (*it)->GetVertex1() << "," << (*it)->GetVertex2() << ") " ;
  }
  std::cout << "\n" ;
}

void AGV::SetNewGoal(){
  int newGoal = agvPlanner->GetSource() ;
  while (newGoal == agvPlanner->GetSource()){
    size_t gID = rand() % agvGoals.size() ;
    newGoal = agvGoals[gID] ;
  }
  goal = newGoal ;
  agvPlanner->SetGoal(goal) ; // set new goal vertex
}

