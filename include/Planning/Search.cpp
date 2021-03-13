#include "Search.h"

Node * Search::PathSearch()
{
  size_t sourceID = FindSourceID() ;
  itsQueue = new Queue(new Node(itsGraph->GetVertices()[sourceID], SOURCE)) ;
  bool pathFound = false ;

  while (!itsQueue->EmptyQueue()){
    // Pop cheapest node from queue
    Node * currentNode = itsQueue->PopQueue() ;

    // Terminate search once path is found
    if (currentNode->GetVertex() == itsGoal){
      pathFound = true ;
	    break ;
    }

    // Find all neighbours excluding ancestor vertices if any
    vector<Edge *> neighbours = itsGraph->GetNeighbours(currentNode) ;

    // Update neighbours
    for (size_t i = 0; i < neighbours.size(); i++){
	    // Create neighbour node
	    Node * currentNeighbour = new Node(currentNode, neighbours[i]) ;
	    itsQueue->UpdateQueue(currentNeighbour) ;
    }
  }

  if (!pathFound){
    cout << "No path found from source to goal. Exiting.\n" ;
    exit(1) ;
  }
  else{
    Node * bestPath ;

    for (size_t i = 0; i < itsQueue->GetClosed().size(); i++){
	    if (itsGoal == itsQueue->GetClosed()[i]->GetVertex()){
		    bestPath = itsQueue->GetClosed()[i] ;
		    break ;
	    }
    }
    
    return bestPath ;
  }
}

void Search::ResetSearch(){
  if (itsQueue){
    delete itsQueue ;
    itsQueue = 0 ;
  }
}

size_t Search::FindSourceID()
{
  for (size_t i = 0; i < itsGraph->GetNumVertices(); i++){
    if (itsSource == itsGraph->GetVertices()[i]){
      return i ;
    }
  }
  cout << "Error: souce ID not found. Exiting.\n" ;
  exit(1) ;
}
