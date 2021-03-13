#include "Graph.h"

size_t Graph::GetEdgeID(Edge * e)
{
  for (size_t i = 0; i < numEdges; i++){
    if (e == itsEdges[i]){
      return i ;
    }
  }
//  std::cout << "Error: edge not found in graph!\n" ;
  return numEdges ;
}

vector<Edge *> Graph::GetNeighbours(Node * n) // Do not include parent vertex in list of neighbours
{
  vector<Edge *> neighbours ;
  int v = n->GetVertex() ;

  for (size_t i = 0; i < numEdges; i++){
	  int v1 = itsEdges[i]->GetVertex1() ;
	  int v2 = itsEdges[i]->GetVertex2() ;
  	if (v1 == v){
		  bool isNeighbour = true ;
		  Node * n0 = n ;
		  while (n0->GetParent()){
				n0 = n0->GetParent() ;
		  	int v0 = n0->GetVertex() ;
				if (v2 == v0){
					isNeighbour = false ;
					break ;
				}
			}
			if (isNeighbour)
				neighbours.push_back(itsEdges[i]) ;
	  }
  }

  return neighbours ;
}

void Graph::GenerateEdges(vector< vector<int> > &edges, vector<double> &costs)
{
  numEdges = edges.size() ;

  for (size_t i = 0; i < numEdges; i++)
  {
    // Initial error checking
    int v1 = edges[i][0] ;
    int v2 = edges[i][1] ;
    bool v1Found = false ;
    bool v2Found = false ;
    for (size_t j = 0; j < numVertices; j++){
      if (v1 == itsVertices[j]){
        v1Found = true ;
      }
      if (v2 == itsVertices[j]){
        v2Found = true ;
      }
      if (v1Found && v2Found){
        break ;
      }
    }
    if (!v1Found || !v2Found){
      std::cout << "Error: edge (" << edges[i][0] << "," << edges[i][1] << ") not found. Mismatch in edges and vertices! Exiting.\n" ;
      exit(1) ;
    }

    Edge * e = new Edge(v1, v2, costs[i]) ;
    itsEdges.push_back(e) ;
  }
}
