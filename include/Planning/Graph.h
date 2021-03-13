#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream> //std::cout
#include <vector> // std::vector
#include "Edge.h"
#include "Node.h"

using std::vector ;

// Graph class to create and store graph structure
class Graph
{
	public:
	  Graph(vector<int> &vertices, vector< vector<int> > &edges, vector<double> &costs): itsVertices(vertices)
	  {
	    numVertices = itsVertices.size() ;
	    GenerateEdges(edges, costs) ;
    }
    
		~Graph()
		{
	    for (unsigned i = 0; i < numEdges; i++){
		  	delete itsEdges[i] ;
		  	itsEdges[i] = 0 ;
    	}
    }
		
		vector<int> GetVertices() const {return itsVertices ;}
		vector<Edge *> GetEdges() const {return itsEdges ;}
		size_t GetNumVertices() const {return numVertices ;}
		size_t GetNumEdges() const {return numEdges ;}
		size_t GetEdgeID(Edge *) ;
		
		vector<Edge *> GetNeighbours(Node * n) ;
		
	private:
		vector<int> itsVertices ;
		vector<Edge *> itsEdges ;
		size_t numVertices ;
		size_t numEdges ;
		
		void GenerateEdges(vector< vector<int> > &edges, vector<double> &costs) ;
} ;

#endif // GRAPH_H_
