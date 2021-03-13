#ifndef NODE_H_
#define NODE_H_

#include <iostream> // std::cout, std::endl
#include <float.h> // DBL_MAX
#include "Edge.h"

using std::cout ;
using std::endl ;

enum nodeType {SOURCE, OTHER} ;

// Node class to maintain path information up to a vertex
// contains parent node, (mu, sigma) of cost-to-come
class Node
{
	public:
		Node(int vertex):
		  itsVertex(vertex), itsParent(0), itsCost(0.0) {}
		
		Node(int vertex, nodeType n):
		itsVertex(vertex), itsParent(0)
		{
	    switch (n)
	    {
		    case SOURCE:
			    itsCost = 0.0 ;
			    break ;
		    default:
			    itsCost = DBL_MAX ;
	    }
    }
    
		Node(Node * parent, Edge * edge):
		itsParent(parent)
		{
	    itsVertex = edge->GetVertex2() ;
	    itsCost = itsParent->GetCost() + edge->GetCost() ;
    }
    
		~Node(){} ;
		
		Node * GetParent() const {return itsParent ;}
		void SetParent(Node * parent) {itsParent = parent ;}
		double GetCost() const {return itsCost ;}
		void SetCost(double cost) {itsCost = cost ;}
		int GetVertex() const {return itsVertex ;}
		void SetVertex(int vertex) {itsVertex = vertex ;}
		
		void DisplayPath() ;
		Node * ReverseList(Node * itsChild) ;
    
	private:
		int itsVertex ;
		Node * itsParent ;
		double itsCost ;
} ;

#endif // NODE_H_
