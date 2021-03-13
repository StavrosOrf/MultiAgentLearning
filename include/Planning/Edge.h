#ifndef EDGE_H_
#define EDGE_H_

#include <stddef.h>

// Edge class to contain mean and variance of cost along an edge
class Edge
{
	public:
		Edge(int v1, int v2, double cost):
		  itsVertex1(v1), itsVertex2(v2), itsCost(cost), itsLength((size_t)cost) {}
		~Edge(){}
		
		int GetVertex1() const {return itsVertex1 ;}
		int GetVertex2() const {return itsVertex2 ;}
		double GetCost() const {return itsCost ;}
		void SetCost(double cost) {itsCost = cost ;}
		size_t GetLength(){return itsLength ;}
		
		friend bool operator==(const Edge &lhs, const Edge &rhs){
		  return lhs.GetVertex1() == rhs.GetVertex1() && lhs.GetVertex2() == rhs.GetVertex2() ;
		}
	private:
		int itsVertex1 ;
		int itsVertex2 ;
		double itsCost ;
		size_t itsLength ;
} ;

#endif // EDGE_H_
