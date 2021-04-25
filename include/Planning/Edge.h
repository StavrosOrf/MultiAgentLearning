#ifndef EDGE_H_
#define EDGE_H_

#include <stddef.h>
#include <cstdint>

typedef int16_t vertex_t;

// Edge class to contain mean and variance of cost along an edge
class Edge{
	public:
		Edge(vertex_t v1, vertex_t v2, float cost):
			itsVertex1(v1), itsVertex2(v2), itsCost(cost), itsLength((size_t)cost) {}
		~Edge(){}

		vertex_t GetVertex1() const {return itsVertex1;}
		vertex_t GetVertex2() const {return itsVertex2;}
		float GetCost() const {return itsCost;}
		void SetCost(float cost) {itsCost = cost;}
		size_t GetLength(){return itsLength;}

		friend bool operator==(const Edge &lhs, const Edge &rhs){
			return lhs.GetVertex1() == rhs.GetVertex1() && lhs.GetVertex2() == rhs.GetVertex2();
		}
	private:
		vertex_t itsVertex1; //parent (or source) vertex
		vertex_t itsVertex2; //child (or destination) vertex
		float itsCost;
		size_t itsLength;
};

#endif // EDGE_H_
