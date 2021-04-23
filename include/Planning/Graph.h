#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream> //std::cout
#include <vector> // std::vector
#include <cassert>
#include <stddef.h>
#include "Edge.h"
#include "Node.h"


// Graph class to create and store graph structure
class Graph{
	public:
		Graph(std::vector<vertex_t> &vertices, std::vector< std::vector<vertex_t> > &edges, std::vector<float> &costs): itsVertices(vertices){
			numVertices = itsVertices.size();
			GenerateEdges(edges, costs);
		}

		~Graph(){
		}

		std::vector<vertex_t> GetVertices() const {return itsVertices;}
		std::vector<const Edge *> GetEdges() const;
		size_t GetNumVertices() const {return numVertices;}
		size_t GetNumEdges() const {return numEdges;}
		size_t GetEdgeID(const Edge *) __attribute__ ((pure));

		std::vector<Edge*> get_outgoing_edges_of_a_vertex(vertex_t vertex);
		std::vector<Edge*> get_incoming_edges_of_a_vertex(vertex_t vertex);

		std::vector<Edge *> GetNeighbours(Node * n);

		void reset_edge_costs();
		void set_edge_cost(std::vector<float> new_costs);

	private:
		std::vector<vertex_t> itsVertices;
		std::vector<Edge> itsEdges;
		size_t numVertices;
		size_t numEdges;

		void GenerateEdges(std::vector< std::vector<vertex_t> > &edges, std::vector<float> &costs);
};

#endif // GRAPH_H_
