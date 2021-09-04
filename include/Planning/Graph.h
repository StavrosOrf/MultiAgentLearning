#pragma once

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
			GenerateEdges(edges, costs);

			itsEdges_ref.reserve(itsEdges.size());
			for (size_t i = 0; i != itsEdges.size(); i++){
				const Edge* e = & itsEdges[i];
				itsEdges_ref.push_back(e);
			}
		}

		~Graph() = default;

		std::vector<vertex_t> GetVertices() const {return itsVertices;}
		const std::vector<const Edge *> GetEdges() const __attribute__ ((const)){return itsEdges_ref;} 
		size_t GetNumVertices() const {return itsVertices.size();}
		size_t GetNumEdges() const {return itsEdges.size();}
		size_t GetEdgeID(const Edge *) __attribute__ ((pure));

		std::vector<Edge*> get_outgoing_edges_of_a_vertex(vertex_t vertex);
		std::vector<Edge*> get_incoming_edges_of_a_vertex(vertex_t vertex);

		std::vector<Edge *> GetNeighbours(const Node * n);

		void reset_edge_costs();
		void set_edge_cost(const std::vector<float>& new_costs);

	private:
		std::vector<vertex_t> itsVertices;
		std::vector<Edge> itsEdges;

		std::vector<const Edge*> itsEdges_ref; //this is a cache of all the references


		void GenerateEdges(std::vector< std::vector<vertex_t> > &edges, std::vector<float> &costs);
};