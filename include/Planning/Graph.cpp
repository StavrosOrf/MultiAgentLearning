#include "Graph.h"

/************************************************************************************************
**Input:An Edge [e]										*
**Output:Returns the ID of that edge as indexed by the graph, returns the number of edges in	*
**		case the edge is not a valid edge (is NULL or exists in a different graph)	*
*************************************************************************************************/
size_t Graph::GetEdgeID(Edge * e){
	for (size_t i = 0; i < numEdges; i++)
		if (e == itsEdges[i])
			return i;
	//std::cout << "Error: edge not found in graph!\n";
	return numEdges;
}

std::vector<Edge *> Graph::GetNeighbours(Node * n){ // Do not include parent vertex in list of neighbours
	std::vector<Edge *> neighbours;
	int v = n->GetVertex();

	for (size_t i = 0; i < numEdges; i++){
		int v1 = itsEdges[i]->GetVertex1();
		int v2 = itsEdges[i]->GetVertex2();
		if (v1 == v){
			bool isNeighbour = true;
			Node * n0 = n;
			while (n0->GetParent()){
				n0 = n0->GetParent();
				int v0 = n0->GetVertex();
				if (v2 == v0){
					isNeighbour = false;
					break;
				}
			}
			if (isNeighbour)
				neighbours.push_back(itsEdges[i]);
		}
	}

	return neighbours;
}

void Graph::GenerateEdges(std::vector< std::vector<int> > &edges, std::vector<float> &costs)
{
	numEdges = edges.size();

	for (size_t i = 0; i < numEdges; i++)
	{
		// Initial error checking
		int v1 = edges[i][0];
		int v2 = edges[i][1];
		bool v1Found = false;
		bool v2Found = false;
		for (size_t j = 0; j < numVertices; j++){
			if (v1 == itsVertices[j]){
				v1Found = true;
			}
			if (v2 == itsVertices[j]){
				v2Found = true;
			}
			if (v1Found && v2Found){
				break;
			}
		}
		if (!v1Found || !v2Found){
			std::cout << "Error: edge (" << edges[i][0] << "," << edges[i][1] << ") not found. Mismatch in edges and vertices! Exiting.\n";
			exit(1);
		}

		Edge * e = new Edge(v1, v2, costs[i]);
		itsEdges.push_back(e);
	}
}

void Graph::reset_edge_costs(){
	for(Edge* e : itsEdges)
		e->SetCost(e->GetLength());
}

std::vector<Edge*> Graph::get_incoming_edges_of_a_vertex(int vertex){
	std::vector<Edge*> edges;
	for(Edge* e : itsEdges)
		if (e->GetVertex2() == vertex)
			edges.push_back(e);
	edges.shrink_to_fit();
	return edges;
}

std::vector<Edge*> Graph::get_outgoing_edges_of_a_vertex(int vertex){
	std::vector<Edge*> edges;
	for(Edge* e : itsEdges)
		if (e->GetVertex1() == vertex)
			edges.push_back(e);
	edges.shrink_to_fit();
	return edges;
}
