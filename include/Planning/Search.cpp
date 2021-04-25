#include "Search.h"

Node * Search::PathSearch(){
	size_t sourceID = FindSourceID();
	//assert(!itsQueue);
	assert(itsQueue->EmptyQueue());
	//itsQueue = new Queue(new Node(itsGraph->GetVertices()[sourceID], nodeType::SOURCE));
	itsQueue->UpdateQueue(new Node(itsGraph->GetVertices()[sourceID], nodeType::SOURCE));
	bool pathFound = false;

	while (!itsQueue->EmptyQueue()){
		// Pop cheapest node from queue
		Node * currentNode = itsQueue->PopQueue();

		// Terminate search once path is found
		if (currentNode->GetVertex() == itsGoal){
			pathFound = true;
			break;
		}

		// Find all neighbours excluding ancestor vertices if any
		std::vector<Edge *> neighbours = itsGraph->GetNeighbours(currentNode);

		// Update neighbours
		for (size_t i = 0; i < neighbours.size(); i++){
			// Create neighbour node
			Node * currentNeighbour = new Node(currentNode, neighbours[i]);
			itsQueue->UpdateQueue(currentNeighbour);
		}
	}

	if (!pathFound){
		std::cout << "No path found from source to goal. Exiting.\n";
		exit(1);
	}else{
		Node * bestPath = 0;
		//assert(itsQueue->EmptyQueue());
		//itsQueue->delete_queue();

		for (size_t i = 0; i < itsQueue->GetClosed().size(); i++)
			if (itsGoal == itsQueue->GetClosed()[i]->GetVertex()){
				bestPath = itsQueue->GetClosed()[i];
				break;
			} else {
				//delete itsQueue->GetClosed()[i];
				//itsQueue->GetClosed()[i] = NULL;
			}

		assert(bestPath != NULL);
		return bestPath;
	}
}

//TODO(Kallinteris) improve performance by reseting existing queue
void Search::ResetSearch(){
	assert(itsQueue);
	itsQueue->reset();
	//delete itsQueue;
	//itsQueue = new Queue();
}

size_t Search::FindSourceID(){
	for (size_t i = 0; i < itsGraph->GetNumVertices(); i++)
		if (itsSource == itsGraph->GetVertices()[i])
			return i;
	std::cout << "Error: souce ID not found. Exiting.\n";
	exit(1);
}

//returns the lenght of the optimal path
float Search::PathSearchLenght(){
	float total = 0;
	for (Node *n = PathSearch(); n != NULL;){
		total += n->GetCost();
		Node* t = n;
		n = n->GetParent();
		delete t;
	}
	return total;
}

