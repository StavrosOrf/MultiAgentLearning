#pragma once

#include <vector> // std::vector
#include <queue> // std::priority_queue
#include "Node.h"

struct CompareNode{
	bool operator() (const Node * n1, const Node * n2) const{
		float n1Cost = n1->GetCost();
		float n2Cost = n2->GetCost();
		return (n2Cost < n1Cost);
	}
};

// Custom queue type to perform priority queue updates
class Queue{
	public:
		typedef std::priority_queue<Node *, std::vector<Node *>, CompareNode> QUEUE;
		Queue(Node * source){
			itsPQ.push(source);
		}

		~Queue(){
			while (!itsPQ.empty()){
				Node * temp = itsPQ.top();
				delete temp;
				temp = 0;
				itsPQ.pop();
			}
			for (size_t i = 0; i < closed.size(); i ++){
				delete closed[i];
				closed[i] = 0;
			}
		}

		std::vector<Node *> GetClosed() const {return closed;}
		bool EmptyQueue() const {return itsPQ.empty();}
		size_t SizeQueue() const {return itsPQ.size();}
		void UpdateQueue(Node * newNode);
		Node * PopQueue();
	private:
		QUEUE itsPQ;
		std::vector<Node *> closed;
};