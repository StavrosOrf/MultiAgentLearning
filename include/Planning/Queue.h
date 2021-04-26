#pragma once

#include <vector> // std::vector
#include <queue> // std::priority_queue
#include "Node.h"
#include <boost/heap/priority_queue.hpp>

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
		//typedef boost::heap::priority_queue<Node *> QUEUE;
		Queue(Node * source){
			itsPQ.push(source);
			//closed.reserve(100);
		}
		Queue(){}

		~Queue(){
			delete_queue();
			delete_closed();
		}

		std::vector<Node *> GetClosed() const {return closed;}
		bool EmptyQueue() const {return itsPQ.empty();}
		size_t SizeQueue() const {return itsPQ.size();}
		void UpdateQueue(Node * newNode);
		Node * PopQueue();
		void reset();
	private:
		QUEUE itsPQ;
		std::vector<Node *> closed;
		void delete_queue();
		void delete_closed();
};