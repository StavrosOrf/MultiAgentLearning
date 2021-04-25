#include "Queue.h"

void Queue::UpdateQueue(Node * newNode){
	itsPQ.push(newNode);
}

Node * Queue::PopQueue(){
	closed.push_back(itsPQ.top());
	itsPQ.pop();
	assert(closed.back() == closed[closed.size()-1]);
	return closed[closed.size()-1];
}

void Queue::delete_queue() {
	while(!itsPQ.empty()){
		assert(itsPQ.top());
		Node * temp = itsPQ.top();
		delete temp;
		temp = 0;
		itsPQ.pop();
	}
}

void Queue::delete_closed() {
	for (size_t i = 0; i < closed.size(); i ++){
		assert (closed[i]);
		delete closed[i];
		closed[i] = 0;
	}
}

void Queue::reset(){
	delete_queue();
	delete_closed();
	itsPQ = QUEUE(); 
	//itsPQ.clear();
	closed.clear(); 
	assert(itsPQ.empty() && closed.empty());
}