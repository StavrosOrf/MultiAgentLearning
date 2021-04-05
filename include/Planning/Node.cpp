#include "Node.h"

void Node::DisplayPath()
{
	std::cout << "Vertex: " << itsVertex << "\n";
	std::cout << "	cost-to-come: " << itsCost << "\n";

	if (itsParent)
		itsParent->DisplayPath();
}

Node * Node::ReverseList(Node * itsChild) // Can cause memleak if returned node pointer is not deleted properly
{
	Node * itsParentR = new Node(GetVertex());
	itsParentR->SetCost(GetCost());
	itsParentR->SetParent(itsChild);

	if (GetParent())
	{
		Node * itsReverse = GetParent()->ReverseList(itsParentR);
		return itsReverse;
	}
	else
		return itsParentR;
}
