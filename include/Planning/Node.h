#pragma once

#include <iostream> // std::cout, std::endl
#include <limits>
#include "Edge.h"

enum class nodeType {SOURCE, OTHER};

// Node class to maintain path information up to a vertex
// contains parent node, (mu, sigma) of cost-to-come
class Node{
	public:
		Node(vertex_t vertex): itsCost(0.0), itsVertex(vertex), itsParent(0) {}


		Node(vertex_t vertex, nodeType n):
		itsVertex(vertex), itsParent(0){
			switch (n){
				case nodeType::SOURCE:
					itsCost = 0.0;
					break;
				default:
					itsCost = std::numeric_limits<double>::max();
			}
		}

		Node(Node * parent, Edge * edge): itsParent(parent){
			itsVertex = edge->GetVertex2();
			itsCost = itsParent->GetCost() + edge->GetCost();
		}

		~Node() = default;
		//Node(const Node &) = delete;

		Node * GetParent() const {return itsParent;}
		float GetCost() const {return itsCost;}
		void SetCost(float cost) {itsCost = cost;}
		vertex_t GetVertex() const {return itsVertex;}
		//void SetVertex(vertex_t vertex) {itsVertex = vertex;}

		void DisplayPath();
		Node * ReverseList(Node * itsChild) [[deprecated("this is a dump function no reason to use it")]];
		bool operator<(Node&& other) noexcept{return this->itsCost < other.itsCost;}
	private:
		void SetParent(Node * parent) {itsParent = parent;}
		float itsCost;
		vertex_t itsVertex;
		Node * itsParent;
};