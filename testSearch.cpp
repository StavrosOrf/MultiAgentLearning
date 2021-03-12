#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <string>
#include "Planning/Graph.h"
#include "Planning/Search.h"
#include "Planning/Node.h"

using std::vector ;
using std::string ;
using std::ifstream ;
using std::stringstream ;

int main(){
  std::cout << "Testing Search class in Search.h\n" ;
  std::cout << "Initialising graph...\n" ;
  
  // Read in data from files
  char v_str[] = "v_test.txt" ;
  cout << "Reading vertices from file: " ;
  ifstream verticesFile(v_str) ;
  cout << v_str << "..." ;
  if (!verticesFile.is_open()){
    cout << "\nFile: " << v_str << " not found, exiting.\n" ;
    return 1 ;
  }
  vector<int> vertices ;
  string line ;
  while (getline(verticesFile,line))
  {
    vertices.push_back(atoi(line.c_str())) ;
  }
  cout << "complete.\n" ;
  
  char e_str[] = "e_test.txt" ;
  cout << "Reading edges from file: " ;
  ifstream edgesFile(e_str) ;
  cout << e_str << "..." ;
  if (!edgesFile.is_open()){
    cout << "\nFile: " << e_str << " not found, exiting.\n" ;
    return 1 ;
  }
  vector< vector<int> > edges ;
  vector<double> costs ;
  while (getline(edgesFile,line))
  {
    stringstream lineStream(line) ;
    string cell ;
    vector<double> ec ;
    while (getline(lineStream,cell,','))
    {
	    ec.push_back(atof(cell.c_str())) ;
    }
    vector<int> e ;
    e.push_back((int)ec[0]) ;
    e.push_back((int)ec[1]) ;
    costs.push_back(ec[2]) ;
    edges.push_back(e) ;
  }
  cout << "complete.\n" ;
  
  Graph * testGraph = new Graph(vertices, edges, costs) ;
  Search * testSearch = new Search(testGraph, 8, 3) ;
  Node * bestPath = testSearch->PathSearch() ;
  Node * pathSG = bestPath->ReverseList(0) ;
  pathSG->DisplayPath() ;
  
  vector<Edge *> graphEdges = testGraph->GetEdges() ;
  bool bOut = false ;
  if (graphEdges[0] == graphEdges[1])
    bOut = true ;
  std::cout << "graphEdges[0] == graphEdges[1]: " << bOut << "\n" ;
  bOut = false ;
  if (graphEdges[0] == graphEdges[0])
    bOut = true ;
  std::cout << "graphEdges[0] == graphEdges[0]: " << bOut << "\n" ;
  
  delete testGraph ;
  testGraph = 0 ;
  delete testSearch ;
  testSearch = 0 ;
  
  while (pathSG->GetParent()){
    Node * curNode = pathSG->GetParent() ;
    delete pathSG ;
    pathSG = curNode ;
  }
  delete pathSG ;
  pathSG = 0 ;
  
  std::cout << "Testing complete!\n" ;
  return 0 ;
}
