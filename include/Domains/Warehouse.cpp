#include "Warehouse.h"

Warehouse::Warehouse(YAML::Node configs){
	// Read in graph definition from vertices and edges files (directories stored in configs)
	string domainDir = configs["domain"]["folder"].as<string>();
	string vFile = domainDir + configs["graph"]["vertices"].as<string>();
	string eFile = domainDir + configs["graph"]["edges"].as<string>();
	string cFile = domainDir + configs["graph"]["capacities"].as<string>();
	nSteps = configs["simulation"]["steps"].as<size_t>();

	InitialiseGraph(vFile, eFile, cFile, configs);
	outputEvals = false;
}

Warehouse::~Warehouse(void){
	delete whGraph;
	whGraph = 0;
	for (size_t i = 0; i < whAGVs.size(); i++){
		delete whAGVs[i];
		whAGVs[i] = 0;
	}
	for (size_t i = 0; i < whAgents.size(); i++){
		delete whAgents[i];
		whAgents[i] = 0;
		//TODO DELETE AGENTS
	}
	if (outputEvals)
		evalFile.close();
	if (outputEpReplay){
		agvStateFile.close();
		agvEdgeFile.close();
		agentStateFile.close();
		agentActionFile.close();
	}
}

void Warehouse::ResetEpochEvals(){
	if (algo = algo_type::ddpg)
		for (size_t i = 0; i < nAgents; i++)
			ddpg_maTeam[i]->ResetEpochEvals();
	else{
		std::cout << "ERROR: Warehouse::ResetEpochEvals() invalid algo" << std::endl;
	}
}

void Warehouse::OutputPerformance(string eval_str){
	if (evalFile.is_open())
		evalFile.close();
	evalFile.open(eval_str.c_str(),std::ios::app);

	outputEvals = true;

	std::cout << "Writing evaluation outputs to file: " << eval_str << "\n";
}

void Warehouse::OutputControlPolicies(string nn_str){
	//TODO
}

void Warehouse::OutputEpisodeReplay(string agv_s_str, string agv_e_str, string a_s_str, string a_a_str){
	if (agvStateFile.is_open())
		agvStateFile.close();
	agvStateFile.open(agv_s_str.c_str(),std::ios::app);

	if (agvEdgeFile.is_open())
		agvEdgeFile.close();
	agvEdgeFile.open(agv_e_str.c_str(),std::ios::app);

	if (agentStateFile.is_open())
		agentStateFile.close();
	agentStateFile.open(a_s_str.c_str(),std::ios::app);

	if (agentActionFile.is_open())
		agentActionFile.close();
	agentActionFile.open(a_a_str.c_str(),std::ios::app);

	outputEpReplay = true;

	std::cout << "Writing AGV logs to files: " << "\n";
	std::cout << "\t" << agv_s_str << "\n";
	std::cout << "\t" << agv_e_str << "\n";

	std::cout << "Writing agent logs to files: " << "\n";
	std::cout << "\t" << a_s_str << "\n";
	std::cout << "\t" << a_a_str << "\n";
}

void Warehouse::LoadPolicies(YAML::Node configs){
}

void Warehouse::InitialiseGraph(string v_str, string e_str, string c_str, YAML::Node configs){
	vector<int> vertices;
	vector< vector<int> > edges;
	vector< double > costs;

	// Read in data from files
	cout << "Reading vertices from file: ";
	ifstream verticesFile(v_str.c_str());
	cout << v_str.c_str() << "...";
	if (!verticesFile.is_open()){
		cout << "\nFile: " << v_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	string line;
	while (getline(verticesFile,line))
		vertices.push_back(atoi(line.c_str()));
	cout << "complete. " << vertices.size() << " vertices in graph.\n";

	cout << "Reading edges from file: ";
	ifstream edgesFile(e_str.c_str());
	cout << e_str.c_str() << "...";
	if (!edgesFile.is_open()){
		cout << "\nFile: " << e_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	while (getline(edgesFile,line)){
		stringstream lineStream(line);
		string cell;
		vector<double> ec;
		while (getline(lineStream,cell,','))
			ec.push_back(atof(cell.c_str()));
		vector<int> e;
		e.push_back((int)ec[0]);
		e.push_back((int)ec[1]);
		costs.push_back(ec[2]);
		edges.push_back(e);
	}
	cout << "complete. " << edges.size() << " edges in graph.\n";

	baseCosts = costs;

	// Read in data from files
	cout << "Reading capacities from file: ";
	ifstream capacitiesFile(c_str.c_str());
	cout << c_str.c_str() << "...";
	if (!capacitiesFile.is_open()){
		cout << "\nFile: " << c_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	while (getline(capacitiesFile,line))
		capacities.push_back((size_t)atoi(line.c_str()));
	cout << "complete.\n";

	whGraph = new Graph(vertices, edges, costs);

//	std::cout << "Number of graph vertices: " << whGraph->GetNumVertices() << "\n";
//	std::cout << "Number of graph edges: " << whGraph->GetNumEdges() << "\n";

//	InitialiseMATeam();
	InitialiseAGVs(configs);
}

void Warehouse::InitialiseAGVs(YAML::Node configs){
	string domainDir = configs["domain"]["folder"].as<string>();
	// Initialise AGV objects
	string agv_str = domainDir + configs["simulation"]["agvs"].as<string>();

	// Read in data from files
	cout << "Reading AGVs from file: ";
	ifstream AGVFile(agv_str.c_str());
	cout << agv_str.c_str() << "...";
	if (!AGVFile.is_open()){
		cout << "\nFile: " << agv_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	nAGVs = 0;
	vector<size_t> agvOrigins;
	string line;
	while (getline(AGVFile,line)){
		agvOrigins.push_back((size_t)atoi(line.c_str()));
		nAGVs++;
	}
	cout << "complete. Created " << nAGVs << " AGVs.\n";

	// Store valid destination vertex IDs
	string goal_str = domainDir + configs["simulation"]["goals"].as<string>();

	// Read in data from files
	cout << "Reading goal vertex IDs from file: ";
	ifstream goalFile(goal_str.c_str());
	cout << goal_str.c_str() << "...";
	if (!goalFile.is_open()){
		cout << "\nFile: " << goal_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	vector<int> agvGoals;
	while (getline(goalFile,line))
		agvGoals.push_back(atoi(line.c_str()));
	cout << "complete. " << agvGoals.size() << " goal vertices.\n";

//	std::cout << "Number of possible goals: " << agvGoals.size() << "\n";

	for (size_t i = 0; i < nAGVs; i++){
		AGV * agv = new AGV(agvOrigins[i], agvGoals, whGraph);
		whAGVs.push_back(agv);
	}
	assert(nAGVs == whAGVs.size());
}

void Warehouse::InitialiseNewEpoch(){
	// Reset all AGVs
	for (size_t i = 0; i < nAGVs; i++)
		whAGVs[i]->ResetAGV();
	for (size_t i = 0; i < nAgents; i++)
		whAgents[i]->agvIDs.clear();
}

vector< vector<size_t> > Warehouse::RandomiseTeams(size_t n){ // n = number of agents in each team
	vector< vector<size_t> > teams;
	vector<size_t> order;
	for (size_t i = 0; i < n; i++){
		order.push_back(i);
	}

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	for (size_t j = 0; j < nAgents; j++){
		shuffle (order.begin(), order.end(), std::default_random_engine(seed));
		teams.push_back(order);
	}

	return teams;
}

void Warehouse::UpdateGraphCosts(vector<double> costs){
	vector<Edge *> e = whGraph->GetEdges();
	for (size_t i = 0; i < e.size(); i++)
		e[i]->SetCost(costs[i]);
}

/************************************************************************************************
 * *Output:Print and number of AGVs that are on each edge					*
 ************************************************************************************************/
void Warehouse::print_warehouse_state(){
	vector<size_t> cur_state = get_edge_utilization();
	std::cout << "Warehouse utilization: {";
	for (size_t n = 0; n < N_EDGES; n++){
		if (cur_state[n] > 0 )
			std::cout<<"[e_"<< n << ", pop= " << cur_state[n] << "], ";
	}
	std::cout << '}' << std::endl;
}

/************************************************************************************************
 * *Input:[verbose] which if true will print progress						*
 * *Method:Count how for each edge how maany AGVs are on it					*
 * *Output:A vector<size_t> which contains the number of AGVs on each edge, indexed by EdgeID	*
 ************************************************************************************************/
vector<size_t> Warehouse::get_edge_utilization(bool verbose){
	vector<size_t> edge_utilization(N_EDGES);
	for(int i = 0; i < N_EDGES; i++)
		edge_utilization[i] = 0;
	for(int i = 0; i < whAGVs.size(); i++){
		Edge* e = whAGVs[i]->GetCurEdge();
		if(e != NULL){//check if AGVs are at an edge
			assert(whGraph->GetEdgeID(e) < edge_utilization.size());
			edge_utilization[whGraph->GetEdgeID(e)]++;
			if (verbose)
				std::cout<<"Edge id-"<<whGraph->GetEdgeID(e)<<
					" length: "<<e->GetLength()<< " "<<std::endl;
		}
	}
	return edge_utilization;
}

/************************************************************************************************
 * *Input:[cost_add] a vector containing aditional planing costs indexed by EdgedIDs		*
 * *Method:Replans the AGVs using Methos of the {Graph} class					*
 * ************************************************************************************************/
void Warehouse::replan_AGVs(std::vector<double> cost_add){
	// Replan AGVs as necessary
	for (size_t k = 0; k < nAGVs; k++){
		whAGVs[k]->CompareCosts(cost_add); // set replanning flags

		if (whAGVs[k]->GetIsReplan()) // replanning needed
			whAGVs[k]->PlanAGV(cost_add);

		// Identify any new AGVs that need to cross an intersection
		if (whAGVs[k]->GetT2V() == 0){
			size_t agentID = 0; // only one agent
			bool onWaitList = false;
			for (list<size_t>::iterator it = whAgents[agentID]->agvIDs.begin(); it!=whAgents[agentID]->agvIDs.end(); ++it){
				if (k == *it){
					onWaitList = true;
					break;
				}
			}
			if (!onWaitList)// only add if not already on wait list
				whAgents[agentID]->agvIDs.push_back(k);
		}
	}
}

/************************************************************************************************
 * *Method:Attempt to move any transitioning AGVs on to new edges (according to wait list order)*
 ************************************************************************************************/
void Warehouse::transition_AGVs(bool verbose){
	vector<size_t> s(N_EDGES);
	GetJointState(whGraph->GetEdges(), s);

	for (size_t k = 0; k < nAgents; k++){
		vector<size_t> toRemove;
		for (list<size_t>::iterator it = whAgents[k]->agvIDs.begin(); it!=whAgents[k]->agvIDs.end(); ++it){
			size_t curAGV = *it;
			size_t nextID = whGraph->GetEdgeID(whAGVs[curAGV]->GetNextEdge()); // next edge ID
			assert(nextID < N_EDGES);
			

			if (nextID < 0 || nextID >= s.size()){
				std::cout << "AGV #" << curAGV << ", nextID: " << nextID << "\n";
				std::cout << "	t2v: " << whAGVs[curAGV]->GetT2V() << "\n";
				std::cout << "	itsQueue: " << (whAGVs[curAGV]->GetAGVPlanner()->GetQueue() != 0) << "\n";
			}
			if (s[nextID] < capacities[nextID]){ // check if next edge is full
				// transfer to new edge and update agv counters
				size_t curID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge());
				whAGVs[curAGV]->EnterNewEdge();
				size_t newID = whGraph->GetEdgeID(whAGVs[curAGV]->GetCurEdge());
				if (curID < s.size()) // if moving off an edge
					s[curID]--; // remove from old edge
				s[newID]++; // add to new edge (note that nextID and newID should be equal!)
				if (nextID != newID)
					std::cout << "Warning: nextID [" << nextID << "] != newID [" << newID << "]\n";
				toRemove.push_back(curAGV); // remember to remove from agent wait list
				if (verbose)
					std::cout << "AGV in: " << curID << " edge" <<
						"is entering: " << nextID << std::endl;
			}
		}
		for (size_t w = 0; w < toRemove.size(); w++)
			whAgents[k]->agvIDs.remove(toRemove[w]);
	}
}

void Warehouse::GetJointState(vector<Edge *> e, vector<size_t> &s){
	for (size_t i = 0; i < nAGVs; i++){
		Edge * curEdge = whAGVs[i]->GetCurEdge();
		size_t j = whGraph->GetEdgeID(curEdge);
		if (j < s.size())
			s[j]++;
	}
}

