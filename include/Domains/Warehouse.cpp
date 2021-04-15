#include "Warehouse.hpp"

Warehouse::Warehouse(YAML::Node configs){
	// Read in graph definition from vertices and edges files (directories stored in configs)
	std::string domainDir = configs["domain"]["folder"].as<std::string>() +
		configs["domain"]["warehouse"].as<std::string>() + '/';
	std::string vFile = domainDir + configs["graph"]["vertices"].as<std::string>();
	std::string eFile = domainDir + configs["graph"]["edges"].as<std::string>();
	std::string cFile = domainDir + configs["graph"]["capacities"].as<std::string>();
	nSteps = configs["simulation"]["steps"].as<size_t>();

	if(configs["mode"]["algo"].as<std::string>() == "DDPG")
		algo = algo_type::ddpg;
	else if(configs["mode"]["algo"].as<std::string>() == "DDPG_MERGED")
		algo = algo_type::ddpg_merged;
	else{
		std::cout << "ERROR: Currently only configured for 'DDPG' and ''! Exiting.\n";
		exit(1);
	}
	std::string agentType = configs["domain"]["agents"].as<std::string>();

	if(agentType.starts_with("centralized"))
		agent_type = agent_def::centralized;
	else if(agentType.starts_with("link"))
		agent_type = agent_def::link;
	else if(agentType.starts_with("intersection"))
		agent_type = agent_def::intersection;
	else{
		std::cout<<"Error in agent definition"<<std::endl;
		exit(1);
	}
	incorporates_time = agentType.ends_with("_t");

	InitialiseGraph(vFile, eFile, cFile, configs);
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
	}
}

void Warehouse::InitialiseGraph(std::string v_str, std::string e_str, std::string c_str, YAML::Node configs, bool verbose){
	std::vector<vertex_t> vertices;
	std::vector<std::vector<vertex_t>> edges;
	std::vector< float > costs;

	// Read in data from files
	if (verbose)
		std::cout << "Reading vertices from file: ";
	std::ifstream verticesFile(v_str.c_str());
	if (verbose)
		std::cout << v_str.c_str() << "...";
	if (!verticesFile.is_open()){
		std::cout << "\nFile: " << v_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	std::string line;
	while (getline(verticesFile,line))
		vertices.push_back(atoi(line.c_str()));
	if (verbose)
		std::cout << "complete. " << vertices.size() << " vertices in graph.\n";

	if (verbose)
		std::cout << "Reading edges from file: ";
	std::ifstream edgesFile(e_str.c_str());
	if (verbose)
		std::cout << e_str.c_str() << "...";
	if (!edgesFile.is_open()){
		std::cout << "\nFile: " << e_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	while (getline(edgesFile,line)){
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<float> ec;
		while (getline(lineStream,cell,','))
			ec.push_back(atof(cell.c_str()));
		std::vector<vertex_t> e;
		e.push_back((vertex_t)ec[0]);
		e.push_back((vertex_t)ec[1]);
		costs.push_back(ec[2]);
		edges.push_back(e);
	}
	if (verbose)
		std::cout << "complete. " << edges.size() << " edges in graph.\n";

	baseCosts = costs;

	// Read in data from files
	if (verbose)
		std::cout << "Reading capacities from file: ";
	std::ifstream capacitiesFile(c_str.c_str());
	if (verbose)
		std::cout << c_str.c_str() << "...";
	if (!capacitiesFile.is_open()){
		std::cout << "\nFile: " << c_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	while (getline(capacitiesFile,line))
		capacities.push_back((size_t)atoi(line.c_str()));
	if (verbose)
		std::cout << "complete.\n";

	whGraph = new Graph(vertices, edges, costs);

	initialise_wh_agents();
	//InitialiseMATeam(); MATeam must be initialed outside the constractor
	InitialiseAGVs(configs);
}

void Warehouse::InitialiseAGVs(YAML::Node configs, bool verbose){
	std::string domainDir = configs["domain"]["folder"].as<std::string>() +
		configs["domain"]["warehouse"].as<std::string>() + '/';
	// Initialise AGV objects
	std::string agv_str = domainDir + configs["simulation"]["agvs"].as<std::string>();

	// Read in data from files
	if (verbose)
		std::cout << "Reading AGVs from file: ";
	std::ifstream AGVFile(agv_str.c_str());
	if (verbose)
		std::cout << agv_str.c_str() << "...";
	if (!AGVFile.is_open()){
		std::cout << "\nFile: " << agv_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	size_t nAGVs = 0;
	std::vector<size_t> agvOrigins;
	std::string line;
	while (getline(AGVFile,line)){
		agvOrigins.push_back((size_t)atoi(line.c_str()));
		nAGVs++;
	}
	if (verbose)
		std::cout << "complete. Created " << nAGVs << " AGVs.\n";

	// Store valid destination vertex IDs
	std::string goal_str = domainDir + configs["simulation"]["goals"].as<std::string>();

	// Read in data from files
	if (verbose)
		std::cout << "Reading goal vertex IDs from file: ";
	std::ifstream goalFile(goal_str.c_str());
	if (verbose)
		std::cout << goal_str.c_str() << "...";
	if (!goalFile.is_open()){
		std::cout << "\nFile: " << goal_str.c_str() << " not found, exiting.\n";
		exit(1);
	}
	std::vector<vertex_t> agvGoals;
	while (getline(goalFile,line))
		agvGoals.push_back(atoi(line.c_str()));
	if (verbose)
		std::cout << "complete. " << agvGoals.size() << " goal vertices.\n";

//	std::cout << "Number of possible goals: " << agvGoals.size() << "\n";

	for (size_t i = 0; i < nAGVs; i++){
		AGV * agv = new AGV(agvOrigins[i], agvGoals, whGraph);
		whAGVs.push_back(agv);
	}
	//assert(nAGVs == whAGVs.size());
}

/************************************************************************************************
 **Method:Resets all AGVs to there Original position						*
 ************************************************************************************************/
void Warehouse::InitialiseNewEpoch(){
	for (size_t i = 0; i < whAGVs.size(); i++)// Reset all AGVs
		whAGVs[i]->ResetAGV();
	for (size_t i = 0; i < whAgents.size(); i++)
		whAgents[i]->agvIDs.clear();
}

/************************************************************************************************
 **Input:a vector of [costs]									*
 **Method:Sets the Cost of the graph Edges to the values provides by [costs]			*
 *************************************************************************************************/
void Warehouse::UpdateGraphCosts(std::vector<float> costs){
	for (size_t i = 0; i < N_EDGES; i++)
		whGraph->GetEdges()[i]->SetCost(costs[i]);
}

/************************************************************************************************
 * *Output:Print and number of AGVs that are on each edge					*
 ************************************************************************************************/
void Warehouse::print_warehouse_state(){
	const std::vector<float> cur_state = get_edge_utilization(true);
	std::cout << "Warehouse edge utilization: {";
	for (size_t n = 0; n < N_EDGES; n++)
		if (cur_state[n] > 0 )
			std::cout<<"[e_"<< n << "p=" << cur_state[n] << "], ";
	// for (size_t n = 0; n < N_EDGES; n++)
	// 	if (cur_state[n+N_EDGES] > 0 )
	// 		std::cout<<"[e_"<< n << "t=" << cur_state[n+N_EDGES] << "], ";
	std::cout << '}' << std::endl;

	std::cout << "Warehouse Vertex utilization: {";
	for (size_t n = 0; n < whGraph->GetNumVertices(); n++)
		if (get_vertex_utilization()[n])
			std::cout<<"[v_"<< n << "p=" << get_vertex_utilization()[n] << "], ";
	std::cout << '}' << std::endl;
}

/************************************************************************************************
*Input:	[verbose] which if true will print progress, [normalize] which if true normalized the	*
*	outputs to [0,1]									*
*Method:Count for each edge how many AGVs are on it and if it [incorporates_time] also		*
*	observe what is the remaining time of the closest to finish AGV,			*
*	finnaly if [normalize] is true, divide edge count by the maximum base travel among all	*
*	the edges and (if [incorporates_time] is true) divide edge time left with it's capacity	*
*Output:A vector<size_t> which contains the number of AGVs on each edge, indexed by EdgeID	*
*	And the minimum remaining distance of AGVs on edge indexed by EdgeID+N_EDGES		*
************************************************************************************************/
std::vector<float> Warehouse::get_edge_utilization(){return get_edge_utilization(incorporates_time, true);}
std::vector<float> Warehouse::get_edge_utilization(bool care_about_time, bool normalize){
	std::vector<float> edge_utilization(N_EDGES * (1+(care_about_time)),0);
	if (care_about_time)
		for(size_t i = N_EDGES; i < N_EDGES*2; i++)
			edge_utilization[i] = std::numeric_limits<float>::infinity();

	for(AGV* a: whAGVs)
		if(a->is_on_edge()){
			const int Edge_ID = whGraph->GetEdgeID(a->GetCurEdge());
			//assert(Edge_ID < edge_utilization.size()/(1+(care_about_time)));
			edge_utilization[Edge_ID]++;
			if (care_about_time)
				edge_utilization[Edge_ID+N_EDGES] = std::min<float>(
						edge_utilization[Edge_ID+N_EDGES], a->GetT2V());
		}

	if (care_about_time)
		for(size_t i = N_EDGES; i < N_EDGES*2; i++)
			if (std::isinf(edge_utilization[i]))
				edge_utilization[i] = 0;
	if (care_about_time)
		for(size_t i = 0; i < N_EDGES; i++)
			assert((edge_utilization[i+N_EDGES]==0) == (edge_utilization[i]==0));

	// if(normalize){//TODO update doc
	// 	for (int i = 0; i != N_EDGES; i++){
	// 		edge_utilization[i] /= max_base_travel_cost();
	// 		// edge_utilization[i] = edge_utilization[i]*2-1;//normalize to -[1,1]
	// 		 edge_utilization[i] = edge_utilization[i]*1.8-0.9;//normalize to [-0.9,0.9]
	// 	}
	// 	if(care_about_time){
	// 		for(size_t i = N_EDGES; i < N_EDGES*2; i++){
	// 			edge_utilization[i] /= capacities[i];
	// 			// edge_utilization[i] = edge_utilization[i]*2-1;
	// 			edge_utilization[i] = edge_utilization[i]*1.8-0.9;
	// 		}
				
	// 	}
	// }

	return edge_utilization;
}

/************************************************************************************************
**Input:[cost_add] a vector containing aditional planing costs indexed by EdgedIDs		*
**Method:Replans the AGVs using Methos of the {Graph} class					*
*************************************************************************************************/
void Warehouse::replan_AGVs(std::vector<float> cost_add){
	// Replan AGVs as necessary
	for (size_t k = 0; k < whAGVs.size(); k++){
		whAGVs[k]->CompareCosts(cost_add); // set replanning flags
		whAGVs[k]->reset_edge_enter();

		if (whAGVs[k]->GetIsReplan()) // replanning needed
			whAGVs[k]->PlanAGV(cost_add);

		// Identify any new AGVs that need to cross an intersection
		if (whAGVs[k]->GetT2V() == 0){
			size_t agentID = 0; // only one agent
			bool onWaitList = false;
			for (std::list<size_t>::iterator it = whAgents[agentID]->agvIDs.begin(); it!=whAgents[agentID]->agvIDs.end(); ++it)
				if (k == *it){
					onWaitList = true;
					break;
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
	std::vector<size_t> s(N_EDGES);
	GetJointState(s);

	for (size_t k = 0; k < whAgents.size(); k++){
		std::vector<size_t> toRemove;
		for (std::list<size_t>::iterator it = whAgents[k]->agvIDs.begin(); it!=whAgents[k]->agvIDs.end(); ++it){
			size_t curAGV = *it;
			size_t nextID = whGraph->GetEdgeID(whAGVs[curAGV]->GetNextEdge()); // next edge ID
			assert(nextID < N_EDGES);


			assert(nextID >= 0);
			if (nextID >= s.size()){
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

/************************************************************************************************
 * *Input:A vector containing the [final_costs] of the edges					*
 * *Method:Executes all the required steps to plan and traverse the AGVs on the warehouse	*
 * ************************************************************************************************/
void Warehouse::traverse_one_step(std::vector<float> final_costs){
	UpdateGraphCosts(final_costs);
	replan_AGVs(final_costs);
	transition_AGVs();
	for (size_t k = 0; k < whAGVs.size(); k++)// Traverse
		whAGVs[k]->Traverse();
}

 // Initialise NE components and domain housekeeping components of the agent type
void Warehouse::initialise_wh_agents(){
	assert(whAgents.empty());
	if(agent_type == agent_def::centralized){
		std::vector<size_t> eIDs;
		for (size_t j = 0; j < N_EDGES; j++)
			eIDs.push_back(j);
		whAgents.push_back(new iAgent{0, eIDs});// only one centralised agent controlling all traffic
	}else if(agent_type == agent_def::link){
		std::vector<vertex_t> v = whGraph->GetVertices();
		std::vector<Edge *> e = whGraph->GetEdges();
		for (size_t i = 0; i < e.size(); i++){
			std::vector<size_t> vIDs;
			vIDs.push_back(e[i]->GetVertex1());
			vIDs.push_back(e[i]->GetVertex2());
			whAgents.push_back(new iAgent{i, vIDs});
		}
	}else if(agent_type == agent_def::intersection)
		for (size_t v : whGraph->GetVertices()){
			std::vector<size_t> eIDs;
			for (size_t i = 0; i != whGraph->GetEdges().size(); i++)
				if ((size_t) whGraph->GetEdges()[i]->GetVertex2() == v)
					eIDs.push_back(i);
			whAgents.push_back(new iAgent{v, eIDs});
		}
}


void Warehouse::GetJointState(std::vector<size_t> &s){
	for (size_t i = 0; i < whAGVs.size(); i++){
		Edge * curEdge = whAGVs[i]->GetCurEdge();
		size_t j = whGraph->GetEdgeID(curEdge);
		if (j < s.size())
			s[j]++;
	}
}

/************************************************************************************************
**Input: [include_start_vertexes] which if true it will AGVs that outside the graph,		*
	[goal_vertex] which if inputed it will count AGVs that have that goal_vertex		*
**Method:Checks how many AGVs are on each Edge, if AGVs are outside the graph they count as	*
**	Being on their origin vertex								*
**Output:The number of AGVs on each Vertex							*
*************************************************************************************************/
std::vector<vertex_t> Warehouse::get_vertex_utilization(const bool include_start_vertexes){
	std::vector<vertex_t> vertex_population(whGraph->GetNumEdges());

	for (AGV* a: whAGVs)
		if (a->is_on_graph() && a->GetT2V() == 0)
			vertex_population[a->get_cur_vertex()]++;
		else if(include_start_vertexes && !a->is_on_graph())
			vertex_population[a->get_start_vertex()]++;

	return vertex_population;
}
std::vector<vertex_t> Warehouse::get_vertex_utilization(const vertex_t goal_vertex, const bool include_start_vertexes){
	std::vector<vertex_t> vertex_population(whGraph->GetNumEdges());

	for (AGV* a: whAGVs)
		if (a->get_goal_vertex() == goal_vertex){
			if (a->is_on_graph() && a->GetT2V() == 0)
				vertex_population[a->get_cur_vertex()]++;
			else if(include_start_vertexes && !a->is_on_graph())
				vertex_population[a->get_start_vertex()]++;
		}

	return vertex_population;
}

float Warehouse::get_vertex_remaining_outgoing_capacity(vertex_t vertex){
	float t = 0;
	for (Edge* e : whGraph->get_outgoing_edges_of_a_vertex(vertex))
		t += capacities[whGraph->GetEdgeID(e)] - 
			get_edge_utilization(false)[whGraph->GetEdgeID(e)];
	return t;
}


/*******************************
** Print Agvs Paths
********************************/
void Warehouse::printAgvPaths(){
	for (size_t k = 0; k < whAGVs.size(); k++){
		std::cout<<", AGV_"<<k<<":";
		whAGVs[k]->DisplayPath();
	}
	std::cout<<'\n';
}



/************************************************************************************************
**Waring:This function Reset the current edge cost, must be not be called in between traversing *
**Note: is only implemented for a warehouse in which all the AGVs have the same possible goals	*
**Method:Returns an evalutation of the current based on the position of the AGVs (in relation	*
**	to there physical distance of their destination)					*
**Note:	Can be used as part of a reward or evaluation function					*
**Output:0 indicates the worst possible state, higher numbers indicate the current state is	*
**	better/more valuable									*
**TODO: optimize Perfomance (with LUTs?)							*
*************************************************************************************************/
float Warehouse::value_of_current_state(){
	whGraph->reset_edge_costs();
	float total = 0;

	for (AGV* a : whAGVs)
		if (a->is_on_edge()){
			Search s0(whGraph, a->get_start_vertex(), a->GetNextVertex());
			Search s1(whGraph, a->GetNextVertex(), a->get_goal_vertex());
			float total_inverse = 0;
			total_inverse += s0.PathSearchLenght();
			total_inverse += s1.PathSearchLenght();
			total += 1/total_inverse;
		}
	
	{//Assert that all AGVs have the same possible 2 goals 
		assert(whAGVs[0]->get_possible_goals().size() == 2);
		for (AGV* a : whAGVs)
			assert(a->get_possible_goals() == whAGVs[0]->get_possible_goals());
	}

	const std::vector<vertex_t> goals = whAGVs[0]->get_possible_goals();
	for (vertex_t g : whAGVs[0]->get_possible_goals())
		//for (vertex_t v : get_vertex_utilization(g, false))
		for (int i = 0; i != whGraph->GetNumVertices(); i++){
			if (get_vertex_utilization(g, false)[i]){
				int source = 0;
				if (goals[0] == g)
					source = goals[1];
				else if (goals[1] == g)
					source = goals[1];
				else {assert(false);}
				Search s0(whGraph, source, i);
				Search s1(whGraph, i, g);
				float total_inverse = 0;
				total_inverse += s0.PathSearchLenght();
				total_inverse += s1.PathSearchLenght();
				int max_length = 0;
				for (Edge * e: whGraph->get_outgoing_edges_of_a_vertex(i))
					max_length = std::max<int>(max_length, e->GetLength());
				total += max_length/total_inverse;
			}
		}

	return total;
}