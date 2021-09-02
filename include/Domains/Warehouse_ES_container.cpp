#include "Warehouse_ES_container.hpp"

Warehouse_ES_container::Warehouse_ES_container(YAML::Node configs){
	epoch = configs["ES"]["epochs"].as<int>();
	learning_rate = configs["ES"]["learning_rate"].as<float>();
	const size_t population_size = configs["ES"]["population_size"].as<int>();
	N_proc_std_dev = configs["DDPG"]["rand_proc_std_dev"].as<float>();
	assert(N_proc_std_dev > 0 && !std::isinf(N_proc_std_dev));
	assert(population_size > 0 && epoch > 0);

	population.reserve(population_size);
	for(size_t i = 0 ; i < population_size; i++){
		population.push_back(new Warehouse_ES(configs));
		population[i]->InitialiseMATeam();
	}
}

uint Warehouse_ES_container::evolution_strategy(const size_t n_threads, bool verbose,size_t run , std::ofstream &file){
	//get initial random policy
	std::vector<esNN*> team = population[0]->produce_random_team_NNs(); 
	std::vector<esNN*> best_team_policy = population[0]->produce_random_team_NNs(); //Allocating a new memory space for this

	int best_epoch = 0;

	// load_best_team_policy(team);

	std::vector<epoch_resultsES> results(population.size());
	epoch_resultsES max_results;
	// typedef std::chrono::high_resolution_clock clockTotal;
	auto startRun = std::chrono::high_resolution_clock::now();

	float avg_G;

	uint max_deliveries_intra = 0;
	for (int e = 0; e < epoch; e++){	
		// max_results.totalDeliveries = 0;
		avg_G = 0;
		auto startEpochh = std::chrono::high_resolution_clock::now();

		if (n_threads == 1)
			for (size_t j = 0; j < population.size(); j++){
				population[j]->set_team_NNs(team);
				results[j] = population[j]->SimulateEpochES();	
			}
		else if (n_threads > 1){
			boost::asio::thread_pool simulator_pool(n_threads);
			for (size_t j = 0; j < population.size(); j++){
				auto train_task = [j, team, &results, this](){
					population[j]->set_team_NNs(team);
					results[j] = population[j]->SimulateEpochES();	
				};
				boost::asio::post(simulator_pool, train_task);
			}
			simulator_pool.join();
		}

		max_results = results[0];
		//Calculate statistics for every epoch
		for (epoch_resultsES r : results){
			avg_G += r.totalDeliveries;			
			max_results.totalDeliveries = std::max<uint>(max_results.totalDeliveries, r.totalDeliveries);
			if(max_results.totalDeliveries < r.totalDeliveries)
				max_results = r;				
		}
		avg_G = avg_G/population.size();
		//if we have a better NN team save it as best TODO

		if(max_deliveries_intra < max_results.totalDeliveries){
			max_deliveries_intra = max_results.totalDeliveries;
			copy_best_team_policy(team,best_team_policy);			
			best_epoch = e;
			std::cout<<"New best team model G: "<<max_deliveries_intra <<"\n";

		}
		
		/*Initialize Sum Tensor
		*sum's outer vector represents each Agent's NN
		* and the inner vector represents each NN's layers' parameters */

		std::vector<std::vector<torch::Tensor>> sum(team.size()) ;
		for (size_t i = 0; i < team.size(); i++){
			// std::vector<torch::Tensor>> temp(team[i][i]->NN->parameters().size());
			sum[i].resize(team[i]->parameters().size());
			for (size_t j = 0; j < team[i]->parameters().size(); j++ )
				sum[i][j] = torch::zeros(team[i]->parameters()[j].sizes());				
		}

		for (epoch_resultsES r : results)
			for (size_t i = 0; i < team.size(); i++)
				for (size_t j = 0; j < team[i]->parameters().size(); j++ )
					sum[i][j] += (float)r.totalDeliveries * r.samples[i][j];					

		for (size_t i = 0; i < team.size(); i++)
			for (size_t j = 0; j < team[i]->parameters().size(); j++ ){
				float delta = 1/ (((float)population.size()) * (float)N_proc_std_dev);
				sum[i][j] = learning_rate * sum[i][j] * delta ;
				// std::cout<<torch::sum(sum[i][j])<<"\n";
				team[i]->parameters()[j].set_data(team[i]->parameters()[j].detach().clone() + sum[i][j])  ;
			}
		auto finishEpochh = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finishEpochh - startEpochh;
	
		if(verbose)
			std::printf("- Epoch: %3d - ( %6.2f sec) =================== Avg G: %6.2f ----- max G: %d \n",e,elapsed.count(),avg_G,max_results.totalDeliveries);
			
						
		//Write results to file
		file << ","<<e<<","<<max_results.totalDeliveries<<","<<max_results.totalMove <<","<<max_results.totalEnter <<","<<max_results.totalWait <<","<<avg_G <<std::endl;
	}
	save_best_team_policy(best_team_policy,best_epoch,max_deliveries_intra);

	auto endRun = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedT = endRun - startRun;
	std::cout<<"Total time elapsed for Run "<<run<<" ( "<<elapsedT.count()<<" sec) ----- MAX G: "<<max_deliveries_intra<<std::endl; 
	
	for (esNN* nn : team){
		delete nn;
		nn = NULL;
	}
	for (esNN* nn : best_team_policy){
		delete nn;
		nn = NULL;
	}

	return max_deliveries_intra;
}


//copy a vector of NNs to another vector of NNs
void Warehouse_ES_container::copy_best_team_policy(std::vector<esNN*> sourceNNs,std::vector<esNN*> targetNNs){

	for (size_t i = 0; i < targetNNs.size(); i++)
		for (size_t j = 0; j < targetNNs[i]->parameters().size(); j++ )
			targetNNs[i]->parameters()[j].set_data(sourceNNs[i]->parameters()[j].detach().clone());

}

// Save a set of team NNs
void Warehouse_ES_container::save_best_team_policy(std::vector<esNN*> teamNNs,int epoch,int G){
	std::string filename;
	auto net = std::make_shared<esNN>(1,1);
	for (size_t i = 0; i < teamNNs.size(); i++){
		filename = "best.epoch" + std::to_string(epoch) + "."+ std::to_string(i + 1) + "from" + std::to_string(teamNNs.size()) + ".G." + std::to_string(G) +".pt";

		for (size_t j = 0; j < teamNNs[i]->parameters().size(); j++ ){
			net->parameters()[j].set_data(teamNNs[i]->parameters()[j].detach().clone())  ;
			// std::cout<<torch::sum(net->parameters()[j])<<"\n";
		}
		
		torch::save(net,filename);	
	}
}


// Load a set of team NNs
void Warehouse_ES_container::load_best_team_policy(std::vector<esNN*> teamNNs){
	std::string filename = "best.epoch102.1from1.G.655.pt";



	auto net = std::make_shared<esNN>(1,1);
	for (size_t i = 0; i < teamNNs.size(); i++){		
		torch::load(net,filename);		
		for (size_t j = 0; j < teamNNs[i]->parameters().size(); j++ ){
			teamNNs[i]->parameters()[j].set_data(net->parameters()[j].detach().clone())  ;
			// std::cout<<torch::sum(teamNNs[i]->parameters()[j])<<"\n";
		}	
	}
}
