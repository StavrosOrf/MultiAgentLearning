#include "Warehouse_ES_container.hpp"

//#define MULTITHREADED

Warehouse_ES_container::Warehouse_ES_container(YAML::Node configs,std::ofstream* eval_file){
	epoch = configs["ES"]["epochs"].as<int>();
	learning_rate = configs["ES"]["learning_rate"].as<float>();
	const size_t population_size = configs["ES"]["population_size"].as<int>();
	N_proc_std_dev = configs["DDPG"]["rand_proc_std_dev"].as<float>();
	assert(N_proc_std_dev > 0 && !std::isinf(N_proc_std_dev));
	assert(population_size > 0 && epoch > 0);
	file = eval_file;
	*file <<",Epoch,MAX_G,MAX_MOVE,MAX_ENTER,MAX_WAIT,AVG_G"<< std::endl;

	for(size_t i = 0 ; i < population_size; i++){
		population.push_back(new Warehouse_ES(configs));
		population[i]->InitialiseMATeam();
	}
}

uint Warehouse_ES_container::evolution_strategy(size_t n_threads, bool verbose,size_t run){
	//get initial random policy
	std::vector<esNN*> team = population[0]->produce_random_team_NNs(); //TODO RENAME team?
	std::vector<epoch_resultsES> results(population.size());
	epoch_resultsES max_results;
	// typedef std::chrono::high_resolution_clock clockTotal;
	auto startRun = std::chrono::high_resolution_clock::now();

	float avg_G;

	uint max_deliveries_intra = 0;
	for (int i = 0; i < epoch; i++){	
		// max_results.totalDeliveries = 0;
		avg_G = 0;
		auto startEpochh = std::chrono::high_resolution_clock::now();
#ifdef MULTITHREADED
		boost::asio::thread_pool simulator_pool(n_threads);
#endif
		for (size_t j = 0; j < population.size(); j++){
#ifdef MULTITHREADED
			boost::asio::post(simulator_pool, [i, j, team, &results, this](){
#endif
				population[j]->set_team_NNs(team);
				results[j] = population[j]->SimulateEpochES();	
#ifdef MULTITHREADED
			});
#endif
		}
#ifdef MULTITHREADED
		simulator_pool.join();
#endif
		max_results = results[0];
		//Calculate statistics for every epoch
		for (epoch_resultsES r : results){
			avg_G += r.totalDeliveries;			
			max_results.totalDeliveries = std::max<uint>(max_results.totalDeliveries, r.totalDeliveries);
			if(max_results.totalDeliveries < r.totalDeliveries){
				max_results = r;				
			}
		}
		avg_G = avg_G/population.size();
		max_deliveries_intra = std::max<uint>(max_results.totalDeliveries, max_deliveries_intra);


		/*Initialize Sum Tensor
		*sum's outer vector represents each Agent's NN
		* and the inner vector represents each NN's layers' parameters */

		std::vector<std::vector<torch::Tensor>> sum(team.size()) ;
		for (size_t i = 0; i < team.size(); i++){
			// std::vector<torch::Tensor>> temp(team[i][i]->NN->parameters().size());
			sum[i].resize(team[i]->parameters().size());
			for (size_t j = 0; j < team[i]->parameters().size(); j++ ){
				sum[i][j] = torch::zeros(team[i]->parameters()[j].sizes());				
			}
		}

		for (epoch_resultsES r : results){
			for (size_t i = 0; i < team.size(); i++){
				for (size_t j = 0; j < team[i]->parameters().size(); j++ ){
					sum[i][j] += (float)r.totalDeliveries * r.samples[i][j];					
				}
			}
		}

		for (size_t i = 0; i < team.size(); i++){
			for (size_t j = 0; j < team[i]->parameters().size(); j++ ){
				float delta = 1/ (((float)population.size()) * (float)N_proc_std_dev);
				sum[i][j] = learning_rate * sum[i][j] * delta ;
				// std::cout<<torch::sum(sum[i][j])<<"\n";
				team[i]->parameters()[j].set_data(team[i]->parameters()[j].detach().clone() + sum[i][j])  ;
			}
		}
		auto finishEpochh = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finishEpochh - startEpochh;
	
		//if(verbose)
		std::printf("- Epoch: %3d - ( %6.2f sec) =================== Avg G: %6.2f ----- max G: %d \n",i,elapsed.count(),avg_G,max_results.totalDeliveries);
			
						
		//Write results to file
		*file << ","<<i<<","<<max_results.totalDeliveries<<","<<max_results.totalMove <<","<<max_results.totalEnter <<","<<max_results.totalWait <<","<<avg_G <<std::endl;
	}

	auto endRun = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedT = endRun - startRun;
	std::cout<<"Total time elapsed for Run "<<run<<" ( "<<elapsedT.count()<<" sec) ----- MAX G: "<<max_deliveries_intra<<std::endl; 

	return max_deliveries_intra;
}
