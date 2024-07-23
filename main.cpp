// Simulation code for the paper "A neural network model for the evolution of social learning"					
// by Jacob Chisausky, Inès Daras, Franz J. Weissing, and Magdalena Kozielska
// (c) Magdalena Kozielska and Jacob Chisausky 2024

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <array>
#include <string>
#include <random>
#include <numeric>
#include "agent.h"
#include "rndutils.hpp"
#include "parameters.h"
#include <algorithm>

std::mt19937_64 reng;		//Random number engine

auto env_change_dir_dist = std::bernoulli_distribution(0.5); //This determines whether a environmental change will be positive or negative in direction.

auto fitness_dist = rndutils::mutable_discrete_distribution<int, rndutils::all_zero_policy_uni>{};   //This sets the distribution of fitnesses for reproduction based on foraging success during each individual's lifetime.

Agent mutate(const Agent &ind, const parameters& p, int individual, bool diff_lr) { //When called this runs mutations on an individual. Called on each offspring when born.
	std::array<double, 33> new_weights = ind.get_weights();
	double new_lr = ind.get_learning_rate();
	double new_lrSL = ind.get_learning_rateSL();
	int new_le = ind.get_learning_episodes();
	double new_sl = ind.get_prop_SL();
	double new_drift = ind.get_drift_locus();
	int lifespan = ind.get_lifespan();
	int social_parent = -1;
	int new_schedule = ind.get_schedule();


	// distribtuions for mutation of network weights
	auto w_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_w);

	// distribtuions for mutation of learning rate
	auto lr_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_lr);

	// distribtuions for mutation of learning episode number
	auto le_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_le);

	// distribution for social parents
	std::uniform_int_distribution<> social_parent_dist(0, p.N - 1);

	// distribtuions for mutation of proportion of social learning
	auto sl_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_sl);

	// distribtuions for mutation of learning schedule
	auto schedule_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_schedule);

	if (p.mut_rate_w != 0.0 && p.mut_step_w !=0.0) { //If there are nonzero mut rates and mut step

		//Mutations to network
		for (int i = 0; i < 33; ++i) {
			if (w_mu_chance_dist(reng)) {	// Random engine determines if a mutation happens for each network weight/bias
				auto w_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_w); //Size of mutation if it happens
				new_weights[i] += w_mu_step_dist(reng);
			}
		}
	}

	//Mutations to number of learning episodes
	if (p.mut_rate_le != 0.0 && p.mut_step_le != 0.0) {
		if (le_mu_chance_dist(reng)) {
			int le_change = 0;
			auto le_mu_step_dist = std::uniform_int_distribution<int>(-p.mut_step_le, p.mut_step_le);
			do {
				le_change = le_mu_step_dist(reng);
			} while (le_change == 0);   // do not accept change of zero
			new_le += le_change;
			new_le = std::min(std::max(new_le, 0), lifespan); // constrain learning rate between 0 and lifespan
		}
	}

	//Mutations to schedule
	if (p.mut_rate_schedule != 0.0) {
		if (schedule_mu_chance_dist(reng)) {
			int sched_change = 0;
			std::uniform_int_distribution<> sched_mu_step_dist(-1, 1);		//Just give a schedule of -1, 0, or 1
			do {
				sched_change = sched_mu_step_dist(reng);
			} while (sched_change == new_schedule);  // schedule has to change if there is mutation
			new_schedule = sched_change;
		}
	}

	//Mutations to proportion of social learning
	if (p.mut_rate_sl != 0.0 && p.mut_step_sl != 0.0) {
		if (sl_mu_chance_dist(reng)) {
			if (p.mut_step_sl == -1){		//If mut step SL is -1, then SL mutations are binary. SL rate goes from 0 to 1 or from 1 to 0. This was not used in the paper.
				if (new_sl > 0.5){
					new_sl = 0;
				} else {
					new_sl = 1;
				}
			} else {						//If mut step SL is not -1, normal SL mutation based on normal distribution
				auto sl_mu_step_dist = std::normal_distribution<double>(0.0, std::abs(p.mut_step_sl));
				new_sl += sl_mu_step_dist(reng);
				new_sl = std::min(std::max(new_sl, 0.0), 1.0); // constrain SL rate between 0 and 1
			}
		}

		//This mutates a 'drift' gene which has no effect on behavior
		if (sl_mu_chance_dist(reng)) {
			if (p.mut_step_sl == -1){		//If mut step SL is -1, then drift mutations are binary. SL rate goes from 0 to 1 or from 1 to 0
				if (new_drift > 0.5){		//meaning that SL is 1 (this allows the binary mutations to work even if the simulation was not initialized with SL = 0 or 1
					new_drift = 0;
				} else {
					new_drift = 1;
				}
			} else {
				auto drift_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_sl);
				new_drift += drift_mu_step_dist(reng);
				new_drift = std::min(std::max(new_drift, 0.0), 1.0); // constrain drift locus between 0 and 1
			}
		}
	}

	//Mutations to learning rates
	if (p.mut_rate_lr != 0.0 && p.mut_step_lr != 0.0) {

		if (diff_lr){ //If different learning rates are used for IL and SL

			if (lr_mu_chance_dist(reng)) { //Learning rate for IL
				auto lr_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_lr);
				new_lr += lr_mu_step_dist(reng);
				new_lr = std::max(new_lr, 0.0); // constrain learning rate to be above 0
			}

			if (lr_mu_chance_dist(reng)) { //Learning rate for SL
				auto lr_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_lr);
				new_lrSL += lr_mu_step_dist(reng);
				new_lrSL = std::max(new_lrSL, 0.0); // constrain learning rate to be above 0
			}
		} else { //If the same learning rate is used for IL and SL (not used in final data)
			if (lr_mu_chance_dist(reng)) {
				auto lr_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_lr);
				new_lr += lr_mu_step_dist(reng);
				new_lr = std::max(new_lr, 0.0); // constrain learning rate to be above 0
				new_lrSL = new_lr;
			}
		}
	}

	return Agent(new_weights, new_lr, new_lrSL, new_le, lifespan, new_sl, new_drift, social_parent, new_schedule);
}


void prepare_output_files(const std::string &file_prefix, int rep) { //This just formats .csv files to store simulation data..

	const std::string fname_details = file_prefix + "_" + std::to_string(rep) + "_Details.csv";
	const std::string fname_weights = file_prefix + "_" + std::to_string(rep) + "_Weights.csv";
	const std::string fname_averages = file_prefix + "_" + std::to_string(rep) + "_Averages.csv";

	///Remove any old file with filename before adding to it
	std::remove(fname_details.c_str());
	std::remove(fname_weights.c_str());
	std::remove(fname_averages.c_str());

	// for details about individuals
	std::ofstream output_details;
	output_details.open(fname_details, std::ios_base::app);
	assert(output_details.is_open());
	output_details << "Replicate,Generation,Individual,Fitness,Prop_SL,Schedule,Drift_locus,Social_parent,Learning_rate,Learning_rateSL,Learning_episodes,Best_fitness,Env_mean_location,Mean_abs_weights,Mean_last_abs_weights,Bias_output\n";
	output_details.close();


	// for network weights and biases
	std::ofstream output_weights;
	output_weights.open(fname_weights, std::ios_base::app);
	assert(output_weights.is_open());
	output_weights << "Replicate,Generation,Individual,"
			<< "b2,b3,b4,b5,b6,b7,b8,b9,b10,"
			<< "w12,w13,w14,w15,w26,w27,w28,w29,w36,w37,w38,w39,w46,w47,w48,w49,w56,w57,w58,w59,w610,w710,w810,w910,new_w610,new_w710,new_w810,new_w910\n";

	output_weights.close();

	// for population averages
	std::ofstream output_averages;
	output_averages.open(fname_averages, std::ios_base::app);
	assert(output_averages.is_open());

	output_averages << "Replicate,Generation,mean fitness,mean prop SL,Schedule,mean learning rate,mean learning rate SL,mean learning episodes,mean best fitness,environmental mean location,sch_num_n1,sch_num_0,sch_num_1\n";

	output_averages.close();

	return;
}

void save_mean_data(const std::string &file_prefix, const std::vector<Agent>& population, int replicate, const std::vector<double>& fit, const std::vector<double>& propSL, const std::vector<double>& sched, const std::vector<double>& lr, const std::vector<double>& lrSL, const std::vector<double>& le, const std::vector<double>& chosen_e, const std::vector<double>& e_mean, const std::vector<int>& save_sch_n1, const std::vector<int>& save_sch_0, const std::vector<int>& save_sch_1) { //When called, this writes population averages to a .csv

	const std::string fname_averages = file_prefix + "_" + std::to_string(replicate) + "_Averages.csv";

	std::ofstream output_file;
	output_file.open(fname_averages, std::ios_base::app);

	int G = static_cast<int>(fit.size());

	for (int gen = 0; gen < G; ++gen) {
		output_file << std::setprecision(10) << replicate << "," << gen << "," << fit[gen] << ", " << propSL[gen] << "," << sched[gen] << "," << lr[gen] << "," << lrSL[gen] << "," << le[gen] << "," << chosen_e[gen] << "," << e_mean[gen] << "," << save_sch_n1[gen] << "," << save_sch_0[gen] << "," << save_sch_1[gen] << "\n";
	}
	output_file.close();

	return;
}

void save_details(const std::string &file_prefix, const std::vector<Agent> &population, int replicate, int generation, double env_mean_loc) { //When called, this writes individual-level variables (besides networks) to a .csv
	const std::string fname_details = file_prefix + "_" + std::to_string(replicate) + "_Details.csv";

	std::ofstream output_file;
	output_file.open(fname_details, std::ios_base::app);
	assert(output_file.is_open());

	double mean_w = 0.0;
	double mean_last_w = 0.0;
	std::array<double, 33> ind_weights;

	int N = static_cast<int>(population.size());

	for (int i = 0; i < N; ++i) {

		mean_w = 0.0;
		mean_last_w = 0.0;
		ind_weights = population[i].get_weights();

		// getting average value of all weights

		for (int w = 0; w < 33; ++w) {
			mean_w += ind_weights[w];
		}
		mean_w = mean_w / 33.0;

		//getting the average value of the final weights, that change in learning

		for (int w = 29; w < 33; ++w) {
			mean_last_w += ind_weights[w];
		}
		mean_last_w = mean_last_w / 4.0;

		output_file << replicate << "," << generation << ',' << i << "," << population[i].get_fitness() << "," << population[i].get_prop_SL() << "," << population[i].get_schedule() << ","  << population[i].get_drift_locus() << "," << population[i].get_social_parent() << "," << population[i].get_learning_rate()<< "," << population[i].get_learning_rateSL() << ',' << population[i].get_learning_episodes() << ','
				<< population[i].get_best_fitness() << ',' << env_mean_loc << ',' << mean_w << "," << mean_last_w << "," << ind_weights[8] << '\n';
	}
	output_file.close();

}

void save_weights(const std::string &file_prefix, const std::vector<Agent>& population, int replicate, int generation) { //When called, this writes all individual's network weights and biases to a .csv
	const std::string fname_details = file_prefix + "_" + std::to_string(replicate) + "_Weights.csv";

	std::ofstream output_file;
	output_file.open(fname_details, std::ios_base::app);
	assert(output_file.is_open());

	std::array<double, 33> ind_weights;

	int N = static_cast<int>(population.size());

	for (int i = 0; i < N; ++i) {

		output_file << replicate << "," << generation << ',' << i;

		ind_weights = population[i].get_weights();

		for(int w = 0; w < 33; ++w) {
			output_file << ',' << ind_weights[w];
		}

		// saves four last weights after learning
		ind_weights = population[i].get_new_weights();
		for (int w = 29; w < 33; ++w) {
			output_file << ',' << ind_weights[w];
		}

		output_file << '\n';
	}
	output_file.close();
}

inline double env_function(double a, double b, double input) {   //This determines the environment. It is a normal distribution function with peak 1, a = mean, b = standard deviation
	return std::exp(-0.5 * ((input - a) * (input - a)) / (b * b));      // something that resembles normal distribution but at the peak value is always 1. exp(quotient/(2*b*b))/(b*sqrt(2.0*PI));
}

inline double sign(double t) {   // a small function to return a sign of a double as double
	return t < 0.0 ? -1.0 : 1.0;
}


int main(int argc, char* argv[])
{
	// getting parameters from the parameter .json file

	std::cout << argv[1] << std::endl;

	nlohmann::json json_in;
	std::ifstream is(argv[1]);   //assumes that the file name is given as a parameter in the command line
	is >> json_in;
	parameters sim_pars = json_in.get<parameters>();

	int pop_size = sim_pars.N; 	//Population size
	int G = sim_pars.G;			//Number of generations to run simulation

	std::uniform_real_distribution<double> init_weights_dist(-std::abs(sim_pars.init_w_range), std::abs(sim_pars.init_w_range));	//Distribution to set initial network weights/biases
	std::uniform_real_distribution<double> init_lr_dist(0.0, std::abs(sim_pars.init_lr_range));										//Distribution to set initial learning rates
	std::poisson_distribution<int> init_le_dist(std::abs(sim_pars.init_le_mean));													//Distribution to set initial number of learning episodes
	std::uniform_real_distribution<> schedule_dist(0.0, 1.0); 																		//Distribution to set initial schedules
	std::uniform_real_distribution<double> init_sl_dist(0.0, 1.0);																	//Distribution to set initial proportion of social learning

	//The following lines take variables for social learning implementation from .json file and stores them as local variables
	bool local_one_social_parent = sim_pars.one_social_parent;	//If TRUE, one demonstrator is used per learner. If FALSE, learners use a new demonstrator each learning episode.
	bool local_success_bias = sim_pars.success_bias;			//If TREU, a success bias in demonstrator selection is used.
	int local_sl_speed = sim_pars.sl_speed;						//Set social learning speed
	bool local_diff_lr = sim_pars.differential_learning_rates;	//If TREU, learning rates for IL and SL are allowed to be different (ie they are controlled by different genes)
	int local_extrapolate_from = sim_pars.extrapolate_from;		//Rather than running a number of foraging episodes equal to lifespan (minus learning episodes) each generation, a smaller number of foraging episodes can be run and the results can be extrapolated to the expected foraging success after a full lifespan (minus foraging) of foraging
		
	std::string SL_type_local = "";
	if (sim_pars.SL_type == "ChooseSelfLearnParent"){
		SL_type_local = "ChooseSelfLearnParent";
	} else if (sim_pars.SL_type == "ChooseParentLearnParent"){
		SL_type_local = "ChooseParentLearnParent";
	} else if (sim_pars.SL_type == "ChooseParentLearnSelf"){
		SL_type_local = "ChooseParentLearnSelf";
	} else {
		std::cout << "Error: SL_type is not valid option. You entered: " << sim_pars.SL_type;
		std::cerr << "Error: SL_type is not valid option. You entered: " << sim_pars.SL_type;
		return 1;
	}

	int env_sample_size_local = sim_pars.env_sample_size; //How many environmental cues to look at when foraging

	std::string IL_type_local = "";
	if (sim_pars.IL_type == "Guided"){
		IL_type_local = "Guided";
	} else if (sim_pars.IL_type == "Unguided"){
		IL_type_local = "Unguided";
	} else {
		std::cout << "Error: IL_type is not valid option. You entered: " << sim_pars.IL_type;
		std::cerr << "Error: IL_type is not valid option. You entered: " << sim_pars.IL_type;
		return 1;
	}

	std::string cue_type_local = "";
	if (sim_pars.cue_type == "Random") { //Each individual gets different cues from others to learn and forage from.
		cue_type_local = "Random";
	}
	else if (sim_pars.cue_type == "Same") { //All individuals get same cues. This option was used in final data.
		cue_type_local = "Same";
	}
	else {
		std::cout << "Error: Invalid entry for cue_type. You entered: " << sim_pars.cue_type;
		std::cerr << "Error: Invalid entry for cue_type. You entered: " << sim_pars.cue_type;
		return 1;
	}
	

	std::uniform_int_distribution<int> rand_cue(0, (env_sample_size_local - 1)); //Picks random cues. When needed, a cue will be given as rand_cue(reng) * mod.

	//Validate some parameters
	assert(sim_pars.env_change <= sim_pars.env_range && sim_pars.env_change >= 0.0);
	assert(sim_pars.env_range > 0.0);
	assert(sim_pars.init_mean_env > -sim_pars.env_range && sim_pars.init_mean_env < sim_pars.env_range);

	//These lines format a string for the .csv output file names
	std::string file_name_prefix = argv[1];
	const std::string toRemove = "_Parameters.json";  // the suffix to be removed
	file_name_prefix.erase(file_name_prefix.find(toRemove), toRemove.length());

	std::vector<std::vector<double>> env_cues(sim_pars.lifespan*local_sl_speed, std::vector<double>(sim_pars.env_sample_size));  //Vector that will contain all the cues encounter by all individuals
	std::vector<std::vector<double>> env_quality(sim_pars.lifespan*local_sl_speed, std::vector<double>(sim_pars.env_sample_size)); //Vector that will contain environmental quality for the cues in env_cues

	std::uniform_real_distribution<double> cues_dist(-sim_pars.env_range, sim_pars.env_range);  // distribution from which to draw cues

	std::normal_distribution<double> env_change_dist(sim_pars.env_change, sim_pars.env_change_sd* sim_pars.env_range); // distribution from which environmental changes are drawn

	double actual_env_change = 0.0; //Initialzie variable which will hold environmental change size

	auto parent_dist = std::uniform_int_distribution<int>(0, pop_size-1);  // distribution to get a random demonstrator

	for (int rep = 0; rep < sim_pars.nr_replicates; ++rep) {  //Simulation start loop. This loop is for each replicate, so multiple replicates can be run sequentially.

		reng.seed(sim_pars.seed*(rep+1));	//Set seed for random number generator for each replicate. We can easily recover seed for any replicate.

		// Setting the distribution of fitness of all parental generation to 1 for the first generation => no difference in fitness
		std::vector<int> vec1(pop_size);
		std::fill(vec1.begin(), vec1.end(), 1);
		fitness_dist.mutate(vec1.cbegin(),vec1.cend());

		//Prepare output files for the replicate
		prepare_output_files(file_name_prefix, rep);

		double max_fitness = -1; //Will store max fitness acheived by any individual.

		std::vector<Agent> pop;  //Holds all individuals - population vector

		std::vector<double> fitnesses(pop_size); //Vector of fitnesses for all individuals

		double mean_env = sim_pars.init_mean_env; //Initial environemtnal mean 

		// vectors to save average population values each generation
		std::vector<double> mean_schedule(G + 1, 0.0);
		std::vector<double> mean_learning_rates(G + 1, 0.0);
		std::vector<double> mean_learning_ratesSL(G + 1, 0.0);
		std::vector<double> mean_learning_episodes(G + 1, 0.0);
		std::vector<double> mean_prop_SL(G + 1, 0.0);
		std::vector<double> mean_fitness(G + 1, 0.0);
		std::vector<double> environmental_mean(G + 1, 0.0);
		std::vector<double> mean_chosen_env(G + 1, 0.0);
		// vectors to save the number of individuals with each schedule
		std::vector<int> save_sch_n1(G + 1, -1);
		std::vector<int> save_sch_0(G + 1, -1);
		std::vector<int> save_sch_1(G + 1, -1);

		//Stores max number learning episodes for any individual in the population (used for exptrapolate_from shortcut)
		int maxLE = 0;
		if (sim_pars.le_homogenous == true){
			maxLE = sim_pars.init_le_mean;
		}

		//Initializing all individuals in population
		for (int i = 0; i < pop_size; ++i) {  //For each individual

			std::array<double, 33> weights;	//Array of network weights and biases for that individual
			double lr = sim_pars.init_lr_range; //Initialize learning rate for IL (these genes may be changed below)
			double lrSL = lr;					//Initialize learning rate for SL to be the same as learning rate for IL
			int le = sim_pars.init_le_mean;		//Initialize number of learning episodes (these genes may be changed below)
			double sl = sim_pars.init_prop_sl;	//Initialize proportion of SL (these genes may be changed below)
			double drift = sim_pars.init_prop_sl; //Initialize drift gene

			//Assign initial social learning proportion if sl_homogenous is false. This was not used as we always set sl_homogeneous to true.
			if (sim_pars.init_prop_sl > 0.0 && sim_pars.sl_homogenous == false){
				sl = init_sl_dist(reng); //Assign initial social learning proportion from a distribution
				drift = sl;
			} else if (sim_pars.init_prop_sl < 0.0){ // This was not used. If init_prop_sl is initialized NEGATIVE, set -init_prop_sl % of individuals to sl=1, the rest to sl=0
				if (i <= -1*sim_pars.init_prop_sl * pop_size){
					sl = 1;
					drift = 1;
				} else {
					sl = 0;
					drift = 0;
				}
			}

			//Assign initial network weights and biases
			if(sim_pars.init_w_range>0.0){
				for (int w = 0; w < 33; ++w) {  //for each weight
					weights[w] = init_weights_dist(reng);
				}
			}
			else {
				weights.fill(0.0);
			}

			//Assign initial learning rate for IL.
			if (sim_pars.init_lr_range > 0.0 && sim_pars.lr_homogenous == false) {
				lr = init_lr_dist(reng);
			}

			//If the learning rate for SL and IL are allowed to be different, then initialize set learning rate for SL independently. Otherwise, make it the same as learning rate for IL.
			if (local_diff_lr){
				if (sim_pars.init_lr_range > 0.0 && sim_pars.lr_homogenous == false) {
					lrSL = init_lr_dist(reng);
				}
			} else {
				lrSL = lr;
			}

			//Initialize initial number of learning episodes.
			if (sim_pars.init_le_mean > 0 && sim_pars.le_homogenous == false) {

				le = init_le_dist(reng);

				if (le > maxLE){
					maxLE = le;
				}
			}

			//Assign social parent (demonstrator) to each learner if only one social parent will be used
			int social_parent = -1;
			if (local_one_social_parent == true){
				std::uniform_int_distribution<> social_parent_dist(0, pop_size - 1);
				do {
					social_parent = social_parent_dist(reng);
				} while (i == social_parent);

			}

			pop.push_back(Agent(weights, lr, lrSL, le, sim_pars.lifespan, sl, drift, social_parent, sim_pars.schedule_init)); //Add the agent we just initialized to the population vector
		}

		//This is the parent vector, which stores the previous generation to socially learn from
		//in the first generation parents are the same as current population
		std::vector<Agent> parents = pop;

		//Run core of the simulation. Run this loop every generation
		for (int gen = 0; gen <= G; ++gen) {  // for each generation

			//Create an new environmental mean if needed (environmental change)
			if (sim_pars.env_change > 0.0 && (gen > 0) && (gen% sim_pars.env_change_rate == 0)) {

				//Determine size of shift
				actual_env_change = env_change_dist(reng);


				if (sim_pars.env_change_type == "random") {
					if (env_change_dir_dist(reng)) {  //direction of env change - 50% chance of increase or decrease.
						mean_env += actual_env_change;
					}
					else {
						mean_env -= actual_env_change;
					}
					if (mean_env > sim_pars.env_range) {				// if mean is higher than range it is moved to the lower side of the range (wrapped environmental profile)
						mean_env = -sim_pars.env_range + (mean_env - sim_pars.env_range);
					}
					else if (mean_env < -sim_pars.env_range) {				// if mean is lower than -range it is moved to the upper side of the range (wrapped environmental profile)
						mean_env = sim_pars.env_range + (mean_env + sim_pars.env_range);
					}

					if (std::abs(mean_env) < 0.00001) { mean_env = 0.0; }    // set to 0 if very small
				}
				else if (sim_pars.env_change_type == "cyclic") { //Not used in the paper
					mean_env += std::max(0.0,actual_env_change);  // env value always moves in the same direction

					if (mean_env > sim_pars.env_range) {				// if mean is higher than range it is moved to the lower side of the range (wrapped environmental profile)
						mean_env = -sim_pars.env_range + (mean_env - sim_pars.env_range);
					}
					if (std::abs(mean_env) < 0.00001) { mean_env = 0.0; }
				}
				else { return 1; }

			}

			//Creating cues matrix. These are cues individuals learn from and forage from
			for (long ls = 0; ls < (local_extrapolate_from + (maxLE * local_sl_speed) ); ++ls){ //Make enough cues for all learning and then for local_extrapolate_from number of foraging episodes
				for (int ess = 0; ess < sim_pars.env_sample_size; ++ess){
					env_cues[ls][ess] = cues_dist(reng); //Create a random cue, just a number from -1 to 1 on the environment. These are associated to enviornmental qualities below.
				}
			}

			// Creating corresponding matrix of environmental quality distribution. Each cue is associated to one quality.
			for (long ls = 0; ls < (local_extrapolate_from + (maxLE * local_sl_speed) ); ++ls){
				for (int ess = 0; ess < sim_pars.env_sample_size; ++ess)
				{
					if (env_cues[ls][ess] < mean_env - sim_pars.env_range) { // wrapping environmental profile
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, 2* sim_pars.env_range + env_cues[ls][ess]);
					}
					else if (env_cues[ls][ess] > mean_env + sim_pars.env_range) { // wrapping environmental profile
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, env_cues[ls][ess] - 2 * sim_pars.env_range);
					}
					else {
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, env_cues[ls][ess]);
					}
				}
			}


			int sch_n1=0;
			int sch_0=0;
			int sch_1=0;

			//Individuals learning
			for (int ind = 0; ind < pop_size; ++ind) {   //for each individual


				// learning phase
				int parent = -1;
				int chosen_env = -1;
				int quality_parent = -1;   // quality of a cue given by the parent
				int learning_episodes = pop[ind].get_learning_episodes();

				if (learning_episodes > 0 && (pop[ind].get_learning_rate() + pop[ind].get_learning_rateSL() ) > 0.0) {   // if there is learning at all

					double ind_sl = pop[ind].get_prop_SL();	//proportion social learning 

					if (ind_sl > 0.0) {		//If there is any social learning

						if (local_one_social_parent == true && local_success_bias == true){    //picking social parents if only one parent is used
							parent = fitness_dist(reng);	//Demonstrator (social parent) drawn from the success/fitness distribution
							pop[ind].set_social_parent(parent);		//Tracks social parent by adding it to agent variable
						} else if (local_one_social_parent == true && local_success_bias == false ){
							parent = parent_dist(reng);		//Demonstrator drawn randomly (no success bias)
							pop[ind].set_social_parent(parent);
						}

						int ind_sched = pop[ind].get_schedule();		//schedule implemented by the given individual

						int numSL = std::round(ind_sl*learning_episodes);  //Number of SL events the individual will do (used if shedule not equal 0)

						if (ind_sched == -1){ 	//This is the 'SL first' schedule. SL events first, then go to IL events

							for (int ls = 0; ls < ( numSL * local_sl_speed ); ls = ls + local_sl_speed) { // for all social learning episodes
								if(SL_type_local == "ChooseParentLearnParent"){ //aka Socially instructed learning
									if (local_one_social_parent == false){  //Choosing social parent (demonstrator) for each learning step. This overwrites the above selection.
										if (local_success_bias == true){
											parent = fitness_dist(reng);
										} else {
											parent = parent_dist(reng);
										}
									}
									for (int c = 0; c < local_sl_speed; c++){ //Multiple learning events with this demonstrator if sl_speed > 1
										chosen_env = parents[parent].choose_environment(env_cues[ls+c]); //demonstrator chooses an environment from offered cues
										quality_parent = parents[parent].network_calculation(env_cues[ls+c][chosen_env]); //demonstrator calculates the expected value of that cue
										pop[ind].learning_SL(env_cues[ls+c][chosen_env], quality_parent); //learner uses delta rule based on demonstrator chosen cue and demonstrator's expected quality
									}

								} else if(SL_type_local == "ChooseParentLearnSelf"){ //aka Socially guided learning
									if (local_one_social_parent == false){  //Choosing social parent for each learning step
										if (local_success_bias == true){
											parent = fitness_dist(reng);
										} else {
											parent = parent_dist(reng);
										}
									}
									for (int c = 0; c < local_sl_speed; c++){ //Multiple learning events with this demonstrator if sl_speed > 1
										chosen_env = parents[parent].choose_environment(env_cues[ls+c]); //Demonstrator picks cue to learn from
										pop[ind].learning_SL(env_cues[ls+c][chosen_env], env_quality[ls+c][chosen_env]); //Learner uses delta rule based on demonstrator chosen cue and real quality of cue
									}
								}
							}//After all sl events, do remaining events in IL
							for (int ls = numSL * local_sl_speed; ls < ( learning_episodes * local_sl_speed ); ls = ls + local_sl_speed){

								if (IL_type_local == "Unguided"){ //Unguided IL
									int cue_current = 0;
									if (cue_type_local == "Random"){
										cue_current = rand_cue(reng);
									}
									pop[ind].learning_IL(env_cues[ls][cue_current], env_quality[ls][cue_current]); //Learn using delta rule - randomly assigned cue and real quality of that cue

								} else if (IL_type_local == "Guided"){ //Self-guided IL
									chosen_env = pop[ind].choose_environment(env_cues[ls]); //Individual chooses cue with highest expected quality
									pop[ind].learning_IL(env_cues[ls][chosen_env], env_quality[ls][chosen_env]); //Learn using delta rule - chosen cue and real quality

								}
							}


						} else if (ind_sched == 1){		//'IL first' schedule. Do early IL events, then go to SL events. Learning implemented as above
							for (int ls = 0; ls < ( (learning_episodes-numSL) * local_sl_speed ); ls = ls + local_sl_speed) {

								if (IL_type_local == "Unguided"){
									int cue_current = 0;
									if (cue_type_local == "Random"){
										cue_current = rand_cue(reng);
									}
									pop[ind].learning_IL(env_cues[ls][cue_current], env_quality[ls][cue_current]);

								} else if (IL_type_local == "Guided"){

									chosen_env = pop[ind].choose_environment(env_cues[ls]);
									pop[ind].learning_IL(env_cues[ls][chosen_env], env_quality[ls][chosen_env]);

								}

							}//After all IL events, do remaining events in SL
							for (int ls = (learning_episodes-numSL) * local_sl_speed; ls < ( learning_episodes * local_sl_speed ); ls = ls + local_sl_speed){


								if(SL_type_local == "ChooseParentLearnParent"){
									if (local_one_social_parent == false){  //Choosing social parent for each learning step
										if (local_success_bias == true){
											parent = fitness_dist(reng);
										} else {
											parent = parent_dist(reng);
										}
									}
									for (int c = 0; c < local_sl_speed; c++){
										chosen_env = parents[parent].choose_environment(env_cues[ls+c]);
										quality_parent = parents[parent].network_calculation(env_cues[ls+c][chosen_env]);
										pop[ind].learning_SL(env_cues[ls+c][chosen_env], quality_parent);		
									}

								} else if(SL_type_local == "ChooseParentLearnSelf"){
									if (local_one_social_parent == false){  //Choosing social parent for each learning step
										if (local_success_bias == true){
											parent = fitness_dist(reng);
										} else {
											parent = parent_dist(reng);
										}
									}
									for (int c = 0; c < local_sl_speed; c++){
										chosen_env = parents[parent].choose_environment(env_cues[ls+c]);
										pop[ind].learning_SL(env_cues[ls+c][chosen_env], env_quality[ls+c][chosen_env]);
									}
								}
							}

						} else if (ind_sched == 0){		//'Shuffled' schedule. Do IL and SL events in random order. Learning implemented as above.

							auto sl_chance_dist = std::bernoulli_distribution(ind_sl);

							for (int ls = 0; ls < (learning_episodes*local_sl_speed); ls = ls + local_sl_speed) {

								if ( sl_chance_dist(reng) ) {  // if the time step uses social learning

									if (local_one_social_parent == false){  //Choosing social parent for each learning step
										if (local_success_bias == true){
											parent = fitness_dist(reng);
										} else {
											parent = parent_dist(reng);
										}
									}

									if(SL_type_local == "ChooseParentLearnParent"){

										for (int c = 0; c < local_sl_speed; c++){
											chosen_env = parents[parent].choose_environment(env_cues[ls+c]);
											quality_parent = parents[parent].network_calculation(env_cues[ls+c][chosen_env]);
											pop[ind].learning_SL(env_cues[ls+c][chosen_env], quality_parent);		
										}

									} else if(SL_type_local == "ChooseParentLearnSelf"){

										for (int c = 0; c < local_sl_speed; c++){
											chosen_env = parents[parent].choose_environment(env_cues[ls+c]);
											pop[ind].learning_SL(env_cues[ls+c][chosen_env], env_quality[ls+c][chosen_env]);
										}

									}
								} else {	//If there is an IL event

									if (IL_type_local == "Unguided"){
										int cue_current = 0;
										if (cue_type_local == "Random"){
											cue_current = rand_cue(reng);
										}
										pop[ind].learning_IL(env_cues[ls][cue_current], env_quality[ls][cue_current]);

									} else if (IL_type_local == "Guided"){

										chosen_env = pop[ind].choose_environment(env_cues[ls]);
										pop[ind].learning_IL(env_cues[ls][chosen_env], env_quality[ls][chosen_env]);

									}
								}
							} //end of learning
						} //end schedule
					} else {  //No chance of SL - ignore schedules and just do IL
						for (int ls = 0; ls < (learning_episodes*local_sl_speed); ls = ls + local_sl_speed) {
							if (IL_type_local == "Unguided"){
								int cue_current = 0;
								if (cue_type_local == "Random"){
									cue_current = rand_cue(reng);
								}
								pop[ind].learning_IL(env_cues[ls][cue_current], env_quality[ls][cue_current]);

							} else { //Self-guided learning

								chosen_env = pop[ind].choose_environment(env_cues[ls]);
								pop[ind].learning_IL(env_cues[ls][chosen_env], env_quality[ls][chosen_env]);
							}
						}
					}
				} //END THE LEARNING PHASE-------------------

				// Foraging
				pop[ind].exploit(env_cues, env_quality, maxLE, local_extrapolate_from);	//an individual goes through available cues and forages. They look at cues, choose the one with the highest expected quality, and gain fitness 

				//This stores number of individual with each schedule for reporting
				int sch = pop[ind].get_schedule();
				if (sch == -1){
					sch_n1 += 1;
				} else if (sch == 0){
					sch_0 += 1;
				} else {
					sch_1 += 1;
				}

				//These vectors are used to caluclate means below
				mean_schedule[gen] += pop[ind].get_schedule();
				mean_learning_rates[gen] += pop[ind].get_learning_rate();
				mean_learning_ratesSL[gen] += pop[ind].get_learning_rateSL();
				mean_learning_episodes[gen] += static_cast<double>(pop[ind].get_learning_episodes());
				mean_prop_SL[gen] += pop[ind].get_prop_SL();
				mean_chosen_env[gen] += pop[ind].get_best_fitness();
				fitnesses[ind] = pop[ind].get_fitness();

			}  // End learning and foraging for that individual. Repeat for all individuals.

			//Reporting schedules
			save_sch_n1[gen] = sch_n1;
			save_sch_0[gen] = sch_0;
			save_sch_1[gen] = sch_1;

			//Calculate means
			mean_schedule[gen] /= static_cast<double>(pop_size);
			mean_learning_rates[gen] /= static_cast<double>(pop_size);
			mean_learning_ratesSL[gen] /= static_cast<double>(pop_size);
			mean_learning_episodes[gen] /= static_cast<double>(pop_size);
			mean_prop_SL[gen] /= static_cast<double>(pop_size);
			mean_chosen_env[gen] /= static_cast<double>(pop_size);
			mean_fitness[gen] = std::accumulate(fitnesses.cbegin(), fitnesses.cend(), 0.0) / static_cast<double>(pop_size);
			environmental_mean[gen] = mean_env;

			// save some output once a while.
			if ((gen % (G/4) == 0) || (gen == G - 1)) {
				save_details(file_name_prefix, pop, rep, gen, mean_env);

				std::cout << "Replication " << rep << " Generation " << gen << " finished\n";
			}

			if ((gen == 0) || (gen >= (G-1))) {  // save weights in 1st and in last generation(s). Also save the penultamite generation to use for social cues in R learning script

				save_weights(file_name_prefix, pop, rep, gen);
			}

			// Reproduction

			//vector of agents to store offspring population
			std::vector<Agent> new_population;

			//Make distribution of population based on fitness for reproducing
			fitness_dist.mutate(fitnesses.cbegin(), fitnesses.cend());

			maxLE = 0; //Store highest number of learning episodes by any individual in population

			for (int i = 0; i < pop_size; ++i) { //Each iteration creates a new offspring individual
				int next_parent = fitness_dist(reng); //Pick an agent to reproduce based on fitness (parent)
				new_population.push_back(mutate(pop[next_parent], sim_pars, i, local_diff_lr)); // Calls mutate to take the chosen parent's gene values, probabilistically mutate them, and add to offspring vector

				//Find new maxLE
				if (new_population[i].get_learning_episodes() > maxLE){
					maxLE = new_population[i].get_learning_episodes();
				}
			}
			assert(static_cast<int>(new_population.size()) == pop_size);

			new_population.swap(pop);  // new_pop (offspring vector) becomes current population; pop becomes new_population
			parents.swap(new_population); //Previous generation stored as parents (demonstrators) for social learning in next generation

		}  // End calculations for that generation, move on to next generation

		// saving averages
		save_mean_data(file_name_prefix, pop, rep, mean_fitness, mean_prop_SL, mean_schedule, mean_learning_rates, mean_learning_ratesSL, mean_learning_episodes, mean_chosen_env, environmental_mean, save_sch_n1, save_sch_0, save_sch_1);

	}  // end replicate loop

	return 0;
}
