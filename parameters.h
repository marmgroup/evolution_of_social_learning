#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "json.hpp"

struct parameters
{
	int seed = 123456;		// Random number generator seed.
	double init_w_range = 1;  	// Initial weights, taken from uniform distribution from -init_w_range to + init_w_range
	double init_lr_range = 1;  	// Inital learning rate, taken from uniform distribution from  0 to init_rl_range (for both IL and SL)
	int init_le_mean = 20;    	// Initial number of learning episodes. This value used for all individuals or this value used as mean of poisson distribution to set values, depending on le_homogeneous
	double init_prop_sl = 0.5;	// Initial proporiton of social learning for all individuals.
	double mut_rate_w = 0.01;  	// Mutation rate for weights. Chance per generation of each weight mutating
	double mut_step_w = 0.1;  	// Mutation step for weights. Mutatations increase or decrease value by sampling from a normal distriubtion of this standard deviation centered on 0
	bool lr_homogenous = false; 	// If true all individuals initialized with same learning rate as lr = init_lr_range. If false, each individual's learning rate is taken from uniform distribution from  0 to init_rl_range
	double mut_rate_lr = 0.01;	// Mutation rate for both IL and SL learning rates
	double mut_step_lr = 0.1;  	// Mutation step for IL and SL learning rate (SD of normal distribution as for mut_step_w)
	bool le_homogenous = true;  	// Do all infividuals have the same number of learning episodes; if true all individuals has le = init_le_mean. Otherwise, sampled from a poisson distribution,
	double mut_rate_le = 0.01; 	// Mutation rate for the number of learning episodes
	int mut_step_le = 1;  		// Mutation step for the number of learning episodes, ie if 1 then mutations to learning episode number change it by +1 or -1
	double mut_rate_sl = 0.01; 	// Mutation rate for the proportion of social learning
	double mut_step_sl = 0.1;    	// Mutation step for the proportion of social learning (SD of normal distribution)
	int N = 1000;  			// Population size
	int G = 20000;  		// Number of generations per replicate
	int lifespan = 500; 		// Total number of learning and foraging episodes per individual
	int extrapolate_from = 30;  	// To reduce the simulation time; each individual conducts extrapolate_from foraging episodes and the total energy gain in the whole foraging period is extrapolated from that
	int nr_replicates = 10;  	// Number of replicates to run when this code is executed.
	int env_sample_size = 10;  	// How many environmental cues are assessed when learning or foraging
	double env_range = 1.0;  	// Range of cue values: between -env_range to + env_range. This delimits the size of the environment.
	double init_mean_env = 0.0;   	// Initial location of the environmental peak, should be within environmental range
	double sd_env = 0.25; 		// The width of the environmental profile. 0.1, 0.25, and 0.4 were tested.
	double env_change = 0.25;    	// The size of environmental changes, when they occur. 0.1, 0.25, and 0.4 were tested.
	double env_change_sd = 0.05;  	// Standard deviation (percentage of env range) of the environmental change distribution
	std::string env_change_type = "random";   	// Supports "random" (environemtnal peak shifts randomly either to the left or to the right) and "cyclic" (goes around the torus, not back and forth) 
	int env_change_rate = 10;    			// Environmental change occurs every env_change_rate generations
	std::string SL_type = "ChooseParnetLeanSelf";  	// Options: ChooseParentLearnParent (socially instructed learning), ChooseParentLearnSelf (socially guided learning), ChooseSelfLearnParent (not used)
	std::string IL_type = "Guided";			// Options: Unguided or Guided. Guided = individual picks the cue it thinks is best when learning (self-guided inidvidual learning). Unguided = individual picks a random cue (unguided individual learning)
	std::string cue_type = "Random";		// Options" Random or Same. If same, all individuals get same set of cues when learning and foraging. If random, each individual gets different cues.
	bool sl_homogenous = true;			// If true, all individuals are initialized with the proportion of social learning = init_prop_sl. If false, all individuals have a random sl rate from 0 to 1
	bool one_social_parent = true;  		// How many social parents do individuals have during the social learning period; if false then for every learning episode a different demonstrator is picked
	bool success_bias = true;  			// Is success bias used (true) or not (false)
	int sl_speed = 1;  				// Social learning speed
	double mut_rate_schedule = 0.01; 		// Mutation rate for the learning schedule
	bool differential_learning_rates = true;  	// Do individual and social learning have the same (true) or independent (false) learning rates? If false, then different genes control learning rate for IL and SL
	int schedule_init = 0;  			// Initial value of schedule: 0 - shuffled; -1 - social learning preceeds individual learning; 1 - individual learning preceeds social learning
};

void from_json(const nlohmann::json& j, parameters& p);


#endif // SIM_PARAMETERS_H
