#include <algorithm>
#include "agent.h"
#include <random>
#include <cassert>

extern std::mt19937_64 reng; //Random engine

Agent::Agent(const std::array<double, 33> &weights, double learningRate, double learningRateSL, int learningEpisodes, int lifespan, double prop_SL, double drift_locus, int social_parent, int schedule) :
							m_weights(weights), m_w(weights), m_learning_rate(learningRate), m_learning_rateSL(learningRateSL), m_learning_episodes(learningEpisodes), m_lifespan(lifespan), m_prop_sl(prop_SL), m_drift_locus(drift_locus),m_social_parent(social_parent), m_schedule(schedule)
{	
}

std::array<double, 33> Agent::get_weights() const {
	return m_weights;
}

std::array<double, 33> Agent::get_new_weights() const {
	return m_w;
}

double Agent::get_learning_rate() const {
	return m_learning_rate;
}

double Agent::get_learning_rateSL() const {
	return m_learning_rateSL;
}

int Agent::get_learning_episodes() const {
	return m_learning_episodes;
}

double Agent::get_fitness() const {
	return m_fitness;
}

double Agent::get_best_fitness() const {
	return m_best_fitness;
}

double Agent::get_lifespan() const {
	return m_lifespan;
}

double Agent::get_prop_SL() const {
	return m_prop_sl;
}

double Agent::get_drift_locus() const {
	return m_drift_locus;
}

int Agent::get_social_parent() const {
	return m_social_parent;
}

void Agent::set_social_parent(int sp) {
	m_social_parent = sp;
	return;
}

int Agent::get_schedule() const {
	return m_schedule;
}

//This is the neural network. Agent's network takes an input and runs it through the network to produce output. This determines expected quality of an environmental cue.
double Agent::network_calculation(double input) {
	// Description of the network:
	// 1 input - 2 hiden layers each wtih 4 nodes - 1 output
	// biases and weights
	//	b2		b3		b4		b5		b6		b7		b8		b9		b10
	//  w[0]	w[1]	w[2]	w[3]	w[4]	w[5]	w[6]	w[7]	w[8]
	//	w12		w13		w14		w15		w26		w27		w28		w29		w36		w37		w38		w39		w46		w47		w48		w49		w56		w57		w58		w59		w610	w710	w810	w910
	//	w[9]	w[10]	w[11]	w[12]	w[13]	w[14]	w[15]	w[16]	w[17]	w[18]	w[19]	w[20]	w[21]	w[22]	w[23]	w[24]	w[25]	w[26]	w[27]	w[28]	w[29]	w[30]	w[31]	w[32]

	double N2 = 0.0, N3 = 0.0, N4 = 0.0, N5 = 0.0, N6 = 0.0, N7 = 0.0, N8 = 0.0, N9 = 0.0;
	double output = 0.0;


	N2 = clamped_reLU(m_w[9] * input + m_w[0]);
	N3 = clamped_reLU(m_w[10] * input + m_w[1]);
	N4 = clamped_reLU(m_w[11] * input + m_w[2]);
	N5 = clamped_reLU(m_w[12] * input + m_w[3]);

	N6 = clamped_reLU(m_w[13] * N2 + m_w[17] * N3 + m_w[21] * N4 + m_w[25] * N5 + m_w[4]);
	N7 = clamped_reLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5]);
	N8 = clamped_reLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6]);
	N9 = clamped_reLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7]);

	output = m_w[29] * N6 + m_w[30] * N7 + m_w[31] * N8 + m_w[32] * N9 + m_w[8];

	return output;

}

//Individual learning. Calculate expected output and use delta rule to update network.
void Agent::learning_IL(double input, double expect) {    // one round of calculation + updating weights

	double N2 = 0.0, N3 = 0.0, N4 = 0.0, N5 = 0.0, N6 = 0.0, N7 = 0.0, N8 = 0.0, N9 = 0.0;
	double output = 0.0;
	double error; 

	N2 = clamped_reLU(m_w[9] * input + m_w[0]);
	N3 = clamped_reLU(m_w[10] * input + m_w[1]);
	N4 = clamped_reLU(m_w[11] * input + m_w[2]);
	N5 = clamped_reLU(m_w[12] * input + m_w[3]);

	N6 = clamped_reLU(m_w[13] * N2 + m_w[17] * N3 + m_w[21] * N4 + m_w[25] * N5 + m_w[4]);
	N7 = clamped_reLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5]);
	N8 = clamped_reLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6]);
	N9 = clamped_reLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7]);

	output = m_w[29] * N6 + m_w[30] * N7 + m_w[31] * N8 + m_w[32] * N9 + m_w[8];

	error = expect - output;


	m_w[29] = m_w[29] + m_learning_rate * error * N6;
	m_w[30] = m_w[30] + m_learning_rate * error * N7;
	m_w[31] = m_w[31] + m_learning_rate * error * N8;
	m_w[32] = m_w[32] + m_learning_rate * error * N9;

	return;

}

//Social learning. Calculate expected output (from own network or demonstrator's network) and use delta rule to update network.
void Agent::learning_SL(double input, double expect) {    // one round of calculation + updating weights

	double N2 = 0.0, N3 = 0.0, N4 = 0.0, N5 = 0.0, N6 = 0.0, N7 = 0.0, N8 = 0.0, N9 = 0.0;
	double output = 0.0;
	double error;

	N2 = clamped_reLU(m_w[9] * input + m_w[0]);
	N3 = clamped_reLU(m_w[10] * input + m_w[1]);
	N4 = clamped_reLU(m_w[11] * input + m_w[2]);
	N5 = clamped_reLU(m_w[12] * input + m_w[3]);

	N6 = clamped_reLU(m_w[13] * N2 + m_w[17] * N3 + m_w[21] * N4 + m_w[25] * N5 + m_w[4]);
	N7 = clamped_reLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5]);
	N8 = clamped_reLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6]);
	N9 = clamped_reLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7]);

	output = m_w[29] * N6 + m_w[30] * N7 + m_w[31] * N8 + m_w[32] * N9 + m_w[8];

	error = expect - output;

	m_w[29] = m_w[29] + m_learning_rateSL * error * N6;
	m_w[30] = m_w[30] + m_learning_rateSL * error * N7;
	m_w[31] = m_w[31] + m_learning_rateSL * error * N8;
	m_w[32] = m_w[32] + m_learning_rateSL * error * N9;

	return;
}

// returns the cue that the individual estimates to be the best among the available cues (input)
int Agent::choose_environment(const std::vector<double>& input) {
	int sample_size = static_cast<int> (input.size());  // getting the size of possible options each time step
	std::vector<double> assessment(sample_size);
	int best_env = -1.0;   //which cue leads to highest estimation

	for (int ss = 0; ss < sample_size; ++ss) {
		assessment[ss] = this->network_calculation(input[ss]);
	}
	best_env = static_cast<int>(std::distance(assessment.cbegin(), std::max_element(assessment.cbegin(), assessment.cend())));
	return best_env;
}


//Foraging. Take cues, use neural network to pick cues with highest expected quality, and add to fitness based on the real quality of that cue.
void Agent::exploit(const std::vector<std::vector<double>>& cues, const std::vector<std::vector<double>>& quality, const int maxLE_var, const int extrapolate_from_var) {

	assert(cues.size() == quality.size());

	int sample_size = static_cast<int> (cues[0].size());  // getting the size of possible options each time step
	std::vector<double> assessment(sample_size);
	int best_env = -1.0;   //which cue leads to the highest estimate

	assert(m_learning_episodes >= 0);
	assert(m_learning_rate >= 0.0);

	if (m_learning_episodes < m_lifespan) {

		for (int t = maxLE_var; t < (extrapolate_from_var + maxLE_var); ++t) {
			for (int ss = 0; ss < sample_size; ++ss) {
				assessment[ss] = this->network_calculation(cues[t][ss]);
			}
			best_env = static_cast<int>(std::distance(assessment.cbegin(), std::max_element(assessment.cbegin(), assessment.cend())));
			m_fitness += quality[t][best_env];
			m_best_fitness += *std::max_element(quality[t].begin(), quality[t].end());   //what is actually the best environment available
		}
		//Extrapolate fitnesses - Every individual gets extrapolate_from cues to exploit and then the fitness from those cues is
		//corrected for the number of learning episodes they had. 
		// Or said differently based on extrapolat_from cues average fitness per time step is calculated and then multiplied by the number of foraging time steps to calculate lifetime fitness
		// This was implemented to speed up the simulations, but with extrapolate_from set to high enough values does not affect the outcome
		//This would slow things down if m_lifespan - m_learning_episodes < extrapolate_from... but that will depend on the settings
		m_fitness = m_fitness * (m_lifespan - m_learning_episodes) / extrapolate_from_var;
		m_best_fitness = m_best_fitness * (m_lifespan - m_learning_episodes) / extrapolate_from_var;
	}
	return;
}

//Truncate input to between 0 and 1
double clamped_reLU(double input) {
	return std::min(std::max(0.0, input), 1.0);
}

