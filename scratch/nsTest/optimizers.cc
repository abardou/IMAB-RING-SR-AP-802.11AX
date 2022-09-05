#include "optimizers.hh"
#include <iostream>

/**
 * Build an Optimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 */
Optimizer::Optimizer(Sampler* sampler, unsigned int testPeriod): _sampler(sampler), _generator(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count())), _distribution(0.0, 1.0), _testPeriod(testPeriod) {  };

/**
 * Virtual destructor for Optimizer
 */
Optimizer::~Optimizer() {  }

/**
 * Add a configuration and its associated reward to the history.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated
 * @param forward bool whether to forward to sampler (optional, default: true)
 */
void Optimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	this->_history.push_back(std::make_tuple(configuration, reward));
	if (forward)
		this->_sampler->addToBase(configuration, reward);
}

bool Optimizer::readyForAnother() const { return true; }

unsigned int Optimizer::getTestPeriod() const { return this->_testPeriod; }

/**
 * Show the average reward and the number of times each configuration is played
 */
void Optimizer::showDecisions() const {
	std::map<NetworkConfiguration, double> rewards;
	std::map<NetworkConfiguration, double> counters;
	std::vector<NetworkConfiguration> keys;
	// Compute statistics
	for (std::tuple<NetworkConfiguration, double> t: this->_history) {
		NetworkConfiguration conf = std::get<0>(t);
		double r = std::get<1>(t);

		if (rewards.find(conf) != rewards.end()) {
			rewards[conf] = (counters[conf] * rewards[conf] + r) / (counters[conf] + 1);
			counters[conf]++;
		} else {
			rewards[conf] = r;
			counters[conf] = 1;
			keys.push_back(conf);
		}
	}

	// Log the stats
	std::cout << "Rewards: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << (round(1000.0*rewards[conf])/1000.0) << ",";
	}
	std::cout << " ]" << std::endl << "Counter: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << counters[conf] << ",";
	}
	std::cout << " ]" << std::endl;
}

RandomNeighborOptimizer::RandomNeighborOptimizer(Sampler* sampler) : Optimizer(sampler) {  }

void RandomNeighborOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, false);
	this->_chosen = configuration;
}

NetworkConfiguration RandomNeighborOptimizer::optimize() {
	if (this->_firstCollection) {
		if (this->_counter == 0) {
			this->_chosen = (*this->_sampler)();
		}
	} else {
		if (this->_counter == 0) {
			int idx = std::uniform_int_distribution<int>(0, this->_chosen.size() - 1)(this->_generator);
			const unsigned int idx2 = std::uniform_int_distribution<int>(0, 1)(this->_generator);
			int sign = std::uniform_int_distribution<int>(0, 1)(this->_generator) == 0 ? -1 : 1;
			if (idx2 == 0) {
				std::get<0>(this->_chosen[idx]) += sign;
			} else {
				std::get<1>(this->_chosen[idx]) += sign;
			}
		}
	}

	this->_counter += 1;
	if (this->_counter == this->_n) {
		this->_secondCollection = !this->_secondCollection;
		this->_firstCollection = !this->_firstCollection;
		this->_counter = 0;
	}

	return this->_chosen;
}

/**
 * Do nothing optimizer. Useful to study reward drifts on the simulator.
 */
IdleOptimizer::IdleOptimizer() : Optimizer(nullptr) {  }

/**
 * Add the configuration to the history but does not forward to an unused
 * sampler.
 * 
 * @param configuration the configuration tested
 * @param reward the reward obtained
 * @param forward whether to forward the data to the sampler
 */
void IdleOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, false);
	this->_chosen = configuration;
}

/**
 * As an idle optimizer, do nothing except chosing the previously chosen
 * configuration.
 * 
 * @return the previously chosen configuration
 */
NetworkConfiguration IdleOptimizer::optimize() {
	return this->_chosen;
}

/**
 * Build an EpsilonGreedyOptimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 * @param epsilon double the exploration parameter
 */
EpsilonGreedyOptimizer::EpsilonGreedyOptimizer(Sampler* sampler, double epsilon): Optimizer(sampler), _epsilon(epsilon) {  };


/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void EpsilonGreedyOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, forward);

	if (this->_results.find(configuration) != this->_results.end())
		this->_results[configuration] = 0.8 * this->_results[configuration] + 0.2 * reward;
	else
		this->_results[configuration] = reward;
}

/**
 * Find the optimal configuration according to e-greedy strategy.
 * 
 * @return the optimal configuration according to e-greedy strategy
 */
NetworkConfiguration EpsilonGreedyOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_distribution(this->_generator) < this->_epsilon;
	if (explore) {
		NetworkConfiguration sampled = (*this->_sampler)();
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Exploitation
	double max = 0;
	NetworkConfiguration confMax;
	for (std::map<NetworkConfiguration, double>::iterator it = this->_results.begin(); it != this->_results.end(); ++it) {
		double challenger = it->second;
		if (max < challenger) {
			max = challenger;
			confMax = it->first;
		}
	}

	return confMax;
}

/**
 * Build a ThompsonGammaNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param sampleSize unsigned int the sample size to use for update
 * @param add double the exploration parameter
 */
ThompsonGammaNormalOptimizer::ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig, bool chain): Optimizer(sampler, sampleSize), _testLeft(sampleSize), _epsilon(eps), _to_ig(to_ig), _chain(chain) {  }

bool ThompsonGammaNormalOptimizer::readyForAnother() const {
	return this->_testLeft == 0;
}


/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonGammaNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	if (this->_testLeft == 0) {
		this->_testLeft = this->_testPeriod;
		this->_chosen = configuration;
	}

	this->_testLeft--;
	bool ignore = this->_testLeft >= this->_testPeriod - this->_to_ig;
	if (!ignore)
		Optimizer::addToBase(configuration, reward, forward);

	if (this->_chosen.size() == 0)
		this->_chosen = configuration;

	// Search in attribute for a preexisting sample
	if (!ignore && this->_gammaNormals.find(configuration) != this->_gammaNormals.end()) {
		GammaNormalSample& lns = this->_gammaNormals[configuration]; 
		std::vector<double>& sample = std::get<4>(lns);
		sample.push_back(reward);
		// Update the gamma normal if we reach sample size
		unsigned int n = this->_testPeriod - this->_to_ig;
		if (sample.size() == n) {
			double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / n;
			double var = n * (std::inner_product(sample.begin(), sample.end(), sample.begin(), 0.0) / n - mean * mean) / (n - 1);
			if (var <= 0) {
				var = 1e-9;
			}
			double &mu = std::get<0>(lns),
						 &lambda = std::get<1>(lns),
						 &alpha = std::get<2>(lns),
						 &beta = std::get<3>(lns),
						 newMu, newLambda, newAlpha, newBeta;
			bool& explore = std::get<5>(lns);
			if (explore) {
				newMu = mean;
				newLambda = n;
				newAlpha = n / 2.0;
				newBeta = n * var / 2.0;
				explore = false;
			} else {
				newMu = (lambda * mu + n * mean) / (lambda + n);
				newLambda = lambda + n;
				newAlpha = alpha + n / 2.0;
				newBeta = beta + (n * var + lambda * n * pow(mean - mu, 2) / (lambda + n)) / 2.0;
			}

			mu = newMu;
			lambda = newLambda;
			alpha = newAlpha;
			beta = newBeta;
			sample.clear();
		}
 	} else {
		 // Create a new instance in logNormals
		this->_gammaNormals[configuration] = std::make_tuple(0.5, 1.0, 0.5, 0.025, std::vector<double>({reward}), true);
	}
}

/**
 * Find the best configuration according to ThompsonGammaNormal strategy
 * 
 * @return the best configuration according to ThompsonGammaNormal strategy 
 */
NetworkConfiguration ThompsonGammaNormalOptimizer::optimize() {
	// this->showDecisions();

	// unsigned _seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::default_random_engine generator(_seed);
  // std::uniform_real_distribution<double> distribution(0.0, 1.0);

	if (this->_chain && this->_testLeft > 0) {
		return this->_chosen;
	}

	// Explore or exploit
	NetworkConfiguration newConf;
	bool confChosen = false;
	bool explore = this->_distribution(this->_generator) < (this->_chain ? 1 - pow(1 - this->_epsilon, this->_testPeriod) : this->_epsilon);
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0) {
			confChosen = true;
			newConf = sampled;
		}

		// std::cout << "Configuration: ";
		// for (std::tuple<double, double> t: newConf) {
		// 	std::cout << '(' << std::get<0>(t) << "," << std::get<1>(t) << "),"; 
		// }
		// std::cout << std::endl;
	}

	// Look for not enough explored configurations
	if (!confChosen) {
		std::vector<NetworkConfiguration> toExplore;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
			if (std::get<5>(it->second))
				toExplore.push_back(it->first);
		if (!toExplore.empty()) {
			// Test a not enough explored configuration
			std::uniform_int_distribution<> d(0, toExplore.size()-1);
			newConf = toExplore[d(this->_generator)];
		} else {
			// Sample taus for normal distributions
			std::vector<double> mus(this->_gammaNormals.size());
			int i = 0;
			for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it) {
				std::gamma_distribution<> gamma(std::get<2>(it->second), 1.0 / std::get<3>(it->second));
				double tau = gamma(this->_generator);
				std::normal_distribution<> normal(std::get<0>(it->second), 1.0 / sqrt(std::get<1>(it->second) * tau));
				mus[i] = normal(this->_generator);
				i++;
			}
			int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
			std::map<NetworkConfiguration, GammaNormalSample>::iterator maxConf = this->_gammaNormals.begin();
			for (int i = 0; i < maxElementIndex; i++) maxConf++;

			newConf = maxConf->first;
		}
	}

	this->_chosen = newConf;
	this->_testLeft = this->_testPeriod;

	// for (std::tuple<double, double> t: this->_chosen) {
	// 	std::cout << '(' << std::get<0>(t) << "," << std::get<1>(t) << "),"; 
	// }
	// std::cout << std::endl;

	return this->_chosen;
}

/**
 * Build a ThompsonNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param add double the exploration parameter
 */
ThompsonNormalOptimizer::ThompsonNormalOptimizer(Sampler* sampler, double eps): Optimizer(sampler), _epsilon(eps) {  }

/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward, bool forward, std::vector<std::tuple<double, unsigned int>> individual_rewards) {
	Optimizer::addToBase(configuration, reward, forward);

	// Search in attribute for a preexisting sample
	if (this->_normals.find(configuration) != this->_normals.end()) {
		NormalParameters& nps = this->_normals[configuration];
		double &mean = std::get<0>(nps),
					 &var = std::get<1>(nps);
		unsigned int &n = std::get<2>(nps);
		// Update the normal
		mean = (n * mean + reward) / (n + 1);
		var = 1.0 / (n + 1);
		n = n + 1;
 	} else {
		 // Create a new instance in normals
		this->_normals[configuration] = std::make_tuple(0, 1, 1);
	}
}

/**
 * Find the best configuration according to ThompsonNormal strategy
 * 
 * @return the best configuration according to ThompsonNormal strategy 
 */
NetworkConfiguration ThompsonNormalOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_distribution(this->_generator) < this->_epsilon;
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Sample mus for normal distributions
	std::vector<double> mus(this->_normals.size());
	int i = 0;
	for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it) {
		std::normal_distribution<> normal(std::get<0>(it->second), sqrt(std::get<1>(it->second)));
		mus[i] = normal(this->_generator);
		i++;
	}
	int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
	std::map<NetworkConfiguration, NormalParameters>::iterator maxConf = this->_normals.begin();
	for (int i = 0; i < maxElementIndex; i++) maxConf++;

	return maxConf->first;
}