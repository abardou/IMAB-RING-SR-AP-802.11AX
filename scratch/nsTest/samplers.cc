#include "samplers.hh"
#include <iostream>
#include "unistd.h"

/**
 * Constraint on the parameters
 * 
 * @param txPower double the transmission power
 * @param obssPd double the sensibility
 * 
 * @return `true` if the constraint is satisfied, `false` otherwise
 */
bool constraint(double txPower, double obssPd) {
	return obssPd <= std::max(-82.0, std::min(-62.0, -82.0 + (20.0 - txPower)));
}

/**
 * Apply the parameter constraint on a whole network configuration
 * 
 * @param conf Container& containing the network configuration
 * 
 * @return `true` if the configuration respects the constraint, `false`
 * otherwise
 */
bool confConstraint(const NetworkConfiguration& conf) {
	for (std::tuple<double, double> t: conf) {
		if (!constraint(std::get<1>(t), std::get<0>(t))) {
			return false;
		}
	}

	return true;
}

/**
 * Initialize a sampler object
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */
Sampler::Sampler(const std::vector<ConstrainedCouple>& parameters) : _parameters(parameters) {  }

/**
 * Virtual destructor for Sampler
 */
Sampler::~Sampler() {  }

/**
 * Build a random action from the specified parameters
 * 
 * @param forbidden Container the actions that are forbidden to be returned
 * 
 * @return a random action not in `forbidden` 
 */
NetworkConfiguration Sampler::randomAction(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) const {
	NetworkConfiguration conf;
	unsigned int i = 0;
	do {
		conf.clear();
		if (i > 10000) {
			return {std::tuple<double, double>(0, 0)};
		}

		for (ConstrainedCouple c: this->_parameters)
			conf.push_back(c.randomValue());
		i++;
	} while (std::find(forbidden.begin(), forbidden.end(), conf) != forbidden.end());

	return conf;
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1).
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void Sampler::addToBase(NetworkConfiguration features, double reward) {
	std::vector<NetworkConfiguration>::iterator it = std::find(this->_features.begin(), this->_features.end(), features);
	if (it != this->_features.end()) {
		int index = it - this->_features.begin();
		this->_rewards[index] = 0.5 * this->_rewards[index] + 0.5 * reward;
	} else {
		this->_features.push_back(features);
		this->_rewards.push_back(reward);
	}
}

/**
 * Initialize a random sampler
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */ 
RandomSampler::RandomSampler(const std::vector<ConstrainedCouple>& parameters) : Sampler(parameters), _generator(std::chrono::system_clock::now().time_since_epoch().count()) {  }

/**
 * Initialize a uniform sampler
 * 
 * @param parameters Container the parameters, defining the space to sample on
 */
UniformSampler::UniformSampler(const std::vector<ConstrainedCouple>& parameters) : RandomSampler(parameters) {  }

/**
 * Return a random action based on collected rewards. Same as randomAction().
 * 
 * @param forbidden Container the network configuration we don't want.
 * 
 * @return a network configuration not in forbidden
 */
inline NetworkConfiguration UniformSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) { return this->randomAction(forbidden); }

/**
 * Operator () overloaded to be used
 * 
 * @param forbidden Container& containing the already explored configurations
 * 
 * @return a configuration not in `forbidden`
 */
inline NetworkConfiguration UniformSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) { return this->randomAction(forbidden); }

/**
 * Build an HGM sampler.
 * 
 * @param parameters Container the parameters that define the sampling space
 * @param def NetworkConfiguration the default configuration of the network
 * @param nMax unsigned int the max number of gaussians in the mixture
 * @param eps double the exploration parameter
 * @param dist double the distance of the new target
 * @param nTests FunctionPointer a way to quantify how many tests must be done 
 */
HGMTSampler::HGMTSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> bases, unsigned int nMax, double eps, double dist, unsigned int (*nTests)(std::vector<GaussianT>)) : RandomSampler(parameters), _nTests(nTests), _eps(eps), _dist(dist), _nMax(nMax), _testCounter(1) {
	for (NetworkConfiguration n: bases)
		this->addGaussian(n, 1.0, -1);
}

/**
 * Add a gaussian to the gaussian mixture. If the mixture is already full,
 * delete the oldest gaussian.
 * 
 * @param center NetworkConfiguration the center of the gaussian
 * @param dist double the distance to put the target at
 * @param reward double the reward max attained by the gaussian
 */ 
void HGMTSampler::addGaussian(const NetworkConfiguration& center, double dist, double reward) {
	// Turn the NetworkConfiguration into a list of normalized double
	std::vector<double> c(2 * center.size());
	unsigned int i = 0;
	for (std::tuple<double, double> t: center) {
		// Normalize the tuple
		std::tuple<double, double> normalized = this->_parameters[(int) (i / 2)].normalize(t);
		// Add it to the list
		c[i] = std::get<0>(normalized);
		c[i + 1] = std::get<1>(normalized);
		i += 2;
	}

	// Add the gaussian to the mixture
	if (this->_gaussians.size() == this->_nMax)
		this->_gaussians.erase(this->_gaussians.begin());

	this->_gaussians.push_back(std::make_tuple(c, dist / (21.0 * c.size()), reward));
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1). Update also the
 * average reward of the gaussian used.
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void HGMTSampler::addToBase(NetworkConfiguration features, double reward) {
	Sampler::addToBase(features, reward);

	// Update a center ?
	for (GaussianT gt: this->_gaussians) {
		NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(gt));
		if (features == ct) {
			double& maxRew = std::get<2>(gt);
			if (maxRew == -1) maxRew = reward;
			else maxRew = 0.9 * maxRew + 0.1 * reward;
		}
	}

	// Enough tests to change the mixture
	if (this->_testCounter >= this->_nTests(this->_gaussians)) {
		// Find the n greatest rewards
		std::vector<double> vec_to_sort = this->_rewards;
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), std::greater<double>());
		int n = std::min(this->_nMax, (unsigned int) vec_to_sort.size());
		std::vector<double> dest(n);
		for (int i = 0; i < n; i++) {
			dest[i] = vec_to_sort[i];
		}

		// Add new gaussians best on the n best
		double target = *std::max_element(dest.begin(), dest.end()) + this->_dist * this->_eps;
		std::vector<GaussianT> gaussians_copy = this->_gaussians;
		this->_gaussians.clear();
		for (double d: dest) {
			if (d != -1) {
				NetworkConfiguration c = this->_features[std::find(this->_rewards.begin(), this->_rewards.end(), d) - this->_rewards.begin()];
				double gtarget = (target - d) / this->_eps;
				for (GaussianT g: gaussians_copy) {
					if (c == this->rebuildConfiguration(std::get<0>(g))) {
						gtarget = std::max(gtarget, 21.0 * std::get<1>(g) * 2.0 * c.size() + 1.0);
						break;
					}
				}
				this->addGaussian(c, gtarget, d);
			}
		}

		this->_testCounter = 0;
	}
}

/**
 * Sample the parameter space according to the gaussian mixture.
 * 
 * @param forbidden Container the NetworkConfiguration we don't want
 * 
 * @return a new NetworkConfiguration, not in forbidden
 */
NetworkConfiguration HGMTSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	std::vector<double> probs;
	for (Gaussian g: this->_gaussians) {
		double weight = std::get<2>(g);
		if (weight < 0) weight = 0.5;

		probs.push_back(weight);
	}

	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());
	NetworkConfiguration selectedConf;
	unsigned int i = 0;
	do {
		i++;
		if (i > 10000) {
			return {std::tuple<double, double>(0, 0)};
		}

		if (i % 1000 == 0)
			this->increaseStds();
		selectedConf = this->rebuildConfiguration(this->normedSample(discreteDist));
	} while (std::find(forbidden.begin(), forbidden.end(), selectedConf) != forbidden.end() || !confConstraint(selectedConf));

	return selectedConf;
}

/**
 * Increase the standard deviations of all gaussians, when possible. 
 */
void HGMTSampler::increaseStds() {
	for (unsigned int i = 0; i < this->_gaussians.size(); i++) {
		double& std = std::get<1>(this->_gaussians[i]);
		std += 1.0 / std::get<0>(this->_gaussians[i]).size();
	}
}

/**
 * Rebuild configuration given a sample coming from a multidimensional normal
 * distribution.
 * 
 * @param sampled Container the sample from the multidim normal distribution
 * 
 * @return the network configuration
 */
NetworkConfiguration HGMTSampler::rebuildConfiguration(const std::vector<double>& sampled) const {
	unsigned int i = 0;
	NetworkConfiguration conf(sampled.size() / 2);
	for (const ConstrainedCouple& c: this->_parameters) {
		conf[(int) (i / 2)] = c.fromNormalized(std::make_tuple(sampled[i], sampled[i + 1]));
		i += 2;
	}

	return conf;
} 

/**
 * Sample from one of the gaussians, chosen with discreteDist
 * 
 * @param discreteDist discrete_distribution a discrete distribution to choose
 * one of the gaussians
 * 
 * @return the vector issued from the normal distribution sampling 
 */ 
std::vector<double> HGMTSampler::normedSample(std::discrete_distribution<int>& discreteDist) {
	int toUse = discreteDist(this->_generator);
	std::vector<double> normed(2 * this->_parameters.size());
	unsigned int i = 0;
	for (double ci: std::get<0>(this->_gaussians[toUse])) {
		std::normal_distribution<double> normalDist(ci, std::get<1>(this->_gaussians[toUse]));
		do {
			normed[i] = normalDist(this->_generator);
		} while (normed[i] < 0 || normed[i] > 1);
		i++;
	}

	return normed;
}

/**
 * Overloading of () operator, use the gaussian mixture to get a new element
 * 
 * @param forbidden Container the configurations we don't want
 * 
 * @return a network configuration not in forbidden
 */
NetworkConfiguration HGMTSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	this->_testCounter++;
	return this->randomActionFromSample(forbidden);
}

/**
 * Build an HGM sampler.
 * 
 * @param parameters Container the parameters that define the sampling space
 * @param def NetworkConfiguration the default configuration of the network
 * @param nMax unsigned int the max number of gaussians in the mixture
 * @param eps double the exploration parameter
 * @param dist double the distance of the new target
 * @param nTests FunctionPointer a way to quantify how many tests must be done 
 */
HCMSampler::HCMSampler(const std::vector<ConstrainedCouple>& parameters, std::vector<NetworkConfiguration> bases, unsigned int nMax, double eps, unsigned int (*nTests)(std::vector<Ring>), DistanceMode dmode) : RandomSampler(parameters), _nTests(nTests), _eps(eps), _dist(5.0), _nMax(nMax), _testCounter(1), _dmode(dmode) {
	for (NetworkConfiguration n: bases)
		this->addRing(n, this->_dist, 0.02, -1);
}

/**
 * Add a gaussian to the gaussian mixture. If the mixture is already full,
 * delete the oldest gaussian.
 * 
 * @param center NetworkConfiguration the center of the gaussian
 * @param dist double the distance to put the target at
 * @param reward double the reward max attained by the gaussian
 */ 
void HCMSampler::addRing(const NetworkConfiguration& center, double dist, double std, double reward) {
	// Turn the NetworkConfiguration into a list of normalized double
	unsigned int n = 2 * center.size();
	std::vector<double> c(n);
	unsigned int i = 0;
	for (std::tuple<double, double> t: center) {
		// Normalize the tuple
		std::tuple<double, double> normalized = this->_parameters[(int) (i / 2)].normalize(t);
		// Add it to the list
		c[i] = std::get<0>(normalized);
		c[i + 1] = std::get<1>(normalized);
		i += 2;
	}

	// Add the circular to the mixture
	if (this->_circulars.size() == this->_nMax)
		this->_circulars.erase(this->_circulars.begin());

	this->_circulars.push_back(std::make_tuple(c, dist / 21.0, std / (21.0 * n), reward));
}

/**
 * Add a NetworkConfiguration and its associated reward to the base of the
 * agent. If the NetworkConfiguration is already present in base, the reward is
 * averaged according to an exponential average (alpha = 0.1). Update also the
 * average reward of the gaussian used.
 * 
 * @param features NetworkConfiguration the network configuration
 * @param reward double the associated reward
 */
void HCMSampler::addToBase(NetworkConfiguration features, double reward) {
	Sampler::addToBase(features, reward);

	// Update a center ?
	for (Ring c: this->_circulars) {
		NetworkConfiguration ct = this->rebuildConfiguration(std::get<0>(c));
		if (features == ct) {
			double& maxRew = std::get<3>(c);
			if (maxRew == -1) maxRew = reward;
			else maxRew = 0.9 * maxRew + 0.1 * reward;
		}
	}

	// std::cout << this->_testCounter << " vs. " << this->_nTests(this->_circulars) << std::endl;

	// Enough tests to change the mixture
	if (this->_dmode == CYCLE || this->_testCounter >= this->_nTests(this->_circulars)) {
		// Find the n greatest rewards
		std::vector<double> vec_to_sort = this->_rewards;
		std::sort(vec_to_sort.begin(), vec_to_sort.end(), std::greater<double>());
		int n = std::min(this->_nMax, (unsigned int) vec_to_sort.size());
		std::vector<double> dest(n);
		for (int i = 0; i < n; i++) {
			dest[i] = vec_to_sort[i];
		}

		// Add new circulars best on the n best
		double max_elem = dest[0];
		double target = std::min(this->_dist * this->_eps + max_elem, 1.0);

		if (this->_dmode == CYCLE) {
			double min_v = 1.0;
			double max_v = 10.0;
			unsigned int cycle_size = 36;
			double t_pos = 2.0 * (double) std::min(this->_testCounter % cycle_size, cycle_size - (this->_testCounter % cycle_size)) / (double) cycle_size;
			double dist = t_pos * (max_v - min_v) + min_v;
			target = std::min(1.0, max_elem + dist * this->_eps);
			// std::cout << this->_testCounter << " and " << t_pos << " and " << dist << " and " << target << std::endl;
		}

		std::vector<Ring> circulars_copy = this->_circulars;
		this->_circulars.clear();
		for (double d: dest) {
			if (d != -1) {
				NetworkConfiguration c = this->_features[std::find(this->_rewards.begin(), this->_rewards.end(), d) - this->_rewards.begin()];
				double ctarget = std::max((target - d) / this->_eps, 1.0);
				// std::cout << ctarget << std::endl;
				this->addRing(c, ctarget, 0.02, d);
			}
		}
		// std::cout << std::endl;

		if (this->_dmode != CYCLE)
			this->_testCounter = 0;
	}
}

/**
 * Sample the parameter space according to the gaussian mixture.
 * 
 * @param forbidden Container the NetworkConfiguration we don't want
 * 
 * @return a new NetworkConfiguration, not in forbidden
 */
NetworkConfiguration HCMSampler::randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	std::vector<double> probs;
	for (Ring c: this->_circulars) {
		double weight = std::get<3>(c);
		if (weight < 0) weight = 0.5;

		probs.push_back(weight);
	}

	// this->printRings();

	std::discrete_distribution<int> discreteDist(probs.begin(), probs.end());
	NetworkConfiguration selectedConf;
	unsigned int i = 0;
	do {
		i++;
		if (i > 10000) {
			return {std::tuple<double, double>(0, 0)};
		}

		selectedConf = this->rebuildConfiguration(this->normedSample(discreteDist));
		// for (std::tuple<double, double> t: selectedConf) { std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),"; }
		// std::cout << confConstraint(selectedConf) << std::endl;
	} while (std::find(forbidden.begin(), forbidden.end(), selectedConf) != forbidden.end() || !confConstraint(selectedConf));

	return selectedConf;
}

/**
 * Increase the standard deviations of all gaussians, when possible. 
 */
void HCMSampler::shiftCenter() {
	for (unsigned int i = 0; i < this->_circulars.size(); i++) {
		bool tx = std::uniform_int_distribution<unsigned int>(0, 99)(this->_generator) >= 25;
		std::vector<double>& center = std::get<0>(this->_circulars[i]);
		unsigned int j = std::uniform_int_distribution<unsigned int>(0, center.size() / 2 - 1)(this->_generator);
		unsigned int idx = 2 * j + tx;
		double delta = ((center[idx] < 0.25) - (0.25 < center[idx])) / 21.0;
		center[idx] += delta;

		if (!confConstraint(rebuildConfiguration({center[2 * j + 1], center[2 * j]}))) {
			center[idx] -= delta;
			i--;
		}
	}
}

void HCMSampler::printRings() const {
	for (Ring c: this->_circulars) {
		std::cout << "Ring: ";
		for (std::tuple<double, double> t: rebuildConfiguration(std::get<0>(c))) { std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),"; }
		std::cout << " " << std::get<1>(c) << std::endl;
	}
}

/**
 * Rebuild configuration given a sample coming from a multidimensional normal
 * distribution.
 * 
 * @param sampled Container the sample from the multidim normal distribution
 * 
 * @return the network configuration
 */
NetworkConfiguration HCMSampler::rebuildConfiguration(const std::vector<double>& sampled, std::vector<unsigned int> location) const {
	unsigned int i = 0;
	NetworkConfiguration conf(sampled.size() / 2);
	if (location.size() == 0) {
		for (unsigned int l = 0; l < sampled.size() / 2; l++)
			location.push_back(l);
	}

	for (unsigned int l: location) {
		conf[(int) (i / 2)] = this->_parameters[l].fromNormalized(std::make_tuple(sampled[i], sampled[i + 1]));
		i += 2;
	}
	return conf;
} 

/**
 * Sample from one of the gaussians, chosen with discreteDist
 * 
 * @param discreteDist discrete_distribution a discrete distribution to choose
 * one of the gaussians
 * 
 * @return the vector issued from the normal distribution sampling 
 */ 
std::vector<double> HCMSampler::normedSample(std::discrete_distribution<int>& discreteDist) {
	int toUse = discreteDist(this->_generator);
	unsigned int n = 2 * this->_parameters.size();
	unsigned int i = 1;
	std::vector<double> normed(n);
	std::vector<double>& center = std::get<0>(this->_circulars[toUse]);
	bool sample_not_ok = true;
	std::vector<double> hypersphere_sample(n);

	do {
		if (i % 500000 == 0) {
			this->shiftCenter();
		}

		std::normal_distribution<double> normalDist(0, 1);
		// Sample a direction in space
		for (unsigned int j = 0; j < n; j++) {
			hypersphere_sample[j] = normalDist(this->_generator);
		}
		
		// Normalize the vector
		double norm = 0;
		for (double d: hypersphere_sample) norm += d * d;
		norm = sqrt(norm);
		for (unsigned int j = 0; j < n; j++) hypersphere_sample[j] /= norm;

		sample_not_ok = false;
		// normalDist = std::normal_distribution<double>(std::get<1>(this->_circulars[toUse]), std::get<2>(this->_circulars[toUse]));
		double dist = std::get<1>(this->_circulars[toUse]); // normalDist(this->_generator);
		for (unsigned int j = 0; j < n; j++) {
			hypersphere_sample[j] = hypersphere_sample[j] * dist + center[j];
			sample_not_ok = hypersphere_sample[j] < 0 || hypersphere_sample[j] > 1;
			if (j % 2 == 1) {
				std::vector<double> couple({hypersphere_sample[j-1], hypersphere_sample[j]});
				if (!confConstraint(rebuildConfiguration(couple))) {
					sample_not_ok = true;
				}
			}
			if (sample_not_ok)
					break;
		}
		i++;
	} while (sample_not_ok);

	return hypersphere_sample;
}

/**
 * Overloading of () operator, use the gaussian mixture to get a new element
 * 
 * @param forbidden Container the configurations we don't want
 * 
 * @return a network configuration not in forbidden
 */
NetworkConfiguration HCMSampler::operator()(const std::vector<NetworkConfiguration>& forbidden /*= std::vector<NetworkConfiguration>()*/) {
	this->_testCounter++;
	return this->randomActionFromSample(forbidden);
}