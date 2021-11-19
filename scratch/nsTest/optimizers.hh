#ifndef __OPTIMIZERS_HH
#define __OPTIMIZERS_HH

#include <vector>
#include <set>
#include <string>
#include <map>
#include <chrono>

#include "gp.h"
#include "gp_utils.h"
#include "my_gps.hh"

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

#include "samplers.hh"

using History = std::vector<std::tuple<NetworkConfiguration, double>>;
bool lexCompareVectors(Eigen::VectorXd v, Eigen::VectorXd w);
class Optimizer {
	public:
		Optimizer(Sampler* sampler, unsigned int testPeriod = 1);
		unsigned int getTestPeriod() const;
		void sampleValidConfig(unsigned int nCouples, std::vector<gsl_vector*>& toFill);
		void sampleValidConfigNoNorm(unsigned int nCouples, std::vector<gsl_vector*>& toFill);
		virtual ~Optimizer();
		void showDecisions() const;
		virtual bool readyForAnother() const;
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize() = 0;

	protected:
		Sampler* _sampler;
		History _history;
		std::default_random_engine _generator;
		std::uniform_real_distribution<double> _distribution;
		unsigned int _testPeriod;
};

class IdleOptimizer : public Optimizer {
	public:
		IdleOptimizer();
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		NetworkConfiguration _chosen;
};

class EpsilonGreedyOptimizer : public Optimizer {
	public:
		EpsilonGreedyOptimizer(Sampler* sampler, double epsilon);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		double _epsilon;
		std::map<NetworkConfiguration, double> _results;
};

using GammaNormalSample = std::tuple<double, double, double, double, std::vector<double>, bool>;
class ThompsonGammaNormalOptimizer : public Optimizer {
	public:
		ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double eps, unsigned int to_ig = 0, bool chain = true);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual bool readyForAnother() const;
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, GammaNormalSample> _gammaNormals;
		NetworkConfiguration _chosen;
		unsigned int _testLeft;
		double _epsilon;
		unsigned int _to_ig;
		bool _chain;
};

using NormalParameters = std::tuple<double, double, unsigned int>;
class ThompsonNormalOptimizer : public Optimizer {
	public:
		ThompsonNormalOptimizer(Sampler* sampler, double eps);
		virtual void addToBase(NetworkConfiguration configuration, double reward, bool forward=true, std::vector<std::tuple<double, unsigned int>> individual_rewards=std::vector<std::tuple<double, unsigned int>>());
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, NormalParameters> _normals;
		double _epsilon;
};

#endif