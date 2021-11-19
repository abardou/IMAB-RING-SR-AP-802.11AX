#ifndef __PARAMETERS_HH
#define __PARAMETERS_HH

#include <cmath>
#include <tuple>
#include <chrono>

class SingleParameter {
	public:
		SingleParameter(double min, double max, double step);
		unsigned int numberOfValues() const;
		double fromNormalized(double normalizedValue) const;
		double normalize(double value) const;
		double randomValue() const;

	protected:
		double _min, _max, _step;
};

class ConstrainedCouple {
	public:
		ConstrainedCouple();
		ConstrainedCouple(SingleParameter p1, SingleParameter p2, bool (*constraint)(double, double));
		std::tuple<double, double> fromNormalized(const std::tuple<double, double>& normalizedValue) const;
		std::tuple<double, double> normalize(std::tuple<double, double> value) const;
		std::tuple<double, double> randomValue() const;

	protected:
		SingleParameter _p1, _p2;
		bool (*_constraint)(double, double);
};

#endif