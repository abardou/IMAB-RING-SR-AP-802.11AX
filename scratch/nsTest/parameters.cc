#include "parameters.hh"

/**
 * SingleParameter constructor
 * 
 * @param min double the min value of the parameter (included)
 * @param max double the max value of the parameter (included)
 * @param step double the gap between two values of the parameter
 */
SingleParameter::SingleParameter(double min, double max, double step) : _min(min), _max(max), _step(step) {
	srand(std::chrono::system_clock::now().time_since_epoch().count());
}

/**
 * Compute the number of possible values
 * 
 * @return the number of possible values
 */
inline unsigned int SingleParameter::numberOfValues() const {
	return 1 + (this->_max - this->_min) / this->_step;
}

/**
 * Convert a normalized value to its original value, according to the interval
 * defined by the parameter
 * 
 * @param normalizedValue double the normalized value
 * 
 * @return the original value 
 */ 
double SingleParameter::fromNormalized(double normalizedValue) const {
	unsigned int n = round((this->numberOfValues() - 1) * normalizedValue);

	return this->_min + n * this->_step;
}

/**
 * Normalize a value according to the interval defined by the parameter.
 * 
 * @param value double the value to normalize
 * 
 * @return the value normalized
 */
inline double SingleParameter::normalize(double value) const {
	return (value - this->_min) / (this->_max - this->_min);
}

/**
 * Return a random value within the interval defined by the parameter.
 * 
 * @return a random value
 */ 
inline double SingleParameter::randomValue() const {
	return this->_min + this->_step * (rand() % this->numberOfValues());
}

/**
 * Default ConstrainedCouple constructor, does not initialize a valid object.
 */
ConstrainedCouple::ConstrainedCouple() : _p1(SingleParameter(0,0,0)), _p2(SingleParameter(0,0,0)), _constraint(nullptr) {  }

/**
 * ConstrainedCouple constructor
 * 
 * @param p1 SingleParameter the first parameter
 * @param p2 SingleParameter the second parameter
 * @param constraint Pointer boolean function for the constraint
 */
ConstrainedCouple::ConstrainedCouple(SingleParameter p1, SingleParameter p2, bool (*constraint)(double, double)) : _p1(p1), _p2(p2), _constraint(constraint) {  }

/**
 * Retrieve the original value of a normalized one according to intrinsec
 * parameters intervals.
 * 
 * @param normalizedValue std::tuple<double,double> the normalized value
 * 
 * @return the original value of a normalized one
 */
std::tuple<double, double> ConstrainedCouple::fromNormalized(const std::tuple<double, double>& normalizedValue) const {
	return std::make_tuple(this->_p1.fromNormalized(std::get<0>(normalizedValue)), this->_p2.fromNormalized(std::get<1>(normalizedValue)));
}

/**
 * Normalize each value of the constrained couple according to their respective
 * intervals
 * 
 * @param value std::tuple<double,double> the value to normalize
 * 
 * @return the tuple of normalized values 
 */
std::tuple<double, double> ConstrainedCouple::normalize(std::tuple<double, double> value) const {
	return std::make_tuple(this->_p1.normalize(std::get<0>(value)), this->_p2.normalize(std::get<1>(value)));
}

/**
 * Produce a tuple of random values that ensures the constraint
 * 
 * @return a tuple of random values
 */
std::tuple<double, double> ConstrainedCouple::randomValue() const {
	double a, b;
	do {
		a = this->_p1.randomValue();
		b = this->_p2.randomValue();
	} while (!this->_constraint(a, b));

	return std::make_tuple(a, b);
}