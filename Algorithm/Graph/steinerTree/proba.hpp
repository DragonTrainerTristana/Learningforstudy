/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/


#ifndef PROBA_H
#define PROBA_H

#include <iostream>
#include <cstring>
#include <cmath>
#include "assertion.hpp"

//double const inf = std::numeric_limits<double>::max();
double const inf = 1e10;



class Proba {
public:	
	//Proba(int depth) : p_(new double[depth]), depth(depth) { i_assert(0 && depth); }
	Proba(int depth, double val = 0) : p_(new double[depth]), depth(depth) {
		i_assert(depth);
		for (int d = 0; d < depth; ++d)
			p_[d] = val;
	}
	Proba(Proba const & other) : p_(new double[other.depth]), depth(other.depth) {
		*this = other;
	}
	Proba & operator=(Proba const & other) {
		i_assert(depth == other.depth);
		//std::copy(p_, p_ + depth, other.p_);
		memcpy(p_, other.p_, depth * sizeof(*other.p_));
		//for (int d = 0; d < depth; ++d)
		//	p_[d] = other[d];
		return *this;
	}
	~Proba() { delete[] p_; }
	double & operator[](int d) {
		return p_[d];
	}
	double const & operator[](int d) const{
		return p_[d];
	}
	friend std::ostream & operator<<(std::ostream & ost, Proba const & p) {
		for (int d = 0; d < p.depth; ++d)
			ost << p[d] << " ";
		return ost;
	}
	Proba & operator+=(Proba const & b) {
		for (int d = depth; d--; )
			p_[d] += b[d];
		return *this;
	}

	Proba & operator*=(double b) {
		for (int d = depth; d--; )
			p_[d] *= b;
		return *this;
	}

	friend double l8dist(Proba const & a, Proba const & b) {
		double n = 0;
		for (int d = a.depth; d--;)
			n = std::max(n, fabs(a[d] - b[d]));
		return n;
	}

	friend void swap(Proba & A, Proba & B) {
		i_assert(A.depth == B.depth);
		//std::swap(A.depth, B.depth);
		std::swap(A.p_, B.p_);
	}

	int size() const { return depth; }	
private:
	double * p_;
	int depth;
};



double l8dist(Proba const & a,  Proba const & b);



#endif
