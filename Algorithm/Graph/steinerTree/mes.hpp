/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with MSGSTEINER; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */




#ifndef MES_H
#define MES_H



#include "proba.hpp"

#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <iostream>


struct Mes {
	Mes(int depth) : A(depth), E(depth), F(depth), B(0), D(0), depth_(depth) { i_assert(depth_); }
	//Mes(Mes const & other): A(other.A), E(other.E), F(other.F), B(other.B), D(other.D), depth_(other.depth_) {}

	int depth() const { return depth_; }
	
	Proba A, E, F;
	double B, D;

	friend std::ostream & operator<<(std::ostream & ost, Mes const & m);

	double maximum() const {
		double m = std::max(B, D);
		for (int d = 0; d < depth_; ++d)
			m = std::max(m, std::max(A[d], E[d]));
		return m;
	}

	Mes & operator=(Mes const & other) {
		A = other.A;
		E = other.E;
		F = other.F;
		B = other.B;
		D = other.D;
		depth_ = other.depth_;
		return *this;
	}
	
	void reduce() {
		double const m = maximum();
		//i_assert(m > -inf/3);
		
		D -= m;
		B -= m;
		for (int d = 0; d < depth_; ++d) {
			A[d] -= m;
			E[d] -= m;
		}
	}
	friend void swap(Mes & u, Mes & v)
	{
		swap(u.A, v.A);
		swap(u.E, v.E);
		swap(u.F, v.F);
		std::swap(u.B, v.B);
		std::swap(u.D, v.D);
	}


private:
	int depth_;
};
double l8dist(Mes const & a,  Mes const & b);

Mes operator+(Mes const & a, Mes const & b);

Mes operator*(double c, Mes const & b);



template<class R>
void randomize(Mes & m, R & mes_real01)
{

	m.B = -mes_real01();
	m.D = m.B;
	for (int d = m.depth(); d--; ) {
		m.A[d] = -mes_real01();
		m.D = std::max(m.A[d], m.D);
	}
	for (int d = m.depth(); d--; ) {
		m.F[d] = 0;
		m.E[d] = d + 1 < m.depth() ? std::max(-mes_real01(), m.D) : m.D;
	}
}

#endif
