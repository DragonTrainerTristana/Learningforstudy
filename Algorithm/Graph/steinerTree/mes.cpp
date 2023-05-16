/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/

#include "mes.hpp"


std::ostream & operator<<(std::ostream & ost, Mes const & m)
{
	ost << "A: " << m.A << std::endl;
	ost << "E: " << m.E << std::endl;
	ost << "F: " << m.F << std::endl;
	ost << "B: " << m.B << " D: " << m.D << std::endl;
	return ost;
}

double l8dist(Mes const & a,  Mes const & b)  
{
	double l8 = 0;
//	if (a.B < -inf/2 || b.B < -inf/2)
//		return inf;
//	if (a.D < -inf/2 || b.D < -inf/2)
//		return inf;
	l8 = std::max(l8, l8dist(a.A, b.A));
	l8 = std::max(l8, l8dist(a.E, b.E));
	l8 = std::max(l8, fabs(a.B - b.B));
	l8 = std::max(l8, fabs(a.D - b.D));
	return l8;
}


Mes operator+(Mes const & a, Mes const & b) 
{
	Mes out = a;
	out.A += b.A;
	out.B += b.B;
	out.E += b.E;
	out.D += b.D;
	return out;
}


Mes operator*(double cc, Mes const & b) 
{
	Mes out = b;
	out.A *= cc;
	out.B *= cc;
	out.E *= cc;
	out.D *= cc;
	return out;
}


