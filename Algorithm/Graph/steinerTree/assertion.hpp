/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/

#ifndef ASSERTION_H
#define ASSERTION_H




#ifdef HARDASSERT

#include <cassert>

#define i_assert(expr) assert(expr)

#else

void bail(char const *, char const *, char const *, int);

#define i_assert(expr) { if(!(expr)) bail(#expr, __FILE__, __FUNCTION__, __LINE__); }

#endif

#endif
