/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/

#include <mex.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/graph/connected_components.hpp>


#include "ms.hpp"

#define NOISE_LEVEL 1e-6 

namespace {

boost::mt19937 gen;
boost::uniform_real<double> const uni_dist(0,1);
boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > real01(gen, uni_dist);



double err = 0;

} //anon


extern "C" bool utIsInterruptPending();

using namespace std;

void bail(char const * exp, char const * file , char const * fun, int line)
{
	mexPrintf("assert (%s) failed in function %s in file  \"%s\" +%i\n", exp, fun, file, line);
	mexErrMsgTxt("Computation aborted");	
}



MsgSteinerGraph::TreeChecker 
converge(MsgSteinerGraph & M, int maxit, double tolerance, int decision, double beta)
{
	int it = 0;
	double err;

	double cost = inf;
	double mincost = inf;
	int coincide = 0;
	double allcoincide = 0;

	MsgSteinerGraph::TreeChecker cbest(M);
	do {
		M.rein = beta * it;
		err = M.iterate();
		double oldcost = cost;
		MsgSteinerGraph::TreeChecker c(M);
		cost = c.cost;
		allcoincide *= .9;

		//mexPrintf("%i %g %g %g %g %i\n", it, cost, c.ecost, c.vcost, allcoincide/10, c.dmax);
		if (! c.unconnected) {
			if (cost < mincost) {
				mincost = cost;
				mexPrintf("%i %g %g %g %g %i\n", it, cost, c.ecost, c.vcost, allcoincide/10, c.dmax);
				mexEvalString("drawnow");  /* fflush stdout (zozzeria) */
				cbest = c;

			}
			if (fabs(oldcost - c.cost) < tolerance) {
				allcoincide++;
				coincide++;
			} else
				coincide = 0;
		} else {
			coincide = 0;
		}

		if (utIsInterruptPending()){
			//free_memory();
			mexErrMsgTxt("Caught a CTRL-C exiting ...\n");
		}

	} while (coincide < decision && err > tolerance && ++it < maxit);
	
	return cbest;
}

void usage()
{
	mexErrMsgTxt("Usage: EdgeWeights<SparseMatrix, NxN> VertexPrizes<Vector, Nx1> D<scalar> niter<scalar> [rein<scalar>]");
}

static void
read(MsgSteinerGraph & M, mxArray const * xEdges, mxArray const * xVertices, unsigned rootid)
{
	/* Get the starting positions of all three data arrays. */ 
	double * pr = mxGetPr(xEdges);
	mwIndex * ir = mxGetIr(xEdges);
	mwIndex * jc = mxGetJc(xEdges);

	/* Display the nonzero elements of the sparse array. */ 
	mwSize total = 0;
	mwSize const n = mxGetN(xEdges);
	if (mxGetN(xVertices) != size_t(n))
		usage();
	vector<pair<pair<unsigned, unsigned>, double> > edges;
	for (mwSize col = 0; col < n; ++col)  { 
		mwIndex const srow = jc[col], erow = jc[col + 1]; 
		for (mwIndex r = srow; r < erow; ++r) {
			mwIndex const row = ir[r];
			//mexPrintf("E\t(%"FMT_SIZE_T"u,%"FMT_SIZE_T"u) = %g\n", 
			//		row + 1, 
			//		col + 1, pr[total]);
			if (row != col)
				edges.push_back(make_pair(make_pair(row, col), pr[total] + /*noise=*/ NOISE_LEVEL * real01()));
			++total;
		}
	}
	vector<double> vertices(mxGetPr(xVertices), mxGetPr(xVertices) + n);
	//for (unsigned i = 0, n = vertices.size(); i < n; ++i) {
	//	mexPrintf("W\t%d\t%f\n", i, vertices[i]);
	//}
	M.read_graph(edges, vertices, rootid - 1);
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 5)
		usage();

	mxArray const * xEdges = prhs[0];
	mwSize const n = mxGetN(xEdges);
	mxArray const * xVertices = prhs[1];
	int const depth = mxGetScalar(prhs[2]);
	MsgSteinerGraph M(depth);

	int maxit = mxGetScalar(prhs[3]);
	unsigned root = mxGetScalar(prhs[4]);

	double beta = nrhs > 5 ? mxGetScalar(prhs[5]) : 0;

	if (!mxIsSparse(xEdges) || mxIsComplex(xEdges) || mxIsSparse(xVertices) 
		|| mxGetN(xEdges) != mxGetM(xEdges) || mxGetN(xEdges) != mxGetN(xVertices) 
		|| root < 1 || root > n || depth <= 0)
		usage();

	read(M, xEdges, xVertices, root);
	MsgSteinerGraph::TreeChecker c = converge(M, maxit, 1e-10, 10, beta);
	
	int N = c.tree.size();

	if (nlhs > 0) {
		plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
		double * outP = mxGetPr(plhs[0]);
		for (int n = 0; n < N; ++n) 
			outP[n] = c.tree[n] + 1;
	}
	if (nlhs > 1) {	
		plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
		double * outS = mxGetPr(plhs[1]);
		outS[0] = c.vcost + c.ecost;
	}
}


