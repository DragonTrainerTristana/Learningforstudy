/*
   Copyright 2010-2011 Alfredo Braunstein

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

#include <boost/config.hpp>

#include "mes.hpp"
#include "ms.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/graph/connected_components.hpp>

#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <math.h>
#include <iomanip>
#include <boost/limits.hpp>
#include <queue>
#include <omp.h>

#define TERMINAL_INFTY 10000

using namespace boost;
using namespace std;

mt19937 gen;
mt19937 mes_gen;

uniform_real<double> const uni_dist(0,1);
variate_generator<mt19937 &, uniform_real<> > real01(gen, uni_dist);
variate_generator<mt19937 &, uniform_real<> > mes_real01(mes_gen, uni_dist);


double & MsgSteinerGraph::getC(Edge e)
{ 
	return source(e, g) < target(e, g) ? g[e].cji : g[e].cij; 
}

double const & MsgSteinerGraph::getC(Edge e) const
{ 
	return source(e, g) < target(e, g) ? g[e].cji : g[e].cij; 
}


inline void 
MsgSteinerGraph::getMes(Edge e, Mes & out) const
{
	EdgeProperty const & ep = g[e];
	omp_set_lock(&ep.lock);
	out = source(e, g) < target(e, g) ? ep.ji : ep.ij;
	omp_unset_lock(&ep.lock);
}


inline void 
MsgSteinerGraph::getBoth(Edge e, pair<Mes, Mes> & both) const
{
	EdgeProperty const & ep = g[e];
	omp_set_lock(&ep.lock);
	if (source(e, g) < target(e, g)) {
		both.first = ep.ji;
		both.second = ep.ij;
	} else {
		both.first = ep.ij;
		both.second = ep.ji;
	}
	omp_unset_lock(&ep.lock);
}



inline void
MsgSteinerGraph::getMesInv(Edge e, Mes & out) const
{
	EdgeProperty const & ep = g[e];
	omp_set_lock(&ep.lock);
	out = source(e, g) > target(e, g) ? ep.ji : ep.ij;
	omp_unset_lock(&ep.lock);
}

inline void 
MsgSteinerGraph::setMes(Edge e, Mes & out) 
{
	EdgeProperty & ep = g[e];
	omp_set_lock(&ep.lock);
	(source(e, g) > target(e, g) ? ep.ji : ep.ij) = out;
	omp_unset_lock(&ep.lock);
}


double 
MsgSteinerGraph::updateroot(Vertex i)
{
	Mes out(depth);

	out.B = -inf;
	for (int d = 0; d < depth; ++d) {
		out.A[d] = d ? -inf : 0;
		out.E[d] = 0;
	}
	out.D = 0;

	edge_iterator eit, eend;
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) 
		setMes(*eit, out);
	return 0;
}




void   
MsgSteinerGraph::chooseroot(int maxit, double tolerance, int decision, double beta)
{
	cerr << "Choosing root..." << endl;
	rootid = idx("_ROOTMSGSTEINER");
	vertex_iterator vit, vend;
	for (tie(vit, vend) = vertices(g); vit != vend; ++vit) if (rootid != *vit) {
		Edge e; bool b;
		tie(e, b) = edge(*vit, rootid, g);
		getC(b ? e : add_edge(*vit, rootid, EdgeProperty(depth, inf, inf), g).first) = inf / num_vertices(g);
	}
	converge(maxit, tolerance, decision, beta);

	double maxtot = -inf;
	Vertex maxi = rootid;

	edge_iterator eit, eend;

	Mes in(depth);
	for (tie(eit, eend) = out_edges(rootid, g); eit != eend; ++eit) {
		Vertex i = target(*eit, g);
		getMes(*eit, in);
		if (in.F[1] - g[i].G >  maxtot) {
			maxtot = in.F[1] - g[i].G;
			maxi = i;
		}
		g[i].G = 0;
	}
	i_assert(maxi != rootid);
	
	cerr << "... done. root: " << g[maxi].name << endl;
	clear_vertex(rootid, g);
	remove_vertex(rootid, g);
	rootid = maxi;

	
	graph_edge_iterator geit, geend;
	for (tie(geit, geend) = edges(g); geit != geend; ++geit) {
		g[*geit].ij = Mes(depth);
		g[*geit].ji = Mes(depth);
		randomize(g[*geit].ij, mes_real01);
		randomize(g[*geit].ji, mes_real01);
	}

	g[maxi].type = VertexProperty::root;
}


/* This helper class finds a cavity maximum of a set of values. Values are added through 
its push_back member. */

template<class T, class S = int>
class CavityMaximizer {
public:
	CavityMaximizer() : first_(-inf), second_(-inf), idxfirst_(-1), size_(0) {}
	void push_back(T const & m) { 
		if (first_ <= m) {
			second_ = first_;
			first_ = m;
			idxfirst_ = size_;
		} else if (second_ < m)
			second_ = m;
		++size_;
	}

	T max_except(S idx) const { return idx == idxfirst_ ? second_ : first_; }
private:
	T first_;
	T second_;
	S idxfirst_;
	S size_;
};

double  
MsgSteinerGraph::update(Vertex i) 
{
	if (i == rootid)
		return updateroot(i);

	Proba sumE(depth, 0);
	vector<CavityMaximizer<double> > cmax(depth);
	double sumD = 0;
	edge_iterator eit, eend;
	vector<pair<Mes, Mes> > & M = Mcache[omp_get_thread_num()];
	
	int idx = 0;
	if (out_degree(i, g) > M.size())
		M.resize(out_degree(i, g), make_pair(Mes(depth), Mes(depth)));
	
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit, ++idx) {
		double const c = getC(*eit);
		getBoth(*eit, M[idx]);
		Mes const & in = M[idx].first;
		Mes const & out = M[idx].second;
		for (int d = 1; d < depth; ++d) {
			sumE[d] += in.E[d];
			cmax[d].push_back(-c - in.E[d] + in.A[d - 1] + rein * out.F[d]);
		}
		sumE[0] += in.E[0];
		cmax[0].push_back(-inf);
		sumD += in.D;
	}

	Mes & old = oldcache[omp_get_thread_num()];
	double eps = 0;
	idx = 0;
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit, ++idx) {
		double const c = getC(*eit);
		Mes & in = M[idx].first;
		Mes & out = M[idx].second;
		swap(old, out);
		double maxA = -inf;
		double sumDp = sumD - in.D;
		out.B = -g[i].c + sumDp + rein * g[i].G;
		for (int d = 0; d < depth; ++d) {
			out.A[d] = sumE[d] - in.E[d] + cmax[d].max_except(idx);
			maxA = max(maxA, out.A[d]);
		}

		out.D = max(out.B, maxA);
		
		double C = -inf;
		for (int d = depth; --d; ) {
			out.E[d] = max(out.D, C);
			double sumEp = sumE[d] - in.E[d];
			C = -c + sumEp + rein * old.F[d];
			out.F[d] = C + in.A[d - 1];
		}
		out.F[0] = -inf;
		out.E[0] = out.D;

		out.reduce();

		eps = max(eps, l8dist(out, old));
		setMes(*eit, out);
	}
	g[i].G = -g[i].c + sumD + rein * g[i].G;

	return eps;
}

double  
MsgSteinerGraph::iterate()
{
	unsigned const n = num_vertices(g);
	vector<int> permutation(n);
	for (unsigned j = 0; j < n; ++j) 
		permutation[j] = j;
	
	random_shuffle(permutation.begin(), permutation.end());

	double eps = 0;
#pragma omp parallel for
	for (unsigned j = 0; j < n; ++j) {
		double err = update(permutation[j]);
#pragma omp critical
		eps = max(eps, err);
	}
	
	return eps;
}


pair<bool, MsgSteinerGraph::Edge>  
MsgSteinerGraph::marginal(Vertex i) const
{
	if (g[i].isroot() || !out_degree(i, g))
		return make_pair(false, Edge());

	double maxtot = -inf;
	Edge maxtotedge = *out_edges(i, g).first;

	edge_iterator eit, eend;

	Mes out(depth);
	for (tie(eit, eend) = out_edges(i, g); eit != eend; ++eit) {
		getMesInv(*eit, out);
		for (int d = 0; d < depth; ++d) {
			if (out.F[d] >= maxtot) {
				maxtot = out.F[d];
				maxtotedge = *eit;
			}
		}
	}

	if (maxtot > g[i].G)
		return make_pair(true, maxtotedge);

	return make_pair(false, Edge());
}


MsgSteinerGraph::TreeChecker::TreeChecker(MsgSteinerGraph const & msg) : tree(num_vertices(msg.g), -1) {
	Graph const & g = msg.g;
	typedef adjacency_list<vecS, vecS, undirectedS> SimpleGraph;
	SimpleGraph g2(num_vertices(g));
	unconnected = 0;
	dmax = 0;
	nnodes = 1;
	ecost = 0;
	vcost = 0;
	cost = 0;

	vertex_iterator vit, vend;
	for (tie(vit, vend) = vertices(g); vit != vend; ++vit) if (*vit != msg.rootid) {
		bool a;
		Edge e;
		tie(a,e) = msg.marginal(*vit);
		if (a) {
			SimpleGraph::vertex_descriptor v1 = target(e, g);
			tree[*vit] = v1;
			add_edge(v1, *vit, g2);
			ecost += msg.getC(e);
		} else
			vcost += g[*vit].c;
	}
	std::vector<int> distances(num_vertices(g2));
	breadth_first_search(g2,
			msg.rootid,
			visitor(make_bfs_visitor(record_distances(&distances[0], on_tree_edge()))));
	for (SimpleGraph::vertex_descriptor j = 0; j < num_vertices(g2); ++j) {
		if (out_degree(j, g2) > 0 && j != msg.rootid) {
			++nnodes;
			if (distances[j] == 0)
				++unconnected;
			else
				dmax = max(dmax, distances[j]);
		}
	}
	cost = vcost + ecost;
}



MsgSteinerGraph::TreeChecker
MsgSteinerGraph::converge(int maxit, double tolerance, int decision, double beta)
{
	int it = 0;
	double err;

	double cost = inf;
	double mincost = inf;
	int coincide = 0;
	double allcoincide = 0;
	TreeChecker cbest(*this);

	do {
		rein = beta * it;
		err = iterate();
		double oldcost = cost;
		TreeChecker c(*this);
		cost = c.cost;
		allcoincide *= .9;
		if (! c.unconnected) {
			if (cost < mincost) {
				cbest = c;
				mincost = cost;
				cerr << it << " " << cost << " " << c.ecost << " " << c.vcost 
					<< " " << (allcoincide/10) << " " <<  c.dmax << endl;
			}
			if (fabs(oldcost - c.cost) < tolerance) {
				allcoincide++;
				coincide++;
			} else
				coincide = 0;
		} else
			coincide = 0;

	} while (coincide < decision && err > tolerance && ++it < maxit);
	
	return cbest;
}


int  
MsgSteinerGraph::idx(string const & id)
{
	static map <string, int> idx_map;
	map<string, int>::iterator mit = idx_map.find(id);
	if (mit == idx_map.end()) {
		idx_map[id] = add_vertex(VertexProperty(id), g);
		return idx_map[id];
	}
	return mit->second;
}

typedef vector<pair<pair<unsigned, unsigned>, double> > CrudeSparseGraph;
void MsgSteinerGraph::read_graph(CrudeSparseGraph const & edges, vector<double> const & vertices, unsigned r)
{
	g = Graph(vertices.size());

	for (unsigned e = 0; e < edges.size(); ++e) {
		Vertex v1 = edges[e].first.first;
		Vertex v2 = edges[e].first.second;
		Edge ed; bool b;
		tie(ed, b) = edge(v1, v2, g);
		getC(b ? ed : add_edge(v1, v2, EdgeProperty(depth, inf, inf), g).first) = edges[e].second;
	}
	for (unsigned v = 0; v < vertices.size(); ++v) {
		g[v].c = vertices[v];
	}

	rootid = r;
	oldcache = vector<Mes>(omp_get_max_threads(), Mes(depth));
	Mcache = vector<vector<pair<Mes, Mes> > >(omp_get_max_threads());
}


void  
MsgSteinerGraph::read_graph(istream & file, double noise)
{
	string tok, tok2;

	while (file >> tok) {
		if (tok == "T") {
			file >> tok2;
			int id = idx(tok2);
			g[id].c = TERMINAL_INFTY;
			//cerr <<  "read T " << tok2 << endl;
		} else if (tok == "W") {
			double w;
			file >> tok2 >> w;
			int id = idx(tok2);
			g[id].c = w + noise * real01();
			//cerr <<  "read W " << tok2 << " " <<  w << endl;
		} else if (tok == "R") {
			file >> tok2;
			int id = idx(tok2);
			g[id].type = VertexProperty::root;
			rootid = id;
		} else if (tok == "E") {
			string i, j;
			double w;
			file >> i >> j >> w;
			Vertex v1 = vertex(idx(i), g);
			Vertex v2 = vertex(idx(j), g);
			i_assert(v1 != v2);
			Edge e; bool b;
			
			// warning: this check could be costly
			tie(e, b) = edge(v1, v2, g);
			if (!b)
				e = add_edge(v1, v2, EdgeProperty(depth, inf, inf), g).first;
			g[e].cij = w  + noise * real01(); 
			g[e].cji = w  + noise * real01(); 
		} else if (tok == "D") {
			string i, j;
			double w;
			file >> i >> j >> w;
			Vertex v1 = vertex(idx(i), g);
			Vertex v2 = vertex(idx(j), g);
			i_assert(v1 != v2);
			// warning: this check could be costly
			Edge e; bool b;
			tie(e, b) = edge(v1, v2, g);
			getC(b ? e : add_edge(v1, v2, EdgeProperty(depth, inf, inf), g).first) = w + noise * real01();
		}

	}

	cerr << num_edges(g) << " edges, " << num_vertices(g) << " vertices" << endl;
}


void  
MsgSteinerGraph::output_tree(MsgSteinerGraph::TreeChecker const  & c)
{
	for (int i = 0, n = c.tree.size(); i < n; ++i) {
		if (c.tree[i] != -1) {
			Edge e; bool b;
			tie(e, b) = edge(i, c.tree[i], g);
			cout << g[i].name << " " << g[c.tree[i]].name 
				<< " " << g[i].c << " " << getC(e) << endl;
		}
	}
}


void  
MsgSteinerGraph::output_messages()
{
	vertex_iterator vit, vend;
	for (tie(vit, vend) = vertices(g); vit != vend; ++vit) {
		edge_iterator eit, eend;
		tie(eit, eend) = out_edges(*vit, g);

		Vertex v = *vit;
		cout << "----------------------------------" << endl 
			<< "Vertex " << g[*vit].name << endl << "c: " << g[v].c << endl 
			<< "G: " << g[v].G << endl;
		Mes m(depth);
		for (; eit != eend; ++eit) {
			Vertex w = target(*eit, g);
			cerr << g[v].name << " -> " <<  g[w].name << endl;
			getMes(*eit, m);
			cerr << m << endl;
		}
	}

}

MsgSteinerGraph::EdgeProperty::EdgeProperty(int depth, double cij = inf, double cji = inf)
	: ij(depth), ji(depth), cij(cij), cji(cji) 
{
	randomize(ij, mes_real01);
	randomize(ji, mes_real01);
	omp_init_lock(&lock);
}

