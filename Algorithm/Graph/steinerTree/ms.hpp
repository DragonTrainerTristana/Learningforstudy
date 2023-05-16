/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/

// 여기가 메인 프레임

#ifndef MS_H
#define MS_H

#include <boost/random.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <omp.h>

#include "mes.hpp" // 얘는 뭘까

extern boost::mt19937 gen;
extern boost::mt19937 mes_gen;


class MsgSteinerGraph {
public:
	MsgSteinerGraph(int depth) : 
		depth(depth), 
		rein(0), 
		rootid(-1),
		g(0),
		Mcache(omp_get_max_threads(), std::vector<std::pair<Mes,Mes> > ()), 
		oldcache(omp_get_max_threads(), Mes(depth)) {}

	typedef std::vector<std::pair<std::pair<unsigned, unsigned>, double> > EdgeList;
	MsgSteinerGraph(int depth, MsgSteinerGraph::EdgeList const & edges, std::vector<double> const & vertices, unsigned r) : 
		depth(depth), 
		rein(0), 
		rootid(r),
		g(vertices.size()),
		Mcache(omp_get_max_threads(), std::vector<std::pair<Mes,Mes> > ()), 
		oldcache(omp_get_max_threads(), Mes(depth))
		{ read_graph(edges, vertices, r); }

	//params
	int depth;
	double rein;

	struct TreeChecker {
		TreeChecker(MsgSteinerGraph const &);
		std::vector<int> tree;
		int unconnected;
		int dmax;
		int nnodes;

		double ecost;
		double vcost;
		double cost;
	};

	//member functions

	void read_graph(EdgeList const & edges, std::vector<double> const & vertices, unsigned r);
	void read_graph(std::istream & file, double noise);
	void output_tree(MsgSteinerGraph::TreeChecker const &);
	void output_messages();
	unsigned rootid;
	double iterate();
	MsgSteinerGraph::TreeChecker
	converge(int maxit, double tolerance, int decision, double beta);
	void chooseroot(int maxit, double tolerance, int decision, double beta);

private:
	struct EdgeProperty {
		EdgeProperty() : ij(0), ji(0), cij(-1), cji(-1) { i_assert(0); }
		EdgeProperty(int depth, double cij, double cji);
		EdgeProperty(EdgeProperty const & ep) : ij(ep.ij), ji(ep.ji), cij(ep.cij), cji(ep.cji) {
			omp_init_lock(&lock);
		}
		~EdgeProperty() { omp_destroy_lock(&lock); }
		Mes ij;
		Mes ji;
		mutable omp_lock_t lock;
		double cij;
		double cji;
	};


	struct VertexProperty  {
	       VertexProperty() : type(normal), G(0), c(0) {}
	       VertexProperty(std::string const & name) : name(name), type(normal), G(0),c(0){}
	       std::string name;
	       enum typeEnum {
		       root,
		       terminal,
		       normal
	       };
	       typeEnum type;
	       bool isroot() const { return type == root; }
	       double G;
	       double c;
	};

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                      VertexProperty, EdgeProperty> Graph;
	typedef boost::graph_traits<Graph>::edge_descriptor Edge;
	typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
	typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;
	typedef boost::graph_traits<Graph>::out_edge_iterator edge_iterator;
	typedef boost::graph_traits<Graph>::edge_iterator graph_edge_iterator;

	double update(Vertex i); 
	inline void getMes(Edge e, Mes &) const; 
	inline double & getC(Edge e); 
	inline double const & getC(Edge e) const; 
	inline void getBoth(Edge e, std::pair<Mes, Mes> &) const; 
	inline void getMesInv(Edge e, Mes &) const;
	inline void setMes(Edge e, Mes & out);
	double updateroot(Vertex i);
	std::pair<bool, Edge> marginal(Vertex i) const;
	int idx(std::string const & id);

	Graph g;
	std::vector<std::vector<std::pair<Mes, Mes> > > Mcache;
	std::vector<Mes> oldcache;
};





#endif
