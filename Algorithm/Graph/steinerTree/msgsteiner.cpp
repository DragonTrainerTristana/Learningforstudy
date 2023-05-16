/*
   Copyright 2012-2016 Alfredo Braunstein

   This file is part of MSGSTEINER (Max Sum for generalized steiner problems on graphs).

   MSGSTEINER is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

*/

// 최종 돌려야 하는 프로그램

#include "ms.hpp"

#include <boost/program_options.hpp>
#include <string>
#include <omp.h>
#include "assertion.hpp"

using namespace boost;
using namespace std;

namespace po = boost::program_options;

void bail(char const * exp, char const * file , char const * fun, int line)
{

	cerr << "assert (" << exp << ")" << "filed in function " << fun << " in file  \"" << file << "\" +" << line << endl;
	assert(0);a
}



po::variables_map parse_command_line(int ac, char ** av)
{
	po::options_description desc("Usage: " + string(av[0]) + " <option> ... \n\twhere <option> is one or more of");
	desc.add_options()
		("help", "produce help message")
		("tree,o", "outputs final tree to std output")
		("messages,M", "output messages on convergence")
		("threads,j", po::value<unsigned>(), "sets number of threads")


		("seed,s", po::value<unsigned>(), "sets instance seed")
		("mseed,z", po::value<unsigned>(), "sets messages seed")


		("depth,d", po::value<int>()->default_value(10), "set maximum depth")
		("maxit,t", po::value<int>()->default_value(100000), "set maximum number of iterations")
		("tolerance,e", po::value<double>()->default_value(1e-5), "set convergence tolerance")
		("noise,r", po::value<double>()->default_value(1e-6), "set random factor")
		("rein,g", po::value<double>()->default_value(0), "sets reinforcement parameter rein")
		("decision,y", po::value<int>()->default_value(10), "program converges after this # repeats of the decision variables");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("threads"))
		omp_set_num_threads(vm["threads"].as<unsigned>());

	cerr << "num threads: " << omp_get_max_threads() << endl;




	if (vm.count("seed")) {
		unsigned s = vm["seed"].as<unsigned>();
		gen.seed(s);
	}
	if (vm.count("mseed")) {
		unsigned s = vm["mseed"].as<unsigned>();
		mes_gen.seed(s);
	}
	
	if (vm.count("help")) {
		cerr << desc << "\n";
		exit(1);
	}

	return vm;
}



// 분석 

int main(int ac, char** av)
{

	// 문법적인 요소
	/*
	ex)
	ClassName :: 
	NameSpaceName :: 
	ios_base ::
	<- "범위 지정 연산자"
	*/
	cout.setf(ios_base::fixed, ios_base::floatfield); // ios_base :: fixed, floatfield
	po::variables_map vm = parse_command_line(ac, av); // 

	MsgSteinerGraph M(vm["depth"].as<int>());


	double noise = vm["noise"].as<double>();
	int decision = vm["decision"].as<int>();
	double tolerance = vm["tolerance"].as<double>();
	int maxit = vm["maxit"].as<int>();
	double beta = vm["rein"].as<double>();
	
	M.read_graph(cin, noise);

	if (M.rootid == unsigned(-1))
		M.chooseroot(maxit, tolerance, 100, beta);
	
	MsgSteinerGraph::TreeChecker c = M.converge(maxit, tolerance, decision, beta);


	if (vm.count("messages")) 
		M.output_messages();

	if (vm.count("tree"))
		M.output_tree(c);
		
	return 0;
}


