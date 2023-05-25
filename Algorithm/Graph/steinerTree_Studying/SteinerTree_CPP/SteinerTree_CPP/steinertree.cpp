#include <iostream>
#include <vector> // auto memory allocation
#include <cstdlib>

#define VERTICES 10 // V at G = (V,E)

using namespace std;

class heap {
public:
		int left, right, largest, temp;
		void max_heapify(int*, int, int);
		void heapsort(int*, int);
		void build_max_heap(int*, int);
		void display(int*, int, string);
};

class Graph {


};


int main() {

	return 0;
}

void heap::max_heapify(int* array, int i, int size) {}
void heap::heapsort(int* array, int size) {}
void heap::build_max_heap(int* array, int size) {}
void Graph::insertedge(int origin, int destination, int weight);
void Graph::printadjacencymatrix(int n);
int* Graph::dijkstra(int src, int dest, int cost, int par);
void Graph::steiner(int numterminal);
int* Graph::pathfind(vector<int> W, int v);