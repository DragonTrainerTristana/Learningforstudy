#include <iostream>
#include <vector> // auto memory allocation
#include <cstdlib>
#include <time.h>
#include <ctime>

using namespace std;

class Node {
public:

	int number; // ID
	Node* link[4]; 
	float weight[4]; 
	int possibleNum;
		
};

Node* createNode(int numNode, Node* root);
Node* deleteNode(Node* vertex);

int main() {

	srand((unsigned int)time(NULL));

	// Variables
	int numNode;// number of Node
	int connectedNum; // random number of Connected Link between Nodes;
	int connectedNode;
	float randomfloat[4] = { };

	cin >> numNode;

	Node* Vertices = new Node[numNode]; // memory
	// Allocation of Link between Nodes at each Node

	// Initialization
	for (int i = 0; i < numNode; i++) {
		(Vertices + i)->number = NULL;
		(Vertices + i)->possibleNum = 2;
		for (int j = 0; j < 4; j++)(Vertices + i)->link[j] = NULL;
		for (int j = 0; j < 4; j++)(Vertices + i)->weight[j] = NULL;
	}

	for (int i = 0; i < numNode; i++) {
		
		(Vertices + i)->number = i + 1; // Node_ID
		connectedNum = ( rand() & 2 ); // 0 ~ 2 random number for generating new links
		
			
		randomfloat[0] = (double)rand()/RAND_MAX;
		

		// We have to connect links between nodes carefully.
		// if Node A is created, then A must be connected to every link. 
		// In my opinion, neighber number might be connected like, 1 - 2, 2 - 3 , 3 - 4 (fully connected of edge Node)
		
		// Shape of Double Linked List
		// each nodes' link array 0 ~ 1 is filled for neighber number connection
		if ((Vertices + i)->link[0] == NULL && (Vertices + i + 1)->link[0] == NULL) { 

			// Connection
			(Vertices + i)->link[0] = (Vertices + i + 1);
			(Vertices + i + 1)->link[0] = (Vertices + i);

			// Weight
			(Vertices + i)->weight[0] = randomfloat[0];
			(Vertices + i + 1)->weight[0] = randomfloat[0];

		}
		else if (i == numNode - 1) {

			//Connection
			(Vertices + i)->link[1] = (Vertices + 0);
			(Vertices + 0)->link[1] = (Vertices + i);

			//Weight
			(Vertices + i)->weight[1] = randomfloat[0];
			(Vertices + 0)->weight[1] = randomfloat[0];
		}
		else {
			
			//Connection
			(Vertices + i)->link[1] = (Vertices + i + 1);
			(Vertices + i + 1)->link[0] = (Vertices + i);

			//Weight
			(Vertices + i)->weight[1] = randomfloat[0];
			(Vertices + i + 1)->weight[0] = randomfloat[0];
		}

		// Link 2 ~ 3 

		// 여기에서 자기자신 제외, 중복 제외한 링크를 연결해줘야함

		if ((Vertices + i)->possibleNum > 0) {
		
			if ((Vertices + i)->possibleNum == 1 && connectedNum >= 1) {
				
			}
			else if((Vertices + i)->possibleNum == 2 && connectedNum >= 2){
				for (int j = 0; j < connectedNum; j++) {
				
				}
			}
			else {} // nothing

		}
		else {} // nothing


		
	}
	


	return 0;
}

Node* createNode(int numNode, Node* root) {



	return 0;
}

Node* deleteNode(Node* vertex) {
	return 0;
}