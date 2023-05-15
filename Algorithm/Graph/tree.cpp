#include <iostream>
#include <queue>

using namespace std;

class NODE{};
NODE* initialize(NODE* root);
NODE* insertNODE(NODE* root, int num_Insert, char dir);
NODE* deleteNODE(NODE* root,int num_Insert, char dir);
void printTree(NODE* root);

int main() {

	NODE* root = new NODE();
	root = initialize(root);

	//Variables
	int num_Option;
	int num_Insert;
	char dir;

	cout << "Input String data of root NODE : ";
	cin >> root->name;
	cout << "your root name is " << root->name << endl;

	for (;;) {
		cout << "1: Insert 2: Delete 3: Print 4: exit\n";
		cin >> num_Option;
		
		if (num_Option == 1) {
			cout << "Option 1\n";
			cin >> num_Insert;
			cin >> dir;
			root = insertNODE(root, num_Insert, dir);
		}
		if (num_Option == 2) {
			cin >> num_Insert;
			cin >> dir;
			root = deleteNODE(root, num_Insert, dir);
		}
		if (num_Option == 3) {
			printTree(root);
		}
		if (num_Option == 4)break;
		cin >> dir;
	}


}

NODE* initialize(NODE* root){
	root->parentNODE = NULL;
	root->leftChild = NULL;
	root->rightChild = NULL;
}
NODE* insertNODE(NODE* root, int num_Insert, char dir){
	if (root == NULL) {
		cout << "ERROR\n";
		return 0;
	}
	// just only root node is exist
	if (root->leftChild == NULL && root->rightChild == NULL) {
		NODE* temp = new NODE();
		temp = initialize(temp);
		if (dir == 'l' || dir == 'L') {
			root->leftChild = temp;
			temp->parentNODE = root;
		}
		else if (dir == 'r' || dir == 'R') {
			root->rightChild = temp;
			temp->parentNODE = root;
		}
	}
	// the other cases
	else {
		NODE* cursor = new NODE();
		cursor = root;
		if (dir == 'l' || dir == 'L') {

		}
		else if (dir == 'r' || dir == 'R') {

		}
	}
}
NODE* deleteNODE(NODE* root, int num_Insert, char dir){}
void printTree(NODE* root){}

class NODE {
public:

	string name;
	int num;

	NODE* parentNODE;
	NODE* leftChild;
	NODE* rightChild;
};