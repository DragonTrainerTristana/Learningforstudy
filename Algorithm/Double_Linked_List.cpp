#include <iostream>

using namespace std;

class NODE {

public:
	int data;
	NODE* left;
	NODE* right;
};

NODE* insertNode(NODE* head, int key, char dir);
NODE* deleteNode(NODE* head, int key, char dir);
NODE* initialize(NODE* head);
void printList(NODE* head);

int main() {
	
	NODE* head = new NODE();
	head = initialize(head);
	int num_Option;
	int num_Insert;
	char str;
	for (;;) {
		cout << "1: Insert 2: Delete 3: Print 4: exit\n";
		cin >> num_Option;

		if (num_Option == 1) {
			cout << "Option 1\n";
			cin >> num_Insert;
			cin >> str;
			head = insertNode(head, num_Insert, str);
		}
		if (num_Option == 2) {
			cin >> num_Insert;
			cin >> str;
			head = deleteNode(head, num_Insert, str);
		}
		if (num_Option == 3) {
			printList(head);
		}
		if (num_Option == 4)break;
	}


	return 0;
}

NODE* initialize(NODE* head) {
	head->left = NULL;
	head->right = NULL;
	return head;
}

NODE* insertNode(NODE* head, int key, char dir) {

	NODE* newNode = new NODE();

	if (dir == 'l' || dir == 'L') { 
		if (head == NULL) { 
			cout << "EMPTY\n";

			return head;
		}

		newNode->data = key;
		newNode->left = head->left;
		newNode->right = head;
		head->left = newNode;

		return head;

	}
	else if (dir == 'r' || dir == 'R') {
		if (head == NULL) {
			cout << "EMPTY\n";

			return head;
		}

		newNode->data = key;
		newNode->right = head->right;
		head->right = newNode;

		return head;
	}

}

NODE* deleteNode(NODE* head, int key, char dir) {

	if (head == NULL) { return 0; }
	NODE* cursor = new NODE();
	NODE* temp = new NODE();
	cursor = head;
	
	if (dir == 'r' || dir == 'R') {
		while (cursor->right->data != key) {
			cursor = cursor->right;
		}
		temp = cursor->right;
		cursor->right->right->left = cursor;
		cursor->right = cursor->right->right;
		delete temp;
		return head;
	}
	
	if (dir == 'l' || dir == 'L') {
		while (cursor->left->data != key) {
			cursor = cursor->left;
		}
		temp = cursor->left;
		cursor->left->left->right = cursor;
		cursor->left = cursor->left->left;
		delete temp;
		return head;
	}

	return head;
}
void printList(NODE* head) {
	char dir;
	NODE* cursor = head;
	cout << "Left or Right?";
	cin >> dir;

	if (dir == 'l' || dir == 'L') {
		while (cursor->left != NULL) {
			cout << cursor->left->data << " ";
			cursor = cursor->left;
		}
	}
	else if (dir == 'r' || dir == 'R') {
		while (cursor->right != NULL) {
			cout << cursor->right->data << " ";
			cursor = cursor->right;
		}
	}
	cout << "\n\n";
}