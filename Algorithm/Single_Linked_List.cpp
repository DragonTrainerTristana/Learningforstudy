#include <iostream>

// DFS, BFS 복습하기
// Spanning Tree 알고리즘 짜기
// Recursion or Stack으로 짜기

// namespace
using namespace std;

class Node{

public:
	int data;
	Node* next;

};

// Summarize 

Node* insert(Node* head, int new_data);
Node* deleteNode(Node* head, int key);
void printList(Node* head);

int main() {
	
	Node* head = new Node(); // Initialization
	head = insert(head, 1);
	head = insert(head, 2);
	head = insert(head, 3);
	deleteNode(head, 2);
	printList(head);

	return 0;
}

Node* insert(Node* head, int new_data) {
	if (head == NULL) {
		cout << "EMPTY\n";
		return head;
	}
	else {
		Node* new_node = new Node();
		new_node->data = new_data;
		new_node->next = head->next;
		head->next = new_node;
		return head;
	}
}
Node* deleteNode(Node* head, int key) {
	if (head == NULL) {
		cout << "EMPTY\n";
	}
	else {
		Node* cursor = new Node();
		cursor = head;
		while (cursor->next->data != key) {
			cursor = cursor->next;
		}
		Node* temp = new Node();
		temp = cursor->next;
		cursor->next = cursor->next->next;
		delete temp;
		return head;
	}
}
void printList(Node* head) {
	//Initialize New Node for flicking through original node;
	Node* cursor = new Node();
	cursor = head->next;
	while (cursor != NULL) {
		cout << cursor->data << " ";
		cursor = cursor->next;
	}
}
