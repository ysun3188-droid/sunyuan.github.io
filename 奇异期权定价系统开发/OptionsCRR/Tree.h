#ifndef TREE_H
#define TREE_H

class Node
{
public:
    Node* leftChild;
    Node* rightChild;
    Node* parent;
    int level;

    Node() : leftChild(0), rightChild(0), parent(0), level(0) {
    }
};

// TreeNode class specialized from Node
class TreeNode : public Node
{
public:
    double sharePrice;   // Share Price at this node
    double pathData;     // Path Data (e.g., running average of share price)
    double optionPrice;  // Option Price calculated during backward evaluation

    TreeNode() : Node(), sharePrice(0.0), pathData(0.0), optionPrice(0.0) {
    }
};

class Tree
{
public:
    TreeNode* root;

    Tree() : root(0), nLevel(0) {};
    void setLevel(int n) { nLevel = n; }
    int getLevel() { return nLevel; }

private:
    int nLevel;
};

#endif
