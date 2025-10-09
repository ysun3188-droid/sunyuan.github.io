#include "BinomialPrice.h"
#include <math.h>
#include <iostream>
#include <iomanip> // 添加iomanip以支持setw

using namespace std;

BinomialPrice::BinomialPrice(Options* pOpt, int nSteps, PathFunc pathFunc, PayOffFunc payF)
    : option(pOpt), steps(nSteps), pathF(pathFunc), payOffF(payF)
{
    tree = new Tree();
    dt = option->maturity / nSteps;
    u = exp(option->vol * sqrt(dt));
    d = 1 / u;
    pu = (exp(option->rate * dt) - d) / (u - d);
    pd = 1.0 - pu;
    discount = exp(-option->rate * dt);
}

void BinomialPrice::buildTree(TreeNode* t, int nFromLeafNode) {
    if (t == nullptr) {
        // If tree is empty, create root node
        t = new TreeNode();
        t->sharePrice = option->spotPrice;  // Initial stock price
        t->pathData = pathF(t->sharePrice, t->sharePrice, 0); // Initial path data
        tree->root = t;
    }

    if (nFromLeafNode > 0) {
        // Create left child
        t->leftChild = new TreeNode();
        t->leftChild->parent = t;
        t->leftChild->level = t->level + 1;

        TreeNode* leftChildNode = static_cast<TreeNode*>(t->leftChild);
        leftChildNode->sharePrice = t->sharePrice * d;  // Down movement
        leftChildNode->pathData = pathF(leftChildNode->sharePrice, t->pathData, t->level + 1);

        // Recursively build left sub-tree
        buildTree(leftChildNode, nFromLeafNode - 1);

        // Create right child
        t->rightChild = new TreeNode();
        t->rightChild->parent = t;
        t->rightChild->level = t->level + 1;

        TreeNode* rightChildNode = static_cast<TreeNode*>(t->rightChild);
        rightChildNode->sharePrice = t->sharePrice * u;  // Up movement
        rightChildNode->pathData = pathF(rightChildNode->sharePrice, t->pathData, t->level + 1);

        // Recursively build right sub-tree
        buildTree(rightChildNode, nFromLeafNode - 1);
    }
}

void BinomialPrice::backwardEval(TreeNode* t) {
    if (t->leftChild == nullptr && t->rightChild == nullptr) {
        // Leaf node
        t->optionPrice = payOffF(t->pathData, t->sharePrice, option->strikePrice, option->c_p);
    } else {
        // Intermediate node
        backwardEval(static_cast<TreeNode*>(t->leftChild));
        backwardEval(static_cast<TreeNode*>(t->rightChild));

        double leftPrice = static_cast<TreeNode*>(t->leftChild)->optionPrice;
        double rightPrice = static_cast<TreeNode*>(t->rightChild)->optionPrice;

        t->optionPrice = discount * (pu * rightPrice + pd * leftPrice);
    }
}


void BinomialPrice::printTree(TreeNode* t, int indent) {
    if (t != nullptr) {
        if (t->rightChild) {
            printTree(static_cast<TreeNode*>(t->rightChild), indent + 4);
        }
        if (indent) {
            cout << setw(indent) << ' ';
        }
        if (t->rightChild) cout << " /\n" << setw(indent) << ' ';
        cout << t->sharePrice << " " << t->pathData << " " << t->optionPrice << "\n ";
        if (t->leftChild) {
            cout << setw(indent) << ' ' << " \\\n";
            printTree(static_cast<TreeNode*>(t->leftChild), indent + 4);
        }
    }
}
