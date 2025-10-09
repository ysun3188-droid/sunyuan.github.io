#ifndef BINOMIALPRICE_H
#define BINOMIALPRICE_H

#include "Options.h"
#include "Tree.h"

typedef double(*PathFunc) (double, int, double);
typedef double(*PayOffFunc) (double, double, double, ExerciseType);

class BinomialPrice
{
public:
	Options*		option;
	int				steps;
	Tree*			tree;


	BinomialPrice(Options* pOpt, int nSteps, PathFunc pathFunc, PayOffFunc payF);

	void buildTree(TreeNode* node, int nFromLeafNode);
	void backwardEval(TreeNode*);
	void printTree(TreeNode* node, int indent = 0);

private:
	double			dt;		// delta time
	double			u;		// up factor
	double			d;		// down factor = 1/u
	double			pu;		// up probability
	double			pd;		// down probability = 1 - pu
	double			discount;
	PathFunc		pathF;	// path dependent function
	PayOffFunc		payOffF;// payoff function
};

#endif