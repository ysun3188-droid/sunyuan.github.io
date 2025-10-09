#include "stdafx.h"
#include "CppUnitTest.h"
#include "BinomialPrice.h"
#include "PayOff.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace OptionsCRRUnitTest
{		
	TEST_CLASS(BinomialCRR)
	{
	public:
		Options* opt;
		BinomialPrice* p;

		TEST_METHOD_INITIALIZE(ALL)
		{
			opt = new Options(100.0, 100.0, 0.05, 0.30, 1, European, Call);
			p = new BinomialPrice(opt, 3, avgF, AsianPayOff);
		}

		TEST_METHOD_CLEANUP(ALL_CLEAN)
		{
			delete opt;
			delete p;
		}

		
		TEST_METHOD(AsianEuropeanCall)
		{
			// TODO: Your test code here
			p->buildTree(0, p->steps);
			p->backwardEval(p->tree->root);
			Assert::AreEqual(7.62876, p->tree->root->optionPrice, 0.00001);
		}

		TEST_METHOD(AsianEuropeanPut)
		{
			opt->c_p = Put;
			p->buildTree(0, p->steps);
			p->backwardEval(p->tree->root);
			Assert::AreEqual(5.20377, p->tree->root->optionPrice, 0.00001);
		}

		TEST_METHOD(AsianAmericanCall)
		{
			opt->e_a = American;
			opt->c_p = Call;
			p->buildTree(0, p->steps);
			p->backwardEval(p->tree->root);
			Assert::AreEqual(7.92877, p->tree->root->optionPrice, 0.00001);
		}

		TEST_METHOD(AsianAmericanPut)
		{
			opt->e_a = American;
			opt->c_p = Put;
			p->buildTree(0, p->steps);
			p->backwardEval(p->tree->root);
			Assert::AreEqual(5.5497, p->tree->root->optionPrice, 0.00001);
		}

		TEST_METHOD(AmericanLookbackPut)
		{
			Options optLB(50, 50, 0.1, 0.4, 0.25, American, Put);
			BinomialPrice pLB(&optLB, 3, maxF, LookbackPayOffPut);
			pLB.buildTree(0, pLB.steps);
			pLB.backwardEval(pLB.tree->root);
			Assert::AreEqual(5.47018, pLB.tree->root->optionPrice, 0.00001);
		}
	};
}