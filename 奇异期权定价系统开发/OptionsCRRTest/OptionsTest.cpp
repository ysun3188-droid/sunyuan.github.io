#include "BinomialPrice.h"
#include "PayOff.h"
#include <iostream>
#include <cmath>  // 添加cmath库以使用abs函数

using namespace std;

int main()
{
    Options opt(100.0, 100.0, 0.05, 0.30, 1, European, Call);
    // For an Asian (European Call) option, the payoff is the amount by which
    // the average stock price during the life time of the option exceeds the strike price.
    BinomialPrice p(&opt, 3, avgF, AsianPayOff);
    p.buildTree(0, p.steps);
    p.backwardEval(p.tree->root);
    cout << "European Call; 3-Step, avgF, AsianPayOff: " << p.tree->root->optionPrice << endl;
    cout << endl;
    p.printTree(p.tree->root);
    cout << endl;

    opt.c_p = Put;
    // For an Asian (European Put) option, the payoff is the amount by which
    // the strike price exceeds the average stock price during the life time of the option.
    p.buildTree(0, p.steps);
    p.backwardEval(p.tree->root);
    cout << "European Put; 3-Step, avgF, AsianPayOff: " << p.tree->root->optionPrice << endl;
    cout << endl;
    p.printTree(p.tree->root);
    cout << endl;

    opt.e_a = American;
    opt.c_p = Call;
    // For an Asian (American Call) option, if exercised at time t, the payoff is the amount by which
    // the average stock price between time 0 and t exceeds the strike price.
    p.buildTree(0, p.steps);
    p.backwardEval(p.tree->root);
    cout << "American Call; 3-Step, avgF, AsianPayOff: " << p.tree->root->optionPrice << endl;
    cout << endl;
    p.printTree(p.tree->root);
    cout << endl;

    opt.c_p = Put;
    // For an Asian (American Put) option, if exercised at time t, the payoff is the amount by which
    // the strike price exceeds the average stock price between time 0 and t.
    p.buildTree(0, p.steps);
    p.backwardEval(p.tree->root);
    cout << "American Put; 3-Step, avgF, AsianPayOff: " << p.tree->root->optionPrice << endl;
    cout << endl;
    p.printTree(p.tree->root);
    cout << endl;

    Options optLB(50, 50, 0.1, 0.4, 0.25, American, Put);
    // For an American lookback put option, if exercised at time t, the payoff is the amount by which
    // the maximum stock price between time 0 and t exceeds the current stock price.
    BinomialPrice pLB(&optLB, 3, maxF, LookbackPayOffPut);
    pLB.buildTree(0, pLB.steps);
    pLB.backwardEval(pLB.tree->root);
    cout << "American Put; 3-Step, maxF, LookbackPayOffPut: " << pLB.tree->root->optionPrice << endl;
    cout << endl;
    pLB.printTree(pLB.tree->root);
    cout << endl;

    // 测试欧式看涨期权
    Options europeanCallOption(100.0, 100.0, 0.05, 0.30, 1, European, Call);
    BinomialPrice europeanCallPricing(&europeanCallOption, 3, avgF, AsianPayOff);

    europeanCallPricing.buildTree(0, europeanCallPricing.steps);
    europeanCallPricing.backwardEval(europeanCallPricing.tree->root);
    double calculatedOptionPrice = europeanCallPricing.tree->root->optionPrice;
    cout << "Custom Test - European Call; 3-Step, avgF, AsianPayOff: " << calculatedOptionPrice << endl;

    // 验证期权价格是否符合预期
    const double expectedOptionPrice = 7.63;
    const double tolerance = 0.00001;
    if (abs(calculatedOptionPrice - expectedOptionPrice) < tolerance) {
        cout << "Custom Test Passed: Calculated price is close to expected price." << endl;
    } else {
        cout << "Custom Test Failed: Calculated price is not close to expected price." << endl;
    }

    cout << endl;
    europeanCallPricing.printTree(europeanCallPricing.tree->root);
    cout << endl;

    return 0;
}
