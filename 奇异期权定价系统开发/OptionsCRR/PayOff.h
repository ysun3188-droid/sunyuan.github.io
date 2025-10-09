#ifndef PAYOFF_H
#define PAYOFF_H

#include "Options.h"

double maxF(double maxSoFar, int currentLevel, double currentPrice)
{
	return maxSoFar > currentPrice ? maxSoFar : currentPrice;
}

double avgF(double avgSoFar, int currentLevel, double currentPrice)
// Note: currentLevel starts at 0
{
	return currentLevel == 0 ?
	currentPrice : ((avgSoFar * currentLevel) + currentPrice) / (currentLevel + 1);
}

double AsianPayOff(double pathData, double currentPrice, double strikePrice, ExerciseType cp)
{
	return MAX(OPTION_MULTIPLIER(cp)*(pathData - strikePrice), 0.0);
}

double LookbackPayOffPut(double pathData, double currentPrice, double strikePrice, ExerciseType cp)
{
	return MAX((pathData - currentPrice), 0.0);
}

#endif