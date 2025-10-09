#ifndef ASIANOPTION_H
#define ASIANOPTION_H

enum	OptionType		{ European, American };
enum	ExerciseType	{ Call, Put };

#define OPTION_MULTIPLIER(x)	((x) == Call ? 1.0 : -1.0)
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

class Options
{
public:
	double			spotPrice;
	double			strikePrice;
	double			rate;
	double			vol;
	double			maturity;
	OptionType		e_a;
	ExerciseType	c_p;

	Options(double S, double X, double r, double v, double T,
		OptionType eORa, ExerciseType cORp)
		:spotPrice(S), strikePrice(X), rate(r), vol(v), maturity(T),
		e_a(eORa), c_p(cORp) {};
};

#endif
