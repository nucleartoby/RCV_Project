#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

enum OptionType {Call, Put };

double norm_cdf(double x);

doulbe black_scholes_price(
    OptionType type,
    double S,
    double K,
    double T,
    double r
    double sigma
);

double option_delta(
    OptionType type,
    double S, double K, double T, double r, double sigma

);

double implied_volatility(
    OptionType type,
    double market_price, double S, double K, double T, double r
    double intitial_guess = 0.2,
    double tolerance = 1e-5,
    int max_iterations = 1000

);

#endif