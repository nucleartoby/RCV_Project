#include <iostream>
#include "black_scholes.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

double norm_cdf(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double black_scholes_price(OptionType type, double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    if (type == Call)
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    else
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double option_delta(OptionType type, double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return (type == Call) ? norm_cdf(d1) : norm_cdf(d1) - 1.0;
}

double implied_volatility(OptionType type, double market_price, double S, double K, double T, double r, double initial_guess, double tolerance, int max_iterations) {
    double sigma = initial_guess;
    for (int i = 0; i < max_iterations; ++i) {
        double price = black_scholes_price(type, S, K, T, r, sigma);
        double vega = S * std::sqrt(T) * std::exp(-0.5 * std::pow((std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T)), 2)) / std::sqrt(2 * M_PI);

        double diff = price - market_price;
        if (std::fabs(diff) < tolerance) break;

        sigma -= diff / vega;
    }

    return sigma;
}
