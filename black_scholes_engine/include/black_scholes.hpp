#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

enum OptionType {Call, Put }; // Option type enumeration

double norm_cdf(double x); // Standard distribution function

doulbe black_scholes_price( // Black-Scholes formula for option pricing
    OptionType type,
    double S, // Current stock price
    double K, // Strike price
    double T, // Time to expiration in years
    double r // Risk free interest rate
    double sigma // Volatility of underlying asset
);

double option_delta( // Delta of option
    OptionType type,
    double S, double K, double T, double r, double sigma // Parameters for option pricing

);

double implied_volatility( // Calculate volatility
    OptionType type,
    double market_price, double S, double K, double T, double r
    double intitial_guess = 0.2, // Initial volatility guess
    double tolerance = 1e-5, // Tolerance
    int max_iterations = 1000 // Max iterations for convergence

);

#endif