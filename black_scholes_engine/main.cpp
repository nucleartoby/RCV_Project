#include "black_scholes.hpp"
#include <iostream>
#include <cmath>

int main() {
    double S = 100.0; // Current stock price
    double K = 100.0; // Strike price
    double T = 1.0; // Expiration time (years)
    double r = 0.05; // Risk free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    double price = black_scholes_price(Call, S, K, T, r, sigma);
    double delta = option_delta(Call, S, K, T, r, sigma);
    double iv = implied_volatility(Call, price, S, K, T, r);

    std::cout << "Option Price: " << price << "\n";
    std::cout << "Delta: " << delta << "\n";
    std::cout << "Implied Vol: " << iv << "\n";

    return 0;
}