#include "black_scholes.hpp"
#include <iostream>

int main() {
    double S = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double sigma = 0.2;

    double price = black_scholes_price(Call, S, K, T, r, sigma);
    double delta = option_delta(Call, S, K, T, r, sigma);
    double iv = implied_volatility(Call, price, S, K, T, r);

    std::cout << "Option Price: " << price << "\n";
    std::cout << "Delta: " << price << "\n";
    std::cout << "Implied Vol: " << price << "\n";

    return 0;
}