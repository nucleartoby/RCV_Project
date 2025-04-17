#include <iostream>
#include <cmath>

// Enum for option type
enum OptionType { CALL, PUT };

// Normal cumulative distribution function
double norm_cdf(double x) {
    return 0.5 * (1 + erf(x / sqrt(2)));
}

// Black-Scholes formula for option pricing
double black_scholes_price(
    OptionType type,    // Option type (CALL or PUT)
    double S,           // Current stock price
    double K,           // Strike price
    double r,           // Risk-free interest rate
    double sigma,       // Volatility
    double T            // Time to expiration in years
) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    if (type == CALL) {
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
    } else { // PUT
        return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

int main() {
    // Example parameters
    double S = 100.0;   // Stock price
    double K = 100.0;   // Strike price
    double r = 0.05;    // Risk-free rate (5%)
    double sigma = 0.2; // Volatility (20%)
    double T = 1.0;     // One year until expiration

    // Calculate and display results
    double call_price = black_scholes_price(CALL, S, K, r, sigma, T);
    double put_price = black_scholes_price(PUT, S, K, r, sigma, T);

    std::cout << "Black-Scholes Option Pricing Model" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Stock Price (S): $" << S << std::endl;
    std::cout << "Strike Price (K): $" << K << std::endl;
    std::cout << "Risk-free Rate (r): " << r * 100 << "%" << std::endl;
    std::cout << "Volatility (sigma): " << sigma * 100 << "%" << std::endl;
    std::cout << "Time to Expiration (T): " << T << " years" << std::endl;
    std::cout << std::endl;
    std::cout << "Call Option Price: $" << call_price << std::endl;
    std::cout << "Put Option Price: $" << put_price << std::endl;
    
    return 0;
}