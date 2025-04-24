#include <iostream>
#include <cmath>
#include <random>
#include <vector>

int main() {
    double S0 = 5.310; // Initial price of the stock.
    double k = 6.30; // Strike price of the option.
    double r = 0.04; // Risk free interest rate annually.
    double sigma = 0.516; // Volatility of the underlying asset.
    double t = 0.1; // Time to maturity of the option (years).
    int num_simulations = 1000000; // Number of Monte Carlo simulations.

    std::random_device rd; // RNG device.
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    double payoff_sum = 0.0; // Initial sum of the payoff.

    for(int i = 0; i < num_simulations; ++i) { // Loop for Monte Carlo simulations.
        double z = d(gen);
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * t + sigma * sqrt(t) * z); // Simulated stock price at maturity.
        double payoff = std::max(0.0, ST - k); // Payoff for a European call option.
        payoff_sum += payoff;
    }

    double option_price = exp(-r * t) * payoff_sum / num_simulations; // Discounted average payoff.

    std::cout << "Monte Carlo Call Option Price: " << option_price << std::endl; //Output.

    return 0;
}