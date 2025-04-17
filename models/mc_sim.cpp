#include <iostream>
#include <cmath>
#include <random>
#include <vector>

int main() {
    double S0 = 100.0;
    double k = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double t = 1.0;
    int num_simulations = 1000000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    double payoff_sum = 0.0;

    for(int i = 0; i < num_simulations; ++i) {
        double z = d(gen);
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * t + sigma * sqrt(t) * z);
        double payoff = std::max(0.0, ST - k);
        payoff_sum += payoff;
    }

    double option_price = exp(-r * t) * payoff_sum / num_simulations;

    std::cout << "Monte Carlo European Call Option Price: " << option_price << std::endl;

    return 0;
}