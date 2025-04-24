#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <Eigen/Dense>

class ArimaGarchModel {
private:
    int p, d, q;      // ARIMA orders.
    int m, s;         // GARCH orders.
    
    std::vector<double> ar_params;  // AR parameters. (phi)
    std::vector<double> ma_params;  // MA parameters. (theta)
    double c;
    
    double omega;
    std::vector<double> alpha;      // ARCH parameters.
    std::vector<double> beta;       // GARCH parameters.
    
    std::vector<double> original_data; // Containers for the data.
    std::vector<double> differenced_data;
    std::vector<double> residuals;
    std::vector<double> conditional_variances;
    
    double learning_rate; // Optimisation settings (DONT CHANGE)
    int max_iterations;
    double convergence_threshold;
    
    std::mt19937 generator; // RNG for GARCH.
    std::normal_distribution<double> normal_dist; // Normal distribution for GARCH.
    
    std::vector<double> difference(const std::vector<double>& data, int order); // Helper method for differencing.
    std::vector<double> inverse_difference(const std::vector<double>& differenced, const std::vector<double>& original, int order); // Inverse.
    double calculate_log_likelihood(); // Log likelihood calculation.
    void update_residuals(); // Update residuals.
    void update_conditional_variances(); // Update conditional variances.
    
public:
    ArimaGarchModel(int p, int d, int q, int m, int s) // Constructor.
        : p(p), d(d), q(q), m(m), s(s), c(0), omega(0.01),
          learning_rate(0.01), max_iterations(1000), convergence_threshold(1e-6),
          generator(std::random_device{}()), normal_dist(0.0, 1.0) {
        
        ar_params.resize(p, 0.1); // ARIMA params .
        ma_params.resize(q, 0.1);
        
        alpha.resize(m, 0.1); // GARCH params.
        beta.resize(s, 0.8);
    }
    
    void set_data(const std::vector<double>& data) { // Set data for model.
        original_data = data;
        if (d > 0) {
            differenced_data = difference(data, d);
        } else {
            differenced_data = data;
        }
        
        residuals.resize(differenced_data.size(), 0.0); // Initialise residuals.
        conditional_variances.resize(differenced_data.size(), 0.0);
        
        double sum = std::accumulate(differenced_data.begin(), differenced_data.end(), 0.0); // Set initial conditional variance to sample variance.
        double mean = sum / differenced_data.size();
        double sq_sum = std::inner_product(differenced_data.begin(), differenced_data.end(), 
                                          differenced_data.begin(), 0.0);
        double variance = sq_sum / differenced_data.size() - mean * mean;
        
        std::fill(conditional_variances.begin(), conditional_variances.end(), variance);
    }
    
    void train() { // Training model.
        update_residuals(); // Initial calculation of residuals and conditional variances.
        update_conditional_variances();
        
        double prev_likelihood = calculate_log_likelihood();
        double current_likelihood;
        
        for (int iter = 0; iter < max_iterations; ++iter) { // Gradient descent optimisation.
            optimize_arima_params();
            
            update_residuals();
            
            optimize_garch_params();
            
            update_conditional_variances();
            
            current_likelihood = calculate_log_likelihood(); // Check convergence.
            if (std::abs(current_likelihood - prev_likelihood) < convergence_threshold) {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
            
            if (iter % 10 == 0) {
                std::cout << "Iteration " << iter << ", Log-likelihood: " << current_likelihood << std::endl;
            }
            
            prev_likelihood = current_likelihood;
        }
        
        print_parameters();
    }
    
    void optimize_arima_params() { // Optimise ARIMA params.
        std::vector<double> ar_gradients(p, 0.0); // Temp storage for gradients.
        std::vector<double> ma_gradients(q, 0.0);
        double c_gradient = 0.0;
        
        for (size_t t = std::max(p, q); t < differenced_data.size(); ++t) { // Calculate gradients for each param. (Important)
            double error = residuals[t];
            double h_t = conditional_variances[t];
            
            c_gradient += error / h_t; // Gradient for constant term.
            
            for (int i = 0; i < p; ++i) { // Gradients for AR params.
                if (t > i) {
                    ar_gradients[i] += error * differenced_data[t - i - 1] / h_t;
                }
            }
            
            for (int j = 0; j < q; ++j) { // Gradients for MA params.
                if (t > j) {
                    ma_gradients[j] += error * residuals[t - j - 1] / h_t;
                }
            }
        }
        
        c -= learning_rate * c_gradient; // Update params with constraints.
        
        for (int i = 0; i < p; ++i) {
            ar_params[i] -= learning_rate * ar_gradients[i];
        }
        
        for (int j = 0; j < q; ++j) {
            ma_params[j] -= learning_rate * ma_gradients[j];
        }
    }

    void optimize_garch_params() { // Optimise GARCH params.
        double omega_gradient = 0.0;
        std::vector<double> alpha_gradients(m, 0.0);
        std::vector<double> beta_gradients(s, 0.0);
        
        for (size_t t = std::max(m, s); t < differenced_data.size(); ++t) { // Calculate gradients for each param. (Important)
            double error_sq = residuals[t] * residuals[t];
            double h_t = conditional_variances[t];
            
            omega_gradient += (error_sq / (h_t * h_t) - 1.0 / h_t); // Gradient for omega.
            
            for (int i = 0; i < m; ++i) { // Gradients for ARCH params.
                if (t > i) {
                    alpha_gradients[i] += (error_sq / (h_t * h_t) - 1.0 / h_t) * 
                                         (residuals[t - i - 1] * residuals[t - i - 1]);
                }
            }
            
            for (int j = 0; j < s; ++j) { // Gradients for GARCH params.
                if (t > j) {
                    beta_gradients[j] += (error_sq / (h_t * h_t) - 1.0 / h_t) * 
                                        conditional_variances[t - j - 1];
                }
            }
        }

        omega = std::max(0.001, omega - learning_rate * omega_gradient); // Update params with constraints for stability.
        
        for (int i = 0; i < m; ++i) {
            alpha[i] = std::max(0.001, std::min(0.999, alpha[i] - learning_rate * alpha_gradients[i]));
        }
        
        for (int j = 0; j < s; ++j) {
            beta[j] = std::max(0.001, std::min(0.999, beta[j] - learning_rate * beta_gradients[j]));
        }

        double sum = std::accumulate(alpha.begin(), alpha.end(), 0.0) + // Make sure sum of ARCH and GARCH params is less than 1. (Stationarity)
                     std::accumulate(beta.begin(), beta.end(), 0.0);
        
        if (sum >= 0.999) { // Rescale.
            double scale = 0.99 / sum;
            for (double& a : alpha) a *= scale;
            for (double& b : beta) b *= scale;
        }
    }
    
    void update_residuals() { // Update residuals.
        for (size_t t = 0; t < differenced_data.size(); ++t) {
            double ar_component = 0.0;
            for (int i = 0; i < p; ++i) {
                if (t > i) {
                    ar_component += ar_params[i] * differenced_data[t - i - 1];
                }
            }
            
            double ma_component = 0.0;
            for (int j = 0; j < q; ++j) {
                if (t > j) {
                    ma_component += ma_params[j] * residuals[t - j - 1];
                }
            }
            
            double fitted = c + ar_component + ma_component; // Calculate fitted value.
            residuals[t] = differenced_data[t] - fitted; // Calculate residual.
        }
    }

    void update_conditional_variances() { // Update conditional variances.
        for (size_t t = 0; t < differenced_data.size(); ++t) {
            double h_t = omega;

            for (int i = 0; i < m; ++i) {
                if (t > i) {
                    h_t += alpha[i] * (residuals[t - i - 1] * residuals[t - i - 1]);
                }
            }

            for (int j = 0; j < s; ++j) {
                if (t > j) {
                    h_t += beta[j] * conditional_variances[t - j - 1];
                }
            }

            conditional_variances[t] = std::max(h_t, 0.000001); // Ensure variance is positive.
        }
    }

    double calculate_log_likelihood() { // Calculates the Log Likelihood.
        double log_likelihood = 0.0;
        
        for (size_t t = std::max(p, q); t < differenced_data.size(); ++t) {
            double h_t = conditional_variances[t];
            double error = residuals[t];

            log_likelihood += -0.5 * std::log(2.0 * M_PI) - 0.5 * std::log(h_t) - 
                              0.5 * (error * error) / h_t;
        }
        
        return log_likelihood;
    }

    std::vector<std::pair<double, double>> forecast(int horizon) { // Forecast future values.
        std::vector<std::pair<double, double>> forecasts;

        std::vector<double> extended_data = original_data; //  Create extended data vectors.
        std::vector<double> extended_diff = differenced_data;
        std::vector<double> extended_residuals = residuals;
        std::vector<double> extended_variances = conditional_variances;
        extended_data.resize(original_data.size() + horizon); // Extend vectors for forecasts.
        extended_diff.resize(differenced_data.size() + horizon);
        extended_residuals.resize(residuals.size() + horizon);
        extended_variances.resize(conditional_variances.size() + horizon);

        for (int h = 1; h <= horizon; ++h) { // Generate forecasts.
            size_t t = original_data.size() - 1 + h;

            double h_t = omega; // Forecast the conditional variance.
            for (int i = 0; i < m; ++i) {
                size_t idx = t - i - 1;
                if (idx < extended_residuals.size()) {
                    h_t += alpha[i] * (extended_residuals[idx] * extended_residuals[idx]);
                }
            }
            
            for (int j = 0; j < s; ++j) {
                size_t idx = t - j - 1;
                if (idx < extended_variances.size()) {
                    h_t += beta[j] * extended_variances[idx];
                }
            }

            extended_variances[t] = std::max(h_t, 0.000001); // Make sure variance is positive.

            double ar_component = 0.0; // Forecast the mean for ARIMA.
            for (int i = 0; i < p; ++i) {
                size_t idx = t - i - 1;
                if (idx < extended_diff.size()) {
                    ar_component += ar_params[i] * extended_diff[idx];
                }
            }
            
            double ma_component = 0.0;
            for (int j = 0; j < q; ++j) {
                size_t idx = t - j - 1;
                if (idx < extended_residuals.size()) {
                    ma_component += ma_params[j] * extended_residuals[idx];
                }
            }

            double forecast_mean = c + ar_component + ma_component; // Calculate point forecast (mean).
            extended_diff[t] = forecast_mean;

            extended_residuals[t] = 0.0;

            if (d > 0) {
                extended_data[t] = extended_data[t-1] + extended_diff[t];
            } else {
                extended_data[t] = extended_diff[t];
            }
            
            // Calculate prediction interval (we use 1.96 for 95% confidence)
            double std_error = std::sqrt(extended_variances[t]); // Calculate prediction interval (used 1.96 for 95% confidence).
            double lower_bound = extended_data[t] - 1.96 * std_error;
            double upper_bound = extended_data[t] + 1.96 * std_error;

            forecasts.push_back({extended_data[t], extended_variances[t]});
        }
        
        return forecasts;
    }

    std::vector<std::vector<double>> simulate_paths(int horizon, int num_paths) { // Simulate future paths.
        std::vector<std::vector<double>> paths(num_paths, std::vector<double>(horizon));
        
        for (int path = 0; path < num_paths; ++path) { // For each path, generate a new set of random shocks.
            std::vector<double> sim_data = original_data;
            std::vector<double> sim_diff = differenced_data;
            std::vector<double> sim_residuals = residuals;
            std::vector<double> sim_variances = conditional_variances;

            sim_data.resize(original_data.size() + horizon);
            sim_diff.resize(differenced_data.size() + horizon);
            sim_residuals.resize(residuals.size() + horizon);
            sim_variances.resize(conditional_variances.size() + horizon);

            for (int h = 1; h <= horizon; ++h) { // Generate simulations.
                size_t t = original_data.size() - 1 + h;

                double h_t = omega; // Forecast the conditional variance.
                for (int i = 0; i < m; ++i) {
                    size_t idx = t - i - 1;
                    if (idx < sim_residuals.size()) {
                        h_t += alpha[i] * (sim_residuals[idx] * sim_residuals[idx]);
                    }
                }
                
                for (int j = 0; j < s; ++j) {
                    size_t idx = t - j - 1;
                    if (idx < sim_variances.size()) {
                        h_t += beta[j] * sim_variances[idx];
                    }
                }
                
                sim_variances[t] = std::max(h_t, 0.000001);

                double ar_component = 0.0;
                for (int i = 0; i < p; ++i) {
                    size_t idx = t - i - 1;
                    if (idx < sim_diff.size()) {
                        ar_component += ar_params[i] * sim_diff[idx];
                    }
                }
                
                double ma_component = 0.0;
                for (int j = 0; j < q; ++j) {
                    size_t idx = t - j - 1;
                    if (idx < sim_residuals.size()) {
                        ma_component += ma_params[j] * sim_residuals[idx];
                    }
                }

                double forecast_mean = c + ar_component + ma_component;

                double shock = std::sqrt(sim_variances[t]) * normal_dist(generator);
                sim_residuals[t] = shock;

                sim_diff[t] = forecast_mean + shock;

                if (d > 0) {
                    sim_data[t] = sim_data[t-1] + sim_diff[t];
                } else {
                    sim_data[t] = sim_diff[t];
                }
                
                paths[path][h-1] = sim_data[t];
            }
        }
        
        return paths;
    }
    
    // Calculate Value at Risk (VaR)
    double calculate_var(const std::vector<std::vector<double>>& paths, int horizon, double confidence_level) { // Calculate VaR (Value at Risk).
        std::vector<double> horizon_returns;
        for (const auto& path : paths) {
            horizon_returns.push_back(path[horizon-1]);
        }

        std::sort(horizon_returns.begin(), horizon_returns.end()); // Sort returns.

        int var_index = static_cast<int>(horizon_returns.size() * (1.0 - confidence_level)); // Calculate VaR index.

        return horizon_returns[var_index];
    }

    double calculate_es(const std::vector<std::vector<double>>& paths, int horizon, double confidence_level) { // Calculate ES (Expected Shortfall)/
        std::vector<double> horizon_returns;
        for (const auto& path : paths) {
            horizon_returns.push_back(path[horizon-1]);
        }

        std::sort(horizon_returns.begin(), horizon_returns.end());

        int var_index = static_cast<int>(horizon_returns.size() * (1.0 - confidence_level));

        double es_sum = 0.0;
        for (int i = 0; i < var_index; ++i) {
            es_sum += horizon_returns[i];
        }

        return es_sum / var_index;
    }

    void print_parameters() {
        std::cout << "===== ARIMA(" << p << "," << d << "," << q << ")-GARCH(" << m << "," << s << ") Parameters =====" << std::endl; // Print model params.
        
        std::cout << "ARIMA Constant (c): " << c << std::endl;
        
        std::cout << "AR Parameters (phi):" << std::endl;
        for (int i = 0; i < p; ++i) {
            std::cout << "  phi_" << i+1 << ": " << ar_params[i] << std::endl;
        }
        
        std::cout << "MA Parameters (theta):" << std::endl;
        for (int i = 0; i < q; ++i) {
            std::cout << "  theta_" << i+1 << ": " << ma_params[i] << std::endl;
        }
        
        std::cout << "GARCH Parameters:" << std::endl;
        std::cout << "  omega: " << omega << std::endl;
        
        for (int i = 0; i < m; ++i) {
            std::cout << "  alpha_" << i+1 << ": " << alpha[i] << std::endl;
        }
        
        for (int i = 0; i < s; ++i) {
            std::cout << "  beta_" << i+1 << ": " << beta[i] << std::endl;
        }
        
        std::cout << "=====================================" << std::endl;
    }

    std::vector<double> difference(const std::vector<double>& data, int order) { // Helper method for differencing.
        if (order == 0) return data;
        
        std::vector<double> diff(data.size() - 1);
        for (size_t i = 0; i < diff.size(); ++i) {
            diff[i] = data[i + 1] - data[i];
        }

        if (order > 1) {
            return difference(diff, order - 1);
        }
        
        return diff;
    }

    std::vector<double> inverse_difference(const std::vector<double>& differenced, const std::vector<double>& original, int order) {
        if (order == 0) return differenced;
        
        std::vector<double> inverted(differenced.size() + 1);
        inverted[0] = original[0];  // Use the original first value
        
        for (size_t i = 0; i < differenced.size(); ++i) {
            inverted[i + 1] = inverted[i] + differenced[i];
        }

        if (order > 1) { // Recursive call for higher orders.
            return inverse_difference(inverted, original, order - 1);
        }
        
        return inverted;
    }

    double ljung_box_test(int lag) { // Diagnostic checking using the Ljung-Box test.
        double q_stat = 0.0;
        int n = residuals.size();

        std::vector<double> autocorrs(lag, 0.0); // Calculate sample autocorrelations.
        double sum_resid_sq = 0.0;
        
        for (size_t t = 0; t < residuals.size(); ++t) {
            sum_resid_sq += residuals[t] * residuals[t];
        }
        
        for (int k = 1; k <= lag; ++k) {
            double sum_product = 0.0;
            
            for (size_t t = k; t < residuals.size(); ++t) {
                sum_product += residuals[t] * residuals[t - k];
            }
            
            autocorrs[k-1] = sum_product / sum_resid_sq;
        }

        for (int k = 1; k <= lag; ++k) { // Calculate Q-statistic.
            q_stat += (autocorrs[k-1] * autocorrs[k-1]) / (n - k);
        }
        
        q_stat = n * (n + 2) * q_stat;
        
        return q_stat;
    }

    void export_results(const std::string& filename) { // Export to CSV file (Find on Desktop).
        std::ofstream outfile(filename);
        
        outfile << "t,original,fitted,residual,conditional_variance,standardized_residual" << std::endl;
        
        for (size_t t = 0; t < original_data.size(); ++t) {
            double fitted = t < differenced_data.size() ? 
                           (differenced_data[t] - residuals[t]) : 0.0;
            
            double std_residual = t < residuals.size() && t < conditional_variances.size() ? 
                                 (residuals[t] / std::sqrt(conditional_variances[t])) : 0.0;
            
            outfile << t << "," 
                   << original_data[t] << "," 
                   << (t < differenced_data.size() ? fitted : 0.0) << "," 
                   << (t < residuals.size() ? residuals[t] : 0.0) << "," 
                   << (t < conditional_variances.size() ? conditional_variances[t] : 0.0) << "," 
                   << std_residual << std::endl;
        }
        
        outfile.close();
    }
};

int main() { // Main function to run the ARIMA-GARCH model.
    std::vector<double> data = {
        1.2, 1.4, 1.3, 1.5, 1.7, 1.6, 1.5, 1.6, 1.8, 2.0,
        2.1, 2.3, 2.2, 2.0, 1.9, 2.1, 2.3, 2.4, 2.6, 2.5,
        2.3, 2.2, 2.4, 2.5, 2.7, 2.9, 2.8, 2.6, 2.5, 2.7
    };

    ArimaGarchModel model(1, 1, 1, 1, 1);

    model.set_data(data);

    std::cout << "Training model..." << std::endl;
    model.train();

    int forecast_horizon = 10;
    std::cout << "\nForecasting " << forecast_horizon << " steps ahead:" << std::endl;
    auto forecasts = model.forecast(forecast_horizon);
    
    for (int i = 0; i < forecast_horizon; ++i) {
        std::cout << "t+" << i+1 << ": " 
                  << "Point forecast = " << forecasts[i].first 
                  << ", Variance = " << forecasts[i].second
                  << ", 95% CI = [" 
                  << forecasts[i].first - 1.96 * std::sqrt(forecasts[i].second) << ", "
                  << forecasts[i].first + 1.96 * std::sqrt(forecasts[i].second) << "]" 
                  << std::endl;
    }

    int num_paths = 1000; // Simulate paths.
    std::cout << "\nSimulating " << num_paths << " future paths..." << std::endl;
    auto paths = model.simulate_paths(forecast_horizon, num_paths);

    double confidence = 0.95; // Calculate VaR and ES at 95% confidence level.
    int var_horizon = 5;
    double var = model.calculate_var(paths, var_horizon, confidence);
    double es = model.calculate_es(paths, var_horizon, confidence);
    
    std::cout << "\nRisk metrics at t+" << var_horizon << ":" << std::endl;
    std::cout << "VaR(" << confidence * 100 << "%) = " << var << std::endl;
    std::cout << "ES(" << confidence * 100 << "%) = " << es << std::endl;

    model.export_results("arima_garch_results.csv"); // Exports.
    std::cout << "\nResults exported to 'arima_garch_results.csv'" << std::endl;
    
    return 0;
}