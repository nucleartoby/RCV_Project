#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

struct PriceData { // Holds the price data for each day.
    std::string date;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

struct Trade { // Holds the details of the trade.
    std::string entryDate;
    double entryPrice;
    std::string exitDate;
    double exitPrice;
    double profit;
    double profitPercentage;
};

class MAcrossoverStrategy { // Adds the moving average crossover strat.
private:
    std::vector<PriceData> priceData;
    std::vector<double> shortMA;
    std::vector<double> longMA;
    std::vector<Trade> trades;
    
    int shortPeriod;
    int longPeriod;
    double initialCapital;
    double currentCapital;
    bool inPosition;
    
    std::vector<double> calculateMA(int period) { // Calculates the moving average.
        std::vector<double> ma(priceData.size(), 0.0);
        
        for (size_t i = period - 1; i < priceData.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += priceData[i - j].close;
            }
            ma[i] = sum / period;
        }
        
        return ma;
    }
    
public:
    MAcrossoverStrategy(int shortPeriod, int longPeriod, double initialCapital) // Constructor.
        : shortPeriod(shortPeriod), longPeriod(longPeriod),
          initialCapital(initialCapital), currentCapital(initialCapital),
          inPosition(false) {}
    
    bool loadCSVData(const std::string& filename) { // Loads the CSV data.
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            PriceData data;

            std::getline(ss, data.date, ',');
            std::getline(ss, item, ','); data.open = std::stod(item);
            std::getline(ss, item, ','); data.high = std::stod(item);
            std::getline(ss, item, ','); data.low = std::stod(item);
            std::getline(ss, item, ','); data.close = std::stod(item);
            std::getline(ss, item, ','); data.volume = std::stod(item);
            
            priceData.push_back(data);
        }
        
        file.close();

        if (priceData.size() > static_cast<size_t>(longPeriod)) { // Checks if there are enough data points.
            shortMA = calculateMA(shortPeriod);
            longMA = calculateMA(longPeriod);
            return true;
        } else {
            std::cerr << "Error: Not enough data points for the selected MA periods." << std::endl;
            return false;
        }
    }
    
    void runBacktest() { // Runs the backtest
        Trade currentTrade;

        for (size_t i = longPeriod; i < priceData.size() - 1; ++i) { // Iterating on the data.
            bool buySignal = shortMA[i-1] <= longMA[i-1] && shortMA[i] > longMA[i];
            bool sellSignal = shortMA[i-1] >= longMA[i-1] && shortMA[i] < longMA[i];

            if (buySignal && !inPosition) { // Checks for the buy signal.
                currentTrade.entryDate = priceData[i+1].date;
                currentTrade.entryPrice = priceData[i+1].open;
                inPosition = true;
            }

            if (sellSignal && inPosition) { // Checks for the sell signal.
                currentTrade.exitDate = priceData[i+1].date;
                currentTrade.exitPrice = priceData[i+1].open;
                currentTrade.profit = currentTrade.exitPrice - currentTrade.entryPrice;
                currentTrade.profitPercentage = (currentTrade.profit / currentTrade.entryPrice) * 100.0;
                
                trades.push_back(currentTrade); // Store the trade details.

                currentCapital *= (1.0 + currentTrade.profitPercentage / 100.0); // Update capital.
                inPosition = false;
            }
        }

        if (inPosition) {
            currentTrade.exitDate = priceData.back().date; // If still in position, close it at the last price.
            currentTrade.exitPrice = priceData.back().close;
            currentTrade.profit = currentTrade.exitPrice - currentTrade.entryPrice; // Calculate profit.
            currentTrade.profitPercentage = (currentTrade.profit / currentTrade.entryPrice) * 100.0; // Percentage profit.
            
            trades.push_back(currentTrade); // Store trade details.

            currentCapital *= (1.0 + currentTrade.profitPercentage / 100.0);
            inPosition = false;
        }
    }
    
    void printResults() { //Prints the results.
        std::cout << "=== Moving Average Crossover Strategy Backtest Results ===" << std::endl;
        std::cout << "Short MA Period: " << shortPeriod << std::endl;
        std::cout << "Long MA Period: " << longPeriod << std::endl;
        std::cout << "Initial Capital: $" << std::fixed << std::setprecision(2) << initialCapital << std::endl;
        std::cout << "Final Capital: $" << std::fixed << std::setprecision(2) << currentCapital << std::endl;
        std::cout << "Total Return: " << std::fixed << std::setprecision(2) 
                  << ((currentCapital - initialCapital) / initialCapital * 100.0) << "%" << std::endl;
        std::cout << "Number of Trades: " << trades.size() << std::endl;

        int winningTrades = 0; // Counts the winnings trades.
        for (const auto& trade : trades) {
            if (trade.profit > 0) {
                winningTrades++;
            }
        }
        
        double winRate = trades.empty() ? 0.0 : (static_cast<double>(winningTrades) / trades.size() * 100.0); // Calculates the win rate.
        std::cout << "Win Rate: " << std::fixed << std::setprecision(2) << winRate << "%" << std::endl;

        double totalProfitPercentage = 0.0; // Calculates the average profit per trade.
        for (const auto& trade : trades) {
            totalProfitPercentage += trade.profitPercentage;
        }
        
        double avgProfitPerTrade = trades.empty() ? 0.0 : (totalProfitPercentage / trades.size());
        std::cout << "Average Profit Per Trade: " << std::fixed << std::setprecision(2) << avgProfitPerTrade << "%" << std::endl; // Prints all details

        std::cout << "\nTrade Details:" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << std::left << std::setw(12) << "Entry Date" 
                  << std::setw(12) << "Entry Price" 
                  << std::setw(12) << "Exit Date" 
                  << std::setw(12) << "Exit Price" 
                  << std::setw(12) << "Profit" 
                  << std::setw(12) << "Profit %" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        
        for (const auto& trade : trades) {
            std::cout << std::left << std::setw(12) << trade.entryDate 
                      << std::setw(12) << std::fixed << std::setprecision(2) << trade.entryPrice 
                      << std::setw(12) << trade.exitDate 
                      << std::setw(12) << std::fixed << std::setprecision(2) << trade.exitPrice 
                      << std::setw(12) << std::fixed << std::setprecision(2) << trade.profit 
                      << std::setw(12) << std::fixed << std::setprecision(2) << trade.profitPercentage << "%" << std::endl;
        }
    }
    
    void exportTradeResults(const std::string& filename) { // Exports the trades to CSV file.
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open output file " << filename << std::endl;
            return;
        }

        file << "Entry Date,Entry Price,Exit Date,Exit Price,Profit,Profit %" << std::endl;

        for (const auto& trade : trades) { // Writes trade details to the file.
            file << trade.entryDate << "," 
                 << std::fixed << std::setprecision(2) << trade.entryPrice << "," 
                 << trade.exitDate << "," 
                 << std::fixed << std::setprecision(2) << trade.exitPrice << "," 
                 << std::fixed << std::setprecision(2) << trade.profit << "," 
                 << std::fixed << std::setprecision(2) << trade.profitPercentage << std::endl;
        }
        
        file.close();
        std::cout << "Trade results exported to " << filename << std::endl;
    }
};

int main(int argc, char* argv[]) { // Main function to run the backtest.
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <csv_filename> [short_period=10] [long_period=30] [initial_capital=10000]" << std::endl; // Input params
        return 1;
    }
    
    std::string filename = argv[1];
    int shortPeriod = (argc > 2) ? std::stoi(argv[2]) : 10; // Default params for short period.
    int longPeriod = (argc > 3) ? std::stoi(argv[3]) : 30; // Default params for long period.
    double initialCapital = (argc > 4) ? std::stod(argv[4]) : 10000.0; // Default initial capital.
    
    MAcrossoverStrategy strategy(shortPeriod, longPeriod, initialCapital); // Creates the strategy object.
    
    if (!strategy.loadCSVData(filename)) { // Loads CSV.
        return 1;
    }
    
    strategy.runBacktest();
    strategy.printResults();

    std::string outputFilename = "ma_crossover_results.csv";
    strategy.exportTradeResults(outputFilename); // Export results.
    
    return 0;
}