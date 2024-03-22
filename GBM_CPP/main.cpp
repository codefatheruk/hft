#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include <LightGBM/c_api.h>
#include <deque>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>


class MaxQuadraticUtilityPortfolioOptimizer {
public:
    Eigen::VectorXd returns; // Expected returns for each asset.
    Eigen::MatrixXd covariance; // Covariance matrix for the assets.
    double risk_aversion_factor; // Risk aversion parameter for the utility function.
    double learning_rate; // Learning rate for the gradient descent.
    int max_iterations; // Maximum number of iterations for the gradient descent.

    MaxQuadraticUtilityPortfolioOptimizer(const Eigen::VectorXd& returns, const Eigen::MatrixXd& covariance, double risk_aversion_factor =1, double learning_rate = 0.01, int max_iterations = 10000)
        : returns(returns), covariance(covariance), risk_aversion_factor(risk_aversion_factor), learning_rate(learning_rate), max_iterations(max_iterations) {}

    Eigen::VectorXd optimize() {
        int n = returns.size();
        Eigen::VectorXd weights = Eigen::VectorXd::Ones(n) / n;
        for (int iter = 0; iter < max_iterations; ++iter) {
            Eigen::VectorXd gradient = covariance * weights - returns * risk_aversion_factor;
            weights -= learning_rate * gradient; 

            double sum_weights = weights.sum();
            weights -= (sum_weights - 1.0) * Eigen::VectorXd::Ones(n) / n;

            for (int i = 0; i < n; ++i) {
                if (weights(i) < -1.0) weights(i) = -1.0;
                else if (weights(i) > 1.0) weights(i) = 1.0;
            }
        }
        return weights;
    }
};

struct OrderBookSnapshot {
    unsigned long long timestamp; // Unix timestamp in seconds
    double bidPrice;
    double bidQty;
    double askPrice;
    double askQty;

    OrderBookSnapshot(unsigned long long timestamp,double bidPrice, double bidQty, double askPrice, double askQty)
        : timestamp(timestamp), bidPrice(bidPrice), bidQty(bidQty),
          askPrice(askPrice), askQty(askQty) {}
};

struct InstrumentOrderBook {
    std::deque<OrderBookSnapshot> snapshots;
    void feedSnapshot(const OrderBookSnapshot& snapshot) {
        if (!snapshots.empty()) {
            auto lastSnapshot = snapshots.back();
            unsigned long long lastTimestamp = lastSnapshot.timestamp;
            for (unsigned long long ts = lastTimestamp + 1; ts < snapshot.timestamp; ++ts) {
                snapshots.push_back(OrderBookSnapshot(ts, lastSnapshot.bidPrice, lastSnapshot.bidQty, lastSnapshot.askPrice, lastSnapshot.askQty));
                if (snapshots.size() > 1000) {
                    snapshots.pop_front();
                }
            }
        }
        snapshots.push_back(snapshot);
        while (snapshots.size() > 1000) {
            snapshots.pop_front();
        }
    }
};

class PortfolioOptimizer {
private:
    std::unordered_map<std::string, InstrumentOrderBook> instrumentOrderBooks;
    std::unordered_map<std::string, BoosterHandle> GBM_models;
    unsigned long long latestTimestamp = 0; 

    void updateAllInstrumentsToLatestTimestamp(){
        for (auto& pair : instrumentOrderBooks) {
            auto& book = pair.second;
            if (!book.snapshots.empty()) {
                const auto& lastSnapshot = book.snapshots.back();
                for (unsigned long long ts = lastSnapshot.timestamp + 1; ts <= latestTimestamp; ++ts) {
                    book.feedSnapshot(OrderBookSnapshot(ts, lastSnapshot.bidPrice, lastSnapshot.bidQty, lastSnapshot.askPrice, lastSnapshot.askQty));
                }
            }
        }
    }

    Eigen::VectorXd calculateReturns(const std::string& instrumentName, size_t look_back ,size_t step) {
        const auto& book = instrumentOrderBooks[instrumentName];
        std::vector<double> returns;
        for (size_t i = 0; i< int(look_back/step) ; i++) {
            double prevPrice = (book.snapshots[book.snapshots.size() - 1 - (i+1) * step].askPrice + book.snapshots[book.snapshots.size() - 1 - (i+1) * step].bidPrice) / 2.0;
            double currPrice = (book.snapshots[book.snapshots.size() - 1 - i * step].askPrice + book.snapshots[book.snapshots.size() - 1 - i * step].bidPrice) / 2.0;
            returns.push_back(std::log(currPrice / prevPrice));
        }

        return Eigen::Map<Eigen::VectorXd>(returns.data(), returns.size());
    }

    Eigen::MatrixXd calculateCovarianceMatrix(size_t look_back ,size_t step) {
        std::vector<Eigen::VectorXd> allReturns;
        for (const auto& pair : instrumentOrderBooks) {
            allReturns.push_back(calculateReturns(pair.first, look_back ,step));
        }
        size_t n = allReturns.size(); // Number of instruments
        Eigen::MatrixXd covMatrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                double cov = (i == j) ? allReturns[i].array().square().mean() - std::pow(allReturns[i].mean(), 2) 
                                      : (allReturns[i].array() * allReturns[j].array()).mean() - allReturns[i].mean() * allReturns[j].mean();
                covMatrix(i, j) = cov;
                covMatrix(j, i) = cov; // Covariance matrix is symmetric
            }
        }

        return covMatrix;
    }

    std::vector<double> get_feature(const std::string& instrumentName, size_t step, size_t memory_horizon) {
        auto& book = instrumentOrderBooks[instrumentName];
        if (book.snapshots.size() < (step+1) * memory_horizon) {
            std::cout << "Not enough data." << std::endl;
            return {};
        }

        std::vector<double> mid_changes, diff_sma, obi;

        for (size_t i = memory_horizon - 1; i < book.snapshots.size(); ++i) {
            auto& snap = book.snapshots[i];
            double midPrice = (snap.askPrice + snap.bidPrice) / 2.0;
            double obiValue = (snap.askQty - snap.bidQty) / (snap.askQty + snap.bidQty);
            obi.push_back(obiValue);
            double midChange = std::log(midPrice) - std::log((book.snapshots[i - 1].askPrice + book.snapshots[i - 1].bidPrice) / 2.0);
            mid_changes.push_back(midChange);
            double sum = 0.0;
            for (size_t j = i; j > i - memory_horizon; --j) {
                sum += (book.snapshots[j].askPrice + book.snapshots[j].bidPrice) / 2.0;
            }
            double average = sum / memory_horizon;
            diff_sma.push_back(std::log(midPrice) - std::log(average));
            
        }
        std::vector<double> features;
        for(int i =0;i<memory_horizon;i++){
            features.push_back(mid_changes[mid_changes.size()-1-i*step]);
            features.push_back(diff_sma[diff_sma.size()-1-i*step]);
            features.push_back(obi[obi.size()-1-i*step]);
        }
        return features;
    }

    std::vector<double> predict_direction_probability(const std::string& instrumentName){
        int64_t out_len;
        int num_iterations = -1; 
        double* out_result;
        const int num_classes = 3;
        const char* param = "";
        std::vector<double> mat_output(num_classes *1, -1);
        if (LGBM_BoosterPredictForMat(GBM_models[instrumentName], &get_feature(instrumentName,4,60)[0], 
                                    C_API_DTYPE_FLOAT64, 1, 180, 1,  C_API_PREDICT_NORMAL, 0, 
                                    -1, param ,&out_len, &mat_output[0]) == 0) {
            return mat_output;
        }
    }
public:
    PortfolioOptimizer(const std::vector<std::string>& instrumentNames) {
        for (const auto& name : instrumentNames) {
            instrumentOrderBooks[name];
            int num_iter = -1;
            const std::string model_filename = "/home/ubuntu/models_240_4_60/" + name + "_model.txt";
            if (!LGBM_BoosterCreateFromModelfile(model_filename.c_str(),&num_iter, &GBM_models[name]) == 0) {
                std::cerr << "Failed to load model "+name << std::endl;
            }
        }
    }

    void feedOrderBookSnapshot(std::string instrumentName ,const OrderBookSnapshot& snapshot) {
        if (instrumentOrderBooks.find(instrumentName) != instrumentOrderBooks.end()) {
            instrumentOrderBooks[instrumentName].feedSnapshot(snapshot);
            if (snapshot.timestamp > latestTimestamp) {
                latestTimestamp = snapshot.timestamp;
                updateAllInstrumentsToLatestTimestamp();
            }
        } else {
            std::cout << "Instrument not found in portfolio: " << instrumentName << "\n";
        }
    }

    std::unordered_map<std::string, double> get_weights() {
        size_t look_back = 100; // Last 100 seconds
        size_t step = 4; // With a step of 4
        Eigen::MatrixXd covMatrix = calculateCovarianceMatrix(look_back, step);
        Eigen::VectorXd expectedReturns(instrumentOrderBooks.size());
        std::vector<std::string> instrumentNames;
        int idx = 0;
        for (const auto& pair : instrumentOrderBooks) {
            std::vector<double> direction_probability = predict_direction_probability(pair.first);

            expectedReturns(idx) = (direction_probability[2] - direction_probability[0])*
                (direction_probability[2] + direction_probability[0]);
            instrumentNames.push_back(pair.first); 
            ++idx;
        }
        MaxQuadraticUtilityPortfolioOptimizer optimizer(expectedReturns, covMatrix);
        Eigen::VectorXd weights = optimizer.optimize();
        double abs_sum = abs(weights(0)) + abs(weights(1)) + abs(weights(2));
        std::unordered_map<std::string, double> instrumentWeights;
        for (size_t i = 0; i < weights.size(); ++i) {
            instrumentWeights[instrumentNames[i]] = weights(i) / abs_sum;  
        }

        return instrumentWeights;
    }

};

long long parseDateTime(const std::string& dateTimeStr) {
    std::tm tm = {};
    std::istringstream ss(dateTimeStr);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    // Assuming the datetime is in UTC and your system is set to UTC
    auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
    return static_cast<long long>(timestamp);
}

OrderBookSnapshot parseLine(const std::string& line) {
    std::istringstream ss(line);
    std::string token;

    std::string dateTimeStr;
    double bidPrice = 0.0, bidQty = 0.0, askPrice = 0.0, askQty = 0.0;

    // Getting the date-time string
    std::getline(ss, dateTimeStr, ',');
    auto timestamp = parseDateTime(dateTimeStr);

    // Iterate through the remaining tokens
    for (int i = 0; std::getline(ss, token, ','); ++i) {
        if (i == 0) bidPrice = std::stod(token);
        else if (i == 1) bidQty = std::stod(token);
        else if (i == 11) askPrice = std::stod(token);
        else if (i == 12) askQty = std::stod(token);
    }

    return OrderBookSnapshot{timestamp, bidPrice, bidQty, askPrice, askQty};
}

void processFile(const std::string& filename, const std::string& instrumentName, PortfolioOptimizer& portfolio) {
    std::ifstream file(filename);
    std::string line;
    int count = 0;
    const int limit = 2000;
    while (std::getline(file, line) && count < limit) {
        OrderBookSnapshot level = parseLine(line);
        portfolio.feedOrderBookSnapshot(instrumentName, level);
        ++count;
    }
}
int main() {
   
    std::vector<std::string> instrumentNames = {"USDJPY", "EURUSD", "GBPUSD"};
    
    PortfolioOptimizer optimizer(instrumentNames);
    std::vector<std::pair<std::string, std::string>> files = {
        {"/home/ubuntu/GBM_CPP/tks_ny_2022-12-01_USDJPY.lob", "USDJPY"},
        {"/home/ubuntu/GBM_CPP/tks_ny_2022-12-01_GBPUSD.lob", "GBPUSD"},
        {"/home/ubuntu/GBM_CPP/tks_ny_2022-12-01_EURUSD.lob", "EURUSD"}
    };
    for (const auto& [filename, instrumentName] : files) {
        processFile(filename, instrumentName, optimizer);
    }
    std::unordered_map<std::string, double> weights = optimizer.get_weights();

    // Output the calculated weights
    for (const auto& weight : weights) {
        std::cout << "Instrument: " << weight.first << ", Weight: " << weight.second << std::endl;
    }
    return 0;
}
