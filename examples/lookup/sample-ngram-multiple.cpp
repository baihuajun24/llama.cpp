#include "ngram-table.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

struct sample_params {
    // Input file to sample from
    std::string input_file;
    
    // Output prefix for sample files
    std::string output_prefix;
    
    // Percentages to sample
    std::vector<int> percentages;
    
    // Random seed for reproducibility
    unsigned int random_seed = 42;
    
    // Print verbose output
    bool verbose = false;
};

static void print_usage() {
    fprintf(stderr, "Usage: sample-ngram-multiple [options] <input_file>\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                    Show this help message and exit\n");
    fprintf(stderr, "  -i, --input FILENAME          Input ngram table file to sample from\n");
    fprintf(stderr, "  -o, --output-prefix PREFIX    Output prefix for sample files (default: sample)\n");
    fprintf(stderr, "  -p, --percentages \"P1,P2,P3\"  Comma-separated list of percentages (default: 1,10,20,40,60,80)\n");
    fprintf(stderr, "  -s, --seed N                  Random seed for reproducibility (default: 42)\n");
    fprintf(stderr, "  -v, --verbose                 Print verbose output\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  sample-ngram-multiple -i merged.bin\n");
    fprintf(stderr, "  sample-ngram-multiple -i merged.bin -p \"1,5,10,25,50\" -o mydata\n");
    fprintf(stderr, "  sample-ngram-multiple -i merged.bin -p \"20,40,60,80\" -o subset\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Output files will be named: <prefix>_<percentage>pct.bin\n");
}

static std::vector<int> parse_percentages(const std::string& input) {
    std::vector<int> percentages;
    std::stringstream ss(input);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Remove whitespace
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            try {
                int percentage = std::stoi(item);
                if (percentage > 0 && percentage <= 100) {
                    percentages.push_back(percentage);
                } else {
                    fprintf(stderr, "Invalid percentage: %d (must be 1-100)\n", percentage);
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "Invalid percentage format: %s\n", item.c_str());
            }
        }
    }
    
    return percentages;
}

static bool parse_params(int argc, char** argv, sample_params& params) {
    bool valid = true;
    
    // Set default percentages
    params.percentages = {1, 10, 20, 40, 60, 80};
    params.output_prefix = "sample";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage();
            exit(0);
        } else if (arg == "-i" || arg == "--input") {
            if (++i < argc) {
                params.input_file = argv[i];
            } else {
                fprintf(stderr, "Missing input filename after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-o" || arg == "--output-prefix") {
            if (++i < argc) {
                params.output_prefix = argv[i];
            } else {
                fprintf(stderr, "Missing output prefix after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-p" || arg == "--percentages") {
            if (++i < argc) {
                params.percentages = parse_percentages(argv[i]);
                if (params.percentages.empty()) {
                    fprintf(stderr, "No valid percentages specified\n");
                    valid = false;
                }
            } else {
                fprintf(stderr, "Missing percentages after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (++i < argc) {
                params.random_seed = std::stoul(argv[i]);
            } else {
                fprintf(stderr, "Missing seed value after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            valid = false;
        } else {
            // If no -i specified, treat as input file
            if (params.input_file.empty()) {
                params.input_file = arg;
            } else {
                fprintf(stderr, "Multiple input files specified\n");
                valid = false;
            }
        }
    }
    
    if (params.input_file.empty()) {
        fprintf(stderr, "No input file specified\n");
        valid = false;
    }
    
    return valid;
}

int main(int argc, char** argv) {
    sample_params params;
    
    if (!parse_params(argc, argv, params)) {
        print_usage();
        return 1;
    }
    
    if (params.verbose) {
        LOG_INF("Sampling ngram table from: %s\n", params.input_file.c_str());
        LOG_INF("Output prefix: %s\n", params.output_prefix.c_str());
        LOG_INF("Percentages: ");
        for (size_t i = 0; i < params.percentages.size(); i++) {
            printf("%d%%", params.percentages[i]);
            if (i < params.percentages.size() - 1) printf(", ");
        }
        printf("\n");
        LOG_INF("Random seed: %u\n", params.random_seed);
    }
    
    // Call the sampling function
    bool success = NGramTable::sample_multiple_percentages(
        params.input_file,
        params.percentages,
        params.output_prefix,
        params.random_seed
    );
    
    if (success) {
        LOG_INF("Successfully created sample files!\n");
        for (int percentage : params.percentages) {
            LOG_INF("%d%% sample: %s_%dpct.bin\n", percentage, params.output_prefix.c_str(), percentage);
        }
        return 0;
    } else {
        LOG_ERR("Failed to create sample files\n");
        return 1;
    }
} 