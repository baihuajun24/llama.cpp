#include "ngram-table.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct sample_params {
    // Input file to sample from
    std::string input_file;
    
    // Output files for samples
    std::string output_file_1pct;
    std::string output_file_10pct;
    
    // Random seed for reproducibility
    unsigned int random_seed = 42;
    
    // Print verbose output
    bool verbose = false;
};

static void print_usage() {
    fprintf(stderr, "Usage: sample-ngram-table [options] <input_file>\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                    Show this help message and exit\n");
    fprintf(stderr, "  -i, --input FILENAME          Input ngram table file to sample from\n");
    fprintf(stderr, "  -o1, --output-1pct FILENAME   Output file for 1%% sample (default: sample_1pct.bin)\n");
    fprintf(stderr, "  -o10, --output-10pct FILENAME Output file for 10%% sample (default: sample_10pct.bin)\n");
    fprintf(stderr, "  -s, --seed N                  Random seed for reproducibility (default: 42)\n");
    fprintf(stderr, "  -v, --verbose                 Print verbose output\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  sample-ngram-table -i merged.bin -o1 merged_1pct.bin -o10 merged_10pct.bin\n");
}

static bool parse_params(int argc, char** argv, sample_params& params) {
    bool valid = true;
    
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
        } else if (arg == "-o1" || arg == "--output-1pct") {
            if (++i < argc) {
                params.output_file_1pct = argv[i];
            } else {
                fprintf(stderr, "Missing output filename after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-o10" || arg == "--output-10pct") {
            if (++i < argc) {
                params.output_file_10pct = argv[i];
            } else {
                fprintf(stderr, "Missing output filename after %s\n", arg.c_str());
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
    
    // Set default output files if not specified
    if (params.output_file_1pct.empty()) {
        params.output_file_1pct = "sample_1pct.bin";
    }
    
    if (params.output_file_10pct.empty()) {
        params.output_file_10pct = "sample_10pct.bin";
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
        LOG_INF("Output files: 1%% -> %s, 10%% -> %s\n", 
                params.output_file_1pct.c_str(), params.output_file_10pct.c_str());
        LOG_INF("Random seed: %u\n", params.random_seed);
    }
    
    // Call the sampling function
    bool success = NGramTable::sample_then_save(
        params.input_file,
        params.output_file_1pct,
        params.output_file_10pct,
        params.random_seed
    );
    
    if (success) {
        LOG_INF("Successfully created sample files!\n");
        LOG_INF("1%% sample: %s\n", params.output_file_1pct.c_str());
        LOG_INF("10%% sample: %s\n", params.output_file_10pct.c_str());
        return 0;
    } else {
        LOG_ERR("Failed to create sample files\n");
        return 1;
    }
} 