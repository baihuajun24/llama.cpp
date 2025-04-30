#include "ngram-index.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <chrono>
#include <filesystem>

struct build_params {
    // Input file(s) to process
    std::vector<std::string> input_files;
    
    // Output file to save the index
    std::string output_file = "basic_ngram_index.bin";
    
    // File to save prompt tokens
    std::string prompt_tokens_file = "basic_prompt.bin";
    
    // Minimum n-gram size
    int ngram_min = 1;
    
    // Maximum n-gram size
    int ngram_max = 6;
    
    // Print verbose statistics
    bool verbose = false;

    // Model path
    std::string model_path;
    
    // Load existing index instead of building one
    std::string load_index;
    
    // Flag to indicate we should test the index after building/loading
    bool run_tests = false;
};

static void print_usage() {
    fprintf(stderr, "Usage: build-ngram-index [options] <input files>\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help               Show this help message and exit\n");
    fprintf(stderr, "  -o, --output FILENAME    Output file for the built index (default: static_ngram_index.bin)\n");
    fprintf(stderr, "  -p, --prompt FILENAME    Output file for prompt tokens (default: basic_prompt.bin)\n");
    fprintf(stderr, "  -m, --model FILENAME     Model path (required)\n");
    fprintf(stderr, "  --ngram-min N            Minimum n-gram size (default: 1)\n");
    fprintf(stderr, "  --ngram-max N            Maximum n-gram size (default: 6)\n");
    fprintf(stderr, "  -v, --verbose            Print verbose statistics\n");
}

static bool parse_params(int argc, char** argv, build_params& params) {
    bool valid = true;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage();
            exit(0);
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) {
                params.output_file = argv[i];
            } else {
                fprintf(stderr, "Missing output filename after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i < argc) {
                params.prompt_tokens_file = argv[i];
            } else {
                fprintf(stderr, "Missing prompt tokens filename after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-m" || arg == "--model") {
            if (++i < argc) {
                params.model_path = argv[i];
            } else {
                fprintf(stderr, "Missing model path after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "--ngram-min") {
            if (++i < argc) {
                params.ngram_min = std::stoi(argv[i]);
                if (params.ngram_min < 1) {
                    fprintf(stderr, "Minimum n-gram size must be at least 1\n");
                    valid = false;
                }
            } else {
                fprintf(stderr, "Missing value after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "--ngram-max") {
            if (++i < argc) {
                params.ngram_max = std::stoi(argv[i]);
                if (params.ngram_max > NGRAM_INDEX_MAX_DEFAULT) {
                    fprintf(stderr, "Maximum n-gram size cannot exceed %d\n", NGRAM_INDEX_MAX_DEFAULT);
                    valid = false;
                }
            } else {
                fprintf(stderr, "Missing value after %s\n", arg.c_str());
                valid = false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            valid = false;
        } else {
            // Assume it's an input file
            params.input_files.push_back(arg);
        }
    }
    
    if (params.input_files.empty()) {
        fprintf(stderr, "No input files specified\n");
        valid = false;
    }
    
    if (params.model_path.empty()) {
        fprintf(stderr, "Model path not specified. Use -m/--model to specify a model file.\n");
        valid = false;
    }
    
    if (params.ngram_min > params.ngram_max) {
        fprintf(stderr, "Minimum n-gram size cannot be greater than maximum\n");
        valid = false;
    }
    
    return valid;
}

// Structure to hold next token frequencies for a given n-gram
struct next_token_stats {
    std::unordered_map<llama_token, int> frequencies;
    llama_token most_frequent_token = LLAMA_TOKEN_NULL;
    int total_occurrences = 0;
    
    void add_occurrence(llama_token token) {
        frequencies[token]++;
        total_occurrences++;
        
        // Update most frequent token if needed
        if (most_frequent_token == LLAMA_TOKEN_NULL || 
            frequencies[token] > frequencies[most_frequent_token]) {
            most_frequent_token = token;
        }
    }
    
    double get_confidence() const {
        if (total_occurrences == 0) return 0.0;
        return static_cast<double>(frequencies.at(most_frequent_token)) / total_occurrences;
    }
};

// Process a single file and update the n-gram statistics
static void process_file(const std::string& filename, 
                  std::unordered_map<ngram_index_key, next_token_stats, ngram_index_key_hash>& stats,
                  llama_context* ctx,
                  int ngram_min, 
                  int ngram_max,
                  bool verbose) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read file content
    std::ifstream file(filename);
    if (!file) {
        LOG_ERR("Failed to open file: %s\n", filename.c_str());
        return;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    if (verbose) {
        LOG_INF("Read %zu bytes from %s\n", content.length(), filename.c_str());
    }
    
    // Tokenize content
    std::vector<llama_token> tokens = common_tokenize(ctx, content, true, true);
    
    if (verbose) {
        LOG_INF("Tokenized to %zu tokens\n", tokens.size());
    }
    
    // Process n-grams
    int ngrams_processed = 0;
    
    for (int n = ngram_min; n <= ngram_max; ++n) {
        for (size_t i = 0; i <= tokens.size() - n - 1; ++i) {
            // Create n-gram key
            ngram_index_key key(&tokens[i], n);
            
            // The next token after this n-gram
            llama_token next_token = tokens[i + n];
            
            // Update statistics
            stats[key].add_occurrence(next_token);
            
            ngrams_processed++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    LOG_INF("Processed %d n-grams from %s in %.2f seconds\n", 
            ngrams_processed, filename.c_str(), duration / 1000.0f);
}

// Build an optimized n-gram index from the statistics
static NGramIndex build_optimized_index(
    const std::unordered_map<ngram_index_key, next_token_stats, ngram_index_key_hash>& stats,
    int ngram_min,
    int ngram_max,
    bool verbose,
    const std::string& prompt_path = "") {
        
    // Create a new index
    NGramIndex index(ngram_min, ngram_max);
    
    // We'll build a virtual prompt of all the tokens we need
    std::vector<llama_token> prompt_tokens;
    
    // Create a mapping of each n-gram to its position in the prompt
    std::vector<std::pair<ngram_index_key, size_t>> ngram_positions;
    
    // First pass: collect all unique n-grams and their next tokens
    for (const auto& entry : stats) {
        const ngram_index_key& key = entry.first;
        const next_token_stats& token_stats = entry.second;
        
        // Only include n-grams with a next token and good confidence
        if (token_stats.most_frequent_token != LLAMA_TOKEN_NULL && token_stats.get_confidence() >= 0.5) {
            // Record the position where this n-gram will be stored
            size_t pos = prompt_tokens.size();
            ngram_positions.push_back(std::make_pair(key, pos));
            
            // Add the n-gram tokens to the prompt
            for (int i = 0; i < key.size; ++i) {
                prompt_tokens.push_back(key.tokens[i]);
            }
            
            // Add the most frequent next token
            prompt_tokens.push_back(token_stats.most_frequent_token);
        }
    }
    
    if (verbose) {
        LOG_INF("Built virtual prompt with %zu tokens for %zu n-grams\n", 
                prompt_tokens.size(), ngram_positions.size());
        
        // Print first 100 tokens of the virtual prompt
        LOG_INF("First 100 tokens of virtual prompt:\n");
        for (size_t i = 0; i < std::min((size_t)100, prompt_tokens.size()); i += 2) {
            LOG_INF("[key token %d] [predicted token %d]", 
                    prompt_tokens[i], prompt_tokens[i+1]);
        }
        LOG_INF("\n");
    }
    
    // Save prompt tokens to file if a path is provided
    if (!prompt_path.empty() && !prompt_tokens.empty()) {
        std::ofstream prompt_file(prompt_path, std::ios::binary);
        if (prompt_file.is_open()) {
            // Write the number of tokens
            size_t num_tokens = prompt_tokens.size();
            prompt_file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));
            
            // Write all tokens
            prompt_file.write(reinterpret_cast<const char*>(prompt_tokens.data()), 
                             num_tokens * sizeof(llama_token));
            
            bool success = prompt_file.good();
            if (success) {
                LOG_INF("Successfully saved virtual prompt with %zu tokens to %s\n", 
                        num_tokens, prompt_path.c_str());
            } else {
                LOG_INF("Error occurred while writing virtual prompt to %s\n", prompt_path.c_str());
            }
            
            prompt_file.close();
        } else {
            LOG_ERR("Failed to open file for writing virtual prompt: %s\n", prompt_path.c_str());
        }
    }
    
    // Now index the virtual prompt
    if (!prompt_tokens.empty()) {
        // First, add the entire virtual prompt to the index
        int source_id = index.index_virtual_prompt(prompt_tokens);
        LOG_INF("Added virtual prompt to index with ID: %d\n", source_id);
        
        if (verbose) {
            LOG_INF("Added %zu direct n-gram entries to index\n", ngram_positions.size());
        }
    }

    index.print_stats();
    
    return index;
}

// Print statistics about the n-gram statistics
static void print_stats(const std::unordered_map<ngram_index_key, next_token_stats, ngram_index_key_hash>& stats, 
                 llama_context* ctx) {
    LOG_INF("N-gram statistics:\n");
    
    // Count by n-gram size
    std::map<int, int> counts_by_size;
    
    // Find n-grams with highest frequency and confidence
    std::vector<std::pair<const ngram_index_key*, double>> top_confidence;
    std::vector<std::pair<const ngram_index_key*, int>> top_frequency;
    
    for (const auto& entry : stats) {
        const ngram_index_key& key = entry.first;
        const next_token_stats& token_stats = entry.second;
        
        // Count by size
        counts_by_size[key.size]++;
        
        // Track top confidence
        if (token_stats.total_occurrences > 5) {  // Only consider n-grams with enough data
            top_confidence.push_back(std::make_pair(&key, token_stats.get_confidence()));
        }
        
        // Track top frequency
        top_frequency.push_back(std::make_pair(&key, token_stats.total_occurrences));
    }
    
    // Print counts by size
    LOG_INF("  N-gram counts by size:\n");
    for (const auto& count : counts_by_size) {
        LOG_INF("    %d-grams: %d\n", count.first, count.second);
    }
    
    // Sort and print top confidence
    std::sort(top_confidence.begin(), top_confidence.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    LOG_INF("\n  Top 10 n-grams by confidence:\n");
    for (size_t i = 0; i < std::min(size_t(10), top_confidence.size()); ++i) {
        const ngram_index_key* key = top_confidence[i].first;
        double confidence = top_confidence[i].second;
        
        std::string key_text = "Input tokens: [";
        for (int j = 0; j < key->size; ++j) {
            key_text += std::to_string(key->tokens[j]);
            if (j < key->size - 1) key_text += ", ";
        }
        key_text += "] -> Predicted: ";
        
        // Add predicted token
        const auto& token_stats = stats.at(*key);
        key_text += std::to_string(token_stats.most_frequent_token);

        LOG_INF("    %s (confidence: %.2f%%)\n", key_text.c_str(), confidence * 100);
    }
}

int main(int argc, char** argv) {
    build_params params;
    
    if (!parse_params(argc, argv, params)) {
        print_usage();
        return 1;
    }
    
    LOG_INF("Building n-gram index from %zu files (n-gram sizes %d-%d)\n", 
            params.input_files.size(), params.ngram_min, params.ngram_max);
    
    // We need a llama context for tokenization
    llama_backend_init();
    
    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    
    // Use the model path specified via command line
    LOG_INF("Using model: %s\n", params.model_path.c_str());
    
    // Use updated API calls
    llama_model* model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }
    
    // Update to use the non-deprecated API
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    // Map to store n-gram statistics
    std::unordered_map<ngram_index_key, next_token_stats, ngram_index_key_hash> ngram_stats;
    
    // Process each input file
    for (const auto& file : params.input_files) {
        process_file(file, ngram_stats, ctx, params.ngram_min, 1, params.verbose); //set ngram_max to 1, for simple static cache
    }
    
    LOG_INF("Collected statistics for %zu unique n-grams\n", ngram_stats.size());
    
    // Print detailed statistics if verbose
    if (params.verbose) {
        print_stats(ngram_stats, ctx);
        //print_stats(ngram_stats);
    }
    
    // Build optimized index
    NGramIndex index = build_optimized_index(ngram_stats, params.ngram_min, params.ngram_max, params.verbose, params.prompt_tokens_file);
    
    // Print index statistics
    //index.print_stats();
    
    // Save the index
    index.save(params.output_file);
    
    LOG_INF("Saved n-gram index to %s\n", params.output_file.c_str());
    
    // Clean up using updated API calls
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // Can we write a part to load prompt from prompt_tokens_file?
    // use a vector of llama_token to store the prompt tokens
    std::vector<llama_token> vp_tokens;
    std::ifstream prompt_file(params.prompt_tokens_file, std::ios::binary);
    if (prompt_file.is_open()) {
        // Read the number of tokens
        size_t num_tokens;
        prompt_file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
        // print tokens
        for (size_t i = 0; i < num_tokens; i++) {
            llama_token token;
            prompt_file.read(reinterpret_cast<char*>(&token), sizeof(token));
            vp_tokens.push_back(token);
        }
    }
    
    // print the prompt tokens
    LOG_INF("Prompt tokens:\n");
    for (size_t i = 0; i < vp_tokens.size(); i++) {
        LOG_INF("%d ", vp_tokens[i]);
    }
    LOG_INF("\n");
    

    return 0;
}
