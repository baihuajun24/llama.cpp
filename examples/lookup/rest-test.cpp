#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "rest-index.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

struct rest_params {
    // Path to a saved index to load
    std::string load_index;
    
    // Maximum prefix length for searching
    int max_prefix_len = 6;
    
    // Minimum prefix length for searching
    int min_prefix_len = 2;
    
    // Number of choices to consider
    int choices = 64;
};

void rest_print_usage(int argc, char ** argv) {
    printf("usage: %s [options]\n\n", argv[0]);
    printf("options:\n");
    printf("  -h, --help                  show this help message and exit\n");
    printf("  -m FNAME, --model FNAME     model path\n");
    printf("  --load-index FNAME          load index from this file\n");
    printf("  --max-prefix-len N          maximum prefix length to consider (default: 6)\n");
    printf("  --min-prefix-len N          minimum prefix length to consider (default: 2)\n");
    printf("  --choices N                 number of choices to consider (default: 64)\n");
}

void print_token_info(llama_context* ctx, const std::vector<llama_token>& tokens) {
    LOG_INF("Token IDs: [");
    for (size_t i = 0; i < tokens.size(); i++) {
        LOG_INF("%d", tokens[i]);
        if (i < tokens.size() - 1) {
            LOG_INF(", ");
        }
    }
    LOG_INF("]\n");
    
    LOG_INF("Token text: '");
    for (const auto& token : tokens) {
        LOG_INF("%s", common_token_to_piece(ctx, token).c_str());
    }
    LOG_INF("'\n");
}

int main(int argc, char** argv) {
    common_params params;
    rest_params rparams;
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP, rest_print_usage)) {
        return 1;
    }
    
    // Hardcode the index path instead of reading from command line
    // rparams.load_index = "/c/Users/Administrator/Documents/REST/datastore/datastore_stack_small.idx";
    rparams.load_index = "C:\\Users\\Administrator\\Documents\\REST\\datastore\\datastore_stack_small.idx";

    rparams.max_prefix_len = 6;
    rparams.min_prefix_len = 2;
    rparams.choices = 64;
    
    // Process command line arguments only for the other parameters
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--max-prefix-len" && i + 1 < argc) {
            rparams.max_prefix_len = std::stoi(argv[++i]);
        } else if (arg == "--min-prefix-len" && i + 1 < argc) {
            rparams.min_prefix_len = std::stoi(argv[++i]);
        } else if (arg == "--choices" && i + 1 < argc) {
            rparams.choices = std::stoi(argv[++i]);
        }
    }
    
    // No need to validate the load_index path since it's hardcoded
    
    if (rparams.max_prefix_len < rparams.min_prefix_len) {
        LOG_ERR("max-prefix-len must be greater than or equal to min-prefix-len\n");
        rest_print_usage(argc, argv);
        return 1;
    }
    
    common_init();
    
    // Initialize llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load the model
    common_init_result llama_init = common_init_from_params(params);
    
    if (!llama_init.model) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }
    
    llama_model* model = llama_init.model.get();
    llama_context* ctx = llama_init.context.get();
    
    // Initialize REST index
    RESTIndex rest_index;
    
    if (!rest_index.loadIndex(rparams.load_index)) {
        LOG_ERR("Failed to load REST index from %s\n", rparams.load_index.c_str());
        return 1;
    }
    
    LOG_INF("Successfully loaded REST index from %s\n", rparams.load_index.c_str());
    
    // Test string "for i in"
    const std::string test_string = "import pandas as";
    
    // Tokenize the test string
    std::vector<llama_token> test_tokens = common_tokenize(ctx, test_string, true, false);
    test_tokens = std::vector<llama_token>(test_tokens.begin() + 1, test_tokens.end()); // Remove <s> token
    
    LOG_INF("==========================================================\n");
    LOG_INF("Testing REST index with string: '%s'\n", test_string.c_str());
    LOG_INF("Tokenized as:\n");
    print_token_info(ctx, test_tokens);
    LOG_INF("==========================================================\n");
    
    // Try different prefix lengths
    for (int prefix_len = std::min(rparams.max_prefix_len, (int)test_tokens.size()); 
         prefix_len >= rparams.min_prefix_len; --prefix_len) {
        
        // Extract the prefix
        std::vector<llama_token> prefix(test_tokens.end() - prefix_len, test_tokens.end());
        
        LOG_DBG("Trying REST search with prefix length %d\n", prefix_len);
        
        // Search for candidate continuations
        std::vector<RESTCandidate> candidates = rest_index.searchCandidates(prefix, rparams.choices);
        
        // Print the candidates for debugging
        rest_index.printCandidates(candidates, ctx, 3, 20);
        
        // Get the first candidate for continuation (maintaining original behavior)
        std::vector<llama_token> continuations;
        if (!candidates.empty()) {
            continuations = candidates[0].tokens;
        }
        
        if (!continuations.empty()) {
            LOG_INF("Found %zu continuation tokens:\n", continuations.size());
            
            // Create a draft with the last token of the prefix
            std::vector<llama_token> draft = { prefix.back() };
            
            // Add continuations
            for (const auto& token : continuations) {
                draft.push_back(token);
            }
            
            // Print draft
            LOG_INF("Draft (with context token):\n");
            print_token_info(ctx, draft);
            
            // Also print just the continuations
            LOG_INF("Continuations only:\n");
            print_token_info(ctx, continuations);
            
            break;
        } else {
            LOG_INF("No continuations found with this prefix length\n");
        }
    }
    
    // Test the rest_draft function
    std::vector<llama_token> input_tokens = test_tokens;
    std::vector<llama_token> draft = { test_tokens.back() };
    
    LOG_INF("\n==========================================================\n");
    LOG_INF("Testing rest_draft function:\n");
    LOG_INF("Input: ");
    print_token_info(ctx, input_tokens);
    LOG_INF("Initial draft: ");
    print_token_info(ctx, draft);
    
    rest_draft(input_tokens, draft, 10, rest_index, rparams.max_prefix_len, rparams.min_prefix_len);
    
    LOG_INF("Result draft: ");
    print_token_info(ctx, draft);
    LOG_INF("==========================================================\n");
    
    llama_backend_free();
    
    return 0;
}
