// rest-index.cpp
#include "rest-index.h"
#include "log.h"
#include <stdexcept>
#include <iostream>

RESTIndex::RESTIndex() : reader(nullptr), loaded(false) {
    // Initialize with nullptr
}

RESTIndex::~RESTIndex() {
    // Clean up if needed
    if (reader) {
        delete reader;
        reader = nullptr;
    }
}

bool RESTIndex::buildIndex(const std::vector<llama_token>& tokens, const std::string& indexPath) {
    LOG_ERR("Building index is not implemented in this version. Please build index using the provided DraftRetriever tools.\n");
    return false;
}

bool RESTIndex::loadIndex(const std::string& indexPath) {
    try {
        // Create the DraftRetriever reader with the index path
        reader = new rust::Box<Reader>(new_reader(indexPath.c_str()));
        loaded = true;
        LOG_INF("0423 REST index loaded: %s\n", indexPath.c_str());
        return true;
    } catch (const std::exception& e) {
        LOG_INF("0423 Failed to load REST index: %s\n", e.what());
        return false;
    }
}

std::vector<llama_token> RESTIndex::search(const std::vector<llama_token>& prefix, int choices) {
    if (!loaded || !reader) {
        LOG_ERR("REST index not loaded or reader is null\n");
        return {};
    }
    
    try {
        // Convert llama_token vector to int32_t slice for DraftRetriever
        std::vector<int32_t> int_prefix(prefix.begin(), prefix.end());
        rust::Slice<const int32_t> slice_prefix(int_prefix.data(), int_prefix.size());
        
        LOG_INF("Searching with prefix of length %zu\n", prefix.size());
        
        // Call the search function with default parameters for k and length
        // k = 5000, length = 10 (these are default values from the header)
        auto paths = (*reader)->search(slice_prefix, 64);
        
        // Process results - extract tokens from paths
        std::vector<llama_token> tokens;
        
        LOG_INF("Found %zu paths\n", paths.size());
        
        // If we found any paths
        if (!paths.empty()) {
            // Get the first path
            const auto& first_path = paths[0];
            
            LOG_INF("First path has %zu tokens\n", first_path.path.size());
            
            // Extract tokens from the path, filtering out padding tokens (-2)
            for (const auto& token : first_path.path) {
                if (token != -2) { // Skip padding tokens
                    tokens.push_back(static_cast<llama_token>(token));
                }
            }
            
            LOG_INF("After filtering padding, found %zu tokens\n", tokens.size());
        }
        
        return tokens;
    } catch (const std::exception& e) {
        LOG_ERR("Error in REST search: %s\n", e.what());
        return {};
    }
}

bool RESTIndex::isLoaded() const {
    return loaded;
}

void rest_draft(
    const std::vector<llama_token>& inp,
    std::vector<llama_token>& draft,
    int n_draft,
    RESTIndex& restIndex,
    int max_prefix_len,
    int min_prefix_len
) {
    // Validate inputs
    if (inp.empty() || draft.empty() || !restIndex.isLoaded()) {
        LOG_DBG("Invalid inputs for REST drafting\n");
        return;
    }
    
    // The first token in draft should match the last token in inp
    llama_token last_token = inp.back();
    if (draft[0] != last_token) {
        LOG_DBG("Draft first token doesn't match input last token\n");
        return;
    }
    
    // Try decreasing prefix lengths
    for (int prefix_len = std::min(max_prefix_len, (int)inp.size()); 
         prefix_len >= min_prefix_len; --prefix_len) {
        
        // Extract the prefix
        std::vector<llama_token> prefix;
        prefix.reserve(prefix_len);
        
        for (int i = inp.size() - prefix_len; i < (int)inp.size(); ++i) {
            prefix.push_back(inp[i]);
        }
        
        LOG_DBG("Trying REST search with prefix length %d\n", prefix_len);
        
        // Search for continuations
        std::vector<llama_token> continuations = restIndex.search(prefix, 64);
        
        if (!continuations.empty()) {
            LOG_DBG("Found %zu continuation tokens with prefix length %d\n", 
                   continuations.size(), prefix_len);
            
            // Clear draft except for the first token
            draft.resize(1);
            
            // Add continuation tokens (up to n_draft)
            size_t tokens_to_add = std::min(continuations.size(), static_cast<size_t>(n_draft));
            for (size_t i = 0; i < tokens_to_add; ++i) {
                draft.push_back(continuations[i]);
            }
            
            LOG_DBG("Added %zu tokens to draft from REST search\n", tokens_to_add);
            return;
        }
    }
    
    LOG_DBG("No continuations found with REST search\n");
}