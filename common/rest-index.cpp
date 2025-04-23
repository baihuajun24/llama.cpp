// rest-index.cpp
#include "rest-index.h"
#include "log.h"
#include <stdexcept>
#include <iostream>
#include "common.h"

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

std::vector<RESTCandidate> RESTIndex::searchCandidates(const std::vector<llama_token>& prefix, int choices) {
    std::vector<RESTCandidate> candidates;
    
    if (!loaded || !reader) {
        LOG_ERR("REST index not loaded or reader is null\n");
        return candidates;
    }
    
    try {
        // Convert llama_token vector to int32_t slice for DraftRetriever
        std::vector<int32_t> int_prefix(prefix.begin(), prefix.end());
        rust::Slice<const int32_t> slice_prefix(int_prefix.data(), int_prefix.size());
        
        LOG_INF("Searching with prefix of length %zu\n", prefix.size());
        
        // Call the search function with requested number of choices
        auto paths = (*reader)->search(slice_prefix, choices);
        
        LOG_INF("Found %zu paths\n", paths.size());
        
        // Process each path into a candidate
        for (const auto& path : paths) {
            RESTCandidate candidate;
            
            // Extract tokens from the path, filtering out padding tokens (-2)
            for (const auto& token : path.path) {
                if (token != -2) { // Skip padding tokens
                    candidate.tokens.push_back(static_cast<llama_token>(token));
                }
            }
            
            // Only add candidates that have tokens
            if (!candidate.tokens.empty()) {
                candidates.push_back(candidate);
            }
        }
        
        LOG_INF("Processed %zu valid candidates\n", candidates.size());
        
        return candidates;
    } catch (const std::exception& e) {
        LOG_ERR("Error in REST search candidates: %s\n", e.what());
        return candidates;
    }
}

std::vector<llama_token> RESTIndex::search(const std::vector<llama_token>& prefix, int choices) {
    // Use the new searchCandidates method and return the first candidate's tokens
    std::vector<RESTCandidate> candidates = searchCandidates(prefix, choices);
    
    if (!candidates.empty()) {
        LOG_INF("Returning first candidate with %zu tokens\n", candidates[0].tokens.size());
        return candidates[0].tokens;
    }
    
    return {};
}

void RESTIndex::printCandidates(const std::vector<RESTCandidate>& candidates, 
                               llama_context* ctx, int max_candidates, int max_tokens) {
    if (candidates.empty()) {
        LOG_INF("No candidates to print\n");
        return;
    }
    
    int num_to_print = std::min((int)candidates.size(), max_candidates);
    LOG_INF("Printing %d out of %zu candidates:\n", num_to_print, candidates.size());
    
    for (int i = 0; i < num_to_print; i++) {
        const auto& candidate = candidates[i];
        LOG_INF("Candidate %d (%zu tokens):\n", i+1, candidate.tokens.size());
        
        // Print tokens in batches for readability
        int tokens_to_print = std::min((int)candidate.tokens.size(), max_tokens);
        
        for (int j = 0; j < tokens_to_print; j++) {
            if (ctx != nullptr) {
                // Use common_token_to_piece instead of llama_token_to_piece
                std::string token_str = common_token_to_piece(ctx, candidate.tokens[j]);
                LOG_INF("  Token %d: %d, '%s'\n", j, candidate.tokens[j], token_str.c_str());
            } else {
                // Without context, just print the token IDs
                LOG_INF("  Token %d: %d\n", j, candidate.tokens[j]);
            }
        }
        
        if ((int)candidate.tokens.size() > max_tokens) {
            LOG_INF("  ... (truncated %zu more tokens)\n", candidate.tokens.size() - max_tokens);
        }
        LOG_INF("\n");
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