// rest-index.cpp
#include "rest-index.h"
#include "log.h"
#include <stdexcept>

RESTIndex::RESTIndex() : reader(nullptr), loaded(false) {
}

RESTIndex::~RESTIndex() {
    if (reader) {
        draftretriever::draftretriever_free_reader(reader);
        reader = nullptr;
    }
    loaded = false;
}

bool RESTIndex::buildIndex(const std::vector<llama_token>& tokens, const std::string& indexPath) {
    // Convert llama_tokens to the format expected by DraftRetriever (int32_t)
    std::vector<int32_t> convertedTokens;
    convertedTokens.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        convertedTokens.push_back(static_cast<int32_t>(token));
    }
    
    try {
        // Use the build_index function from draftretriever
        int result = draftretriever::draftretriever_build_index(
            convertedTokens.data(),
            convertedTokens.size(),
            indexPath.c_str(),
            35000 // default vocab size
        );
        
        return result == 0;
    } catch (const std::exception& e) {
        LOG_ERR("Error building REST index: %s\n", e.what());
        return false;
    }
}

bool RESTIndex::loadIndex(const std::string& indexPath) {
    // Clean up any existing reader
    if (reader) {
        draftretriever::draftretriever_free_reader(reader);
        reader = nullptr;
        loaded = false;
    }
    
    try {
        // Create a new reader using the C API
        reader = draftretriever::draftretriever_create_reader(indexPath.c_str());
        loaded = (reader != nullptr);
        return loaded;
    } catch (const std::exception& e) {
        LOG_ERR("Error loading REST index: %s\n", e.what());
        return false;
    }
}

std::vector<llama_token> RESTIndex::search(const std::vector<llama_token>& prefix, int choices) {
    if (!loaded || !reader) {
        return {};
    }
    
    // Convert llama_tokens to int32_t
    std::vector<int32_t> convertedPrefix;
    convertedPrefix.reserve(prefix.size());
    
    for (const auto& token : prefix) {
        convertedPrefix.push_back(static_cast<int32_t>(token));
    }
    
    try {
        // Prepare the SearchResults struct
        draftretriever::SearchResults results = {};
        
        // Call the search function
        int ret = draftretriever::draftretriever_search(
            reader,
            convertedPrefix.data(),
            convertedPrefix.size(),
            choices, // choices
            5000,    // k - default value from Python code
            10,      // long_ - default value from Python code
            &results
        );
        
        if (ret != 0 || results.list_count == 0 || results.token_lists == nullptr) {
            return {};
        }
        
        // Extract the first list of tokens (first candidate)
        int32_t *first_list = results.token_lists[0];
        size_t count = results.token_counts[0];
        
        // Convert to llama_token and filter out padding tokens (-2)
        std::vector<llama_token> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; i++) {
            int32_t token = first_list[i];
            if (token != -2) { // Skip padding tokens
                result.push_back(static_cast<llama_token>(token));
            }
        }
        
        // Free the search results
        draftretriever::draftretriever_free_search_results(&results);
        
        return result;
    } catch (const std::exception& e) {
        LOG_ERR("Error during REST search: %s\n", e.what());
        return {};
    }
}

bool RESTIndex::isLoaded() const {
    return loaded && reader != nullptr;
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