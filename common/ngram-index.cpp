#include "ngram-index.h"
#include "common.h"
#include "log.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstring>

// Constructor with parameter validation
NGramIndex::NGramIndex(int min_n, int max_n, int loc_max) 
    : ngram_min(min_n), 
      ngram_max(std::min(max_n, LLAMA_NGRAM_MAX)),
      location_max(loc_max) {
    
    // Validate n-gram size limits
    if (ngram_min < 1) {
        LOG_INF("Minimum n-gram size %d is less than 1, setting to 1", ngram_min);
        ngram_min = 1;
    }
    
    if (ngram_max < ngram_min) {
        LOG_INF("Maximum n-gram size %d is less than minimum %d, setting to %d", 
                ngram_max, ngram_min, ngram_min);
        ngram_max = ngram_min;
    }
    
    if (location_max < 1) {
        LOG_INF("Maximum locations %d is less than 1, setting to %d", 
                location_max, LLAMA_LOCATION_MAX);
        location_max = LLAMA_LOCATION_MAX;
    }
}

// Initialize index with a sequence of tokens
void NGramIndex::index_prompt(const std::vector<llama_token>& tokens) {
    // Clear existing data
    clear();
    
    // Store tokens
    prompt_tokens = tokens;
    
    // Process all possible n-grams within the size limits
    for (int n = ngram_min; n <= ngram_max; ++n) {
        for (size_t i = 0; i + n <= tokens.size(); ++i) {
            // Create n-gram key from the current sequence
            common_ngram key(&tokens[i], n);
            
            // Add the location to the index
            index[key].add(ngram_location(i));
        }
    }
    
    LOG_INF("Indexed %zu tokens with %zu unique n-grams", tokens.size(), index.size());
}

// Add a single token to the index incrementally
void NGramIndex::add_token(const llama_token* context, int context_size) {
    // Validate context
    if (context == nullptr || context_size <= 0) {
        return;
    }
    
    // Get the new token (last one in context)
    llama_token new_token = context[context_size - 1];
    
    // Add the token to our prompt tokens
    prompt_tokens.push_back(new_token);
    
    // Current position in prompt_tokens
    int32_t position = prompt_tokens.size() - 1;
    
    // Add n-grams of different sizes ending with the new token
    for (int n = ngram_min; n <= ngram_max && n <= context_size; ++n) {
        // Calculate how many tokens we need from the context
        int start_pos = context_size - n;
        
        // Create an n-gram key
        common_ngram key(&context[start_pos], n);
        
        // Add the location to the index
        index[key].add(ngram_location(position - (n - 1)));
    }
}

// Draft tokens based on context
std::pair<int, int> NGramIndex::draft(const std::vector<llama_token>& inp, std::vector<llama_token>& draft, int n_draft) {
    // Make sure we have some context and draft has at least one token
    if (inp.empty() || draft.empty() || n_draft <= 0) {
        return {0, 0}; // Using brace initialization for pair
    }
    
    // Keep the first token in draft (the previously sampled token)
    llama_token first_token = draft[0];
    
    // Reset draft to just the first token
    draft.resize(1);
    
    // The total tokens added to draft
    int tokens_added = 0;
    
    // Try different n-gram sizes, starting from the largest
    for (int n = std::min((int)inp.size(), ngram_max); n >= ngram_min; --n) {
        // Skip if we don't have enough tokens for this n-gram size
        if (n > (int)inp.size()) continue;
        
        // Create key from the last n tokens in inp
        common_ngram key(&inp[inp.size() - n], n);
        
        // Look up in the index
        auto it = index.find(key);
        if (it != index.end() && !it->second.empty()) {
            // Get locations for this n-gram
            const auto& locations = it->second.get_locations();
            
            // Use the most recently added location
            const auto& loc = locations.back();
            
            // The position after the n-gram in prompt_tokens
            int32_t next_pos = loc.index + n;
            
            // Add tokens to the draft until we reach n_draft or the end of prompt_tokens
            while (tokens_added < n_draft && next_pos < (int32_t)prompt_tokens.size()) {
                // Get the next token
                llama_token next_token = prompt_tokens[next_pos++];
                
                // Add it to the draft
                draft.push_back(next_token);
                tokens_added++;
            }
            
            // If we added tokens, return
            if (tokens_added > 0) {
                return {n, tokens_added}; // Using brace initialization for pair
            }
        }
    }
    
    // No tokens added
    return {0, 0}; // Using brace initialization for pair
}

// Save the index to a file
bool NGramIndex::save(const std::string& file_path) const {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s", file_path.c_str());
        return false;
    }
    
    // Write file signature and version
    const char* signature = "NGRAMIDX";
    file.write(signature, 8);
    
    // Write configuration
    file.write(reinterpret_cast<const char*>(&ngram_min), sizeof(ngram_min));
    file.write(reinterpret_cast<const char*>(&ngram_max), sizeof(ngram_max));
    file.write(reinterpret_cast<const char*>(&location_max), sizeof(location_max));
    
    // Write prompt tokens
    size_t token_count = prompt_tokens.size();
    file.write(reinterpret_cast<const char*>(&token_count), sizeof(token_count));
    for (const auto& token : prompt_tokens) {
        file.write(reinterpret_cast<const char*>(&token), sizeof(token));
    }
    
    // Write index size
    size_t index_size = index.size();
    file.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
    
    // Write each index entry
    for (const auto& entry : index) {
        // Write the key
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        
        // Write locations
        const auto& locations = entry.second.get_locations();
        size_t loc_count = locations.size();
        file.write(reinterpret_cast<const char*>(&loc_count), sizeof(loc_count));
        
        for (const auto& loc : locations) {
            file.write(reinterpret_cast<const char*>(&loc.index), sizeof(loc.index));
        }
    }
    
    return file.good();
}

// Load the index from a file
bool NGramIndex::load(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for reading: %s", file_path.c_str());
        return false;
    }
    
    // Clear existing data
    clear();
    
    // Read and verify file signature
    char signature[9] = {0};
    file.read(signature, 8);
    if (strcmp(signature, "NGRAMIDX") != 0) {
        LOG_ERR("Invalid file format or signature");
        return false;
    }
    
    // Read configuration
    file.read(reinterpret_cast<char*>(&ngram_min), sizeof(ngram_min));
    file.read(reinterpret_cast<char*>(&ngram_max), sizeof(ngram_max));
    file.read(reinterpret_cast<char*>(&location_max), sizeof(location_max));
    
    // Read prompt tokens
    size_t token_count;
    file.read(reinterpret_cast<char*>(&token_count), sizeof(token_count));
    prompt_tokens.resize(token_count);
    for (size_t i = 0; i < token_count; i++) {
        file.read(reinterpret_cast<char*>(&prompt_tokens[i]), sizeof(llama_token));
    }
    
    // Read index size
    size_t index_size;
    file.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
    
    // Read each index entry
    for (size_t i = 0; i < index_size; i++) {
        // Read the key
        common_ngram key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        
        // Read locations
        size_t loc_count;
        file.read(reinterpret_cast<char*>(&loc_count), sizeof(loc_count));
        
        // Create location buffer with proper size
        location_buffer buf(location_max);
        
        // Read each location
        for (size_t j = 0; j < loc_count; j++) {
            int32_t idx;
            file.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            buf.add(ngram_location(idx));
        }
        
        // Add to index
        index[key] = buf;
    }
    
    LOG_INF("Loaded index with %zu tokens and %zu n-grams", prompt_tokens.size(), index.size());
    return file.good();
}

// Print statistics about the index
void NGramIndex::print_stats() const {
    std::cout << "NGramIndex Statistics:" << std::endl;
    std::cout << "---------------------" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Min n-gram size: " << ngram_min << std::endl;
    std::cout << "  Max n-gram size: " << ngram_max << std::endl;
    std::cout << "  Max locations per n-gram: " << location_max << std::endl;
    std::cout << "Content:" << std::endl;
    std::cout << "  Prompt tokens: " << prompt_tokens.size() << std::endl;
    std::cout << "  Unique n-grams: " << index.size() << std::endl;
    
    // Count n-grams by size
    std::vector<int> counts(ngram_max + 1, 0);
    for (const auto& entry : index) {
        // Determine the actual n-gram size by counting non-null tokens
        int size = 0;
        for (int i = 0; i < LLAMA_NGRAM_MAX; i++) {
            if (entry.first.tokens[i] != LLAMA_TOKEN_NULL) {
                size++;
            } else {
                break;
            }
        }
        if (size > 0 && size <= ngram_max) {
            counts[size]++;
        }
    }
    
    // Print n-gram counts by size
    for (int n = ngram_min; n <= ngram_max; n++) {
        std::cout << "  " << n << "-grams: " << counts[n] << std::endl;
    }
    
    // Calculate approximate memory usage
    size_t mem_tokens = prompt_tokens.size() * sizeof(llama_token);
    size_t mem_index = index.size() * (sizeof(common_ngram) + sizeof(location_buffer));
    size_t mem_total = mem_tokens + mem_index + sizeof(NGramIndex);
    
    std::cout << "Memory usage (approximate):" << std::endl;
    std::cout << "  Tokens: " << mem_tokens / 1024 << " KB" << std::endl;
    std::cout << "  Index: " << mem_index / 1024 << " KB" << std::endl;
    std::cout << "  Total: " << mem_total / 1024 << " KB" << std::endl;
}

// Clear the index
void NGramIndex::clear() {
    prompt_tokens.clear();
    index.clear();
}