#include "ngram-index.h"
#include "common.h"
#include "log.h"
#include <iostream>
#include <algorithm>
#include <cstring>

// Constructor
NGramIndex::NGramIndex(int min_n, int max_n) 
    : ngram_min(min_n), 
      ngram_max(std::min(max_n, NGRAM_INDEX_MAX_DEFAULT)) {
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
}

// Index a virtual prompt
int NGramIndex::index_virtual_prompt(const std::vector<llama_token>& tokens) {
    // Add the tokens to the source manager and get the source ID
    int source_id = source_manager.add_virtual_prompt(tokens);
    
    // Process all possible n-grams within the size limits
    for (int n = ngram_min; n <= ngram_max; ++n) {
        for (size_t i = 0; i + n <= tokens.size(); ++i) {
            // Create n-gram key from the current sequence
            ngram_index_key key(&tokens[i], n);
            
            // Add the location to the index
            index[key].add(ngram_location(
                ngram_location::StorageType::VIRTUAL_PROMPT,
                source_id,
                i
            ));
        }
    }
    
    return source_id;
}

// Draft using the n-gram index
llama_token NGramIndex::draft(const llama_token* tokens, int n_tokens, llama_context* /* ctx */) {
    // Try different n-gram sizes, starting from the largest
    for (int n = std::min(n_tokens, ngram_max); n >= ngram_min; --n) {
        // Skip if we don't have enough tokens for this n-gram size
        if (n > n_tokens) continue;
        
        // Create key from the last n tokens
        ngram_index_key key(tokens + (n_tokens - n), n);
        
        // Look up in the index
        auto it = index.find(key);
        if (it != index.end() && !it->second.empty()) {
            // Get locations for this n-gram
            auto locations = it->second.get_locations();
            
            // For now, we'll use the most recently added location
            // More sophisticated selection strategies could be implemented here
            const auto& loc = locations.back();
            
            // Get the token following this n-gram
            llama_token next_token = source_manager.get_token_at_location(
                ngram_location(loc.type, loc.source_id, loc.position + n)
            );
            
            // Check if we got a valid token
            if (next_token != LLAMA_TOKEN_NULL) {
                return next_token;
            }
        }
    }
    
    // No draft available
    return LLAMA_TOKEN_NULL;
}

// Save the index to a file
bool NGramIndex::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_INF("Failed to open file for writing: %s", filename.c_str());
        return false;
    }
    
    // Write the header
    file.write("NGRMIDX1", 8); // File signature and version
    
    // Write n-gram size limits
    file.write(reinterpret_cast<const char*>(&ngram_min), sizeof(ngram_min));
    file.write(reinterpret_cast<const char*>(&ngram_max), sizeof(ngram_max));
    
    // Write the number of virtual prompts
    size_t num_prompts = source_manager.get_total_virtual_prompt_size();
    file.write(reinterpret_cast<const char*>(&num_prompts), sizeof(num_prompts));
    
    // For now, we only save the index structure, not the actual virtual prompts
    // A more complete implementation would save the virtual prompts as well
    
    // Write the number of entries in the index
    size_t num_entries = index.size();
    file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    
    // Write each index entry
    for (const auto& entry : index) {
        // Write the key
        file.write(reinterpret_cast<const char*>(&entry.first.size), sizeof(entry.first.size));
        file.write(reinterpret_cast<const char*>(entry.first.tokens), 
                   entry.first.size * sizeof(llama_token));
        
        // Write the number of locations
        int num_locations = entry.second.size();
        file.write(reinterpret_cast<const char*>(&num_locations), sizeof(num_locations));
        
        // Write each location
        auto locations = entry.second.get_locations();
        for (const auto& loc : locations) {
            int type_int = static_cast<int>(loc.type);
            file.write(reinterpret_cast<const char*>(&type_int), sizeof(type_int));
            file.write(reinterpret_cast<const char*>(&loc.source_id), sizeof(loc.source_id));
            file.write(reinterpret_cast<const char*>(&loc.position), sizeof(loc.position));
        }
    }
    
    return file.good();
}

// Load the index from a file
bool NGramIndex::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_INF("Failed to open file for reading: %s", filename.c_str());
        return false;
    }
    
    // Read and verify the header
    char signature[8];
    file.read(signature, 8);
    if (std::strncmp(signature, "NGRMIDX1", 8) != 0) {
        LOG_INF("Invalid file format or version: %s", filename.c_str());
        return false;
    }
    
    // Clear existing data
    clear();
    
    // Read n-gram size limits
    file.read(reinterpret_cast<char*>(&ngram_min), sizeof(ngram_min));
    file.read(reinterpret_cast<char*>(&ngram_max), sizeof(ngram_max));
    
    // Read the number of virtual prompts
    size_t num_prompts;
    file.read(reinterpret_cast<char*>(&num_prompts), sizeof(num_prompts));
    
    // For now, we don't load virtual prompts
    // A more complete implementation would load them
    
    // Read the number of entries in the index
    size_t num_entries;
    file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    
    // Read each index entry
    for (size_t i = 0; i < num_entries; ++i) {
        // Read the key
        ngram_index_key key;
        file.read(reinterpret_cast<char*>(&key.size), sizeof(key.size));
        file.read(reinterpret_cast<char*>(key.tokens), key.size * sizeof(llama_token));
        
        // Read the number of locations
        int num_locations;
        file.read(reinterpret_cast<char*>(&num_locations), sizeof(num_locations));
        
        // Create a location buffer
        location_buffer buffer;
        
        // Read each location
        for (int j = 0; j < num_locations; ++j) {
            int type_int;
            int source_id;
            size_t position;
            
            file.read(reinterpret_cast<char*>(&type_int), sizeof(type_int));
            file.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
            file.read(reinterpret_cast<char*>(&position), sizeof(position));
            
            ngram_location::StorageType type = 
                static_cast<ngram_location::StorageType>(type_int);
            
            buffer.add(ngram_location(type, source_id, position));
        }
        
        // Add to the index
        index[key] = buffer;
    }
    
    return file.good();
}

// Merge with another index
void NGramIndex::merge(const NGramIndex& other) {
    // Adjust n-gram size limits if necessary
    ngram_min = std::min(ngram_min, other.ngram_min);
    ngram_max = std::max(ngram_max, other.ngram_max);
    
    // Merge index entries
    for (const auto& entry : other.get_index()) {
        auto& dest_buffer = index[entry.first];
        auto locations = entry.second.get_locations();
        
        // Add each location to our buffer
        for (const auto& loc : locations) {
            dest_buffer.add(loc);
        }
    }
}

// Print statistics about the index
void NGramIndex::print_stats() const {
    // Count n-grams of each size
    std::vector<int> counts(ngram_max + 1, 0);
    
    for (const auto& entry : index) {
        counts[entry.first.size]++;
    }
    
    // Print statistics
    std::cout << "N-gram index statistics:" << std::endl;
    std::cout << "  Total entries: " << index.size() << std::endl;
    
    for (int n = ngram_min; n <= ngram_max; ++n) {
        std::cout << "  " << n << "-grams: " << counts[n] << std::endl;
    }
    
    // Calculate memory usage (approximate)
    size_t memory_usage = sizeof(NGramIndex);
    memory_usage += index.size() * (sizeof(ngram_index_key) + sizeof(location_buffer));
    
    std::cout << "  Approximate memory usage: " << (memory_usage / 1024) << " KB" << std::endl;
    
    // Print detailed index information for debugging
    std::cout << "\nDetailed index contents (first 20 entries):" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    int count = 0;
    for (const auto& entry : index) {
        if (count >= 20) break;
        
        const ngram_index_key& key = entry.first;
        const location_buffer& locs = entry.second;
        
        // Print the key
        std::cout << "  Key: [";
        for (int i = 0; i < key.size; i++) {
            std::cout << key.tokens[i];
            if (i < key.size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Print the locations
        std::cout << "  Locations (" << locs.size() << "):" << std::endl;
        
        // Get the locations as a vector
        auto locations = locs.get_locations();
        for (size_t i = 0; i < locations.size(); i++) {
            const auto& loc = locations[i];
            std::cout << "    - Type: " << static_cast<int>(loc.type) 
                      << ", Source: " << loc.source_id
                      << ", Position: " << loc.position << std::endl;
        }
        
        std::cout << std::endl;
        count++;
    }
    
    // Print verification of token type counts in the virtual prompts
    std::cout << "\nVirtual prompt analysis:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    int total_virtual_prompts = source_manager.get_total_virtual_prompt_size();
    std::cout << "  Total virtual prompts: " << total_virtual_prompts << std::endl;
    
    // Since we can't directly access tokens, provide information about how to do so
    std::cout << "  To see virtual prompt tokens, add them to the prompt array first" << std::endl;
    std::cout << "  and then use logging in your build_optimized_index function." << std::endl;
    
    std::cout << "------------------------------------------" << std::endl;
}

// Clear the index
void NGramIndex::clear() {
    index.clear();
    source_manager.clear_virtual_prompts();
}

// Global function to draft using an n-gram index
llama_token draft_with_ngram_index(const NGramIndex& index, const llama_token* tokens, int n_tokens, llama_context* ctx) {
    return const_cast<NGramIndex&>(index).draft(tokens, n_tokens, ctx);
}