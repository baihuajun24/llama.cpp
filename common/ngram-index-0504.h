#ifndef NGRAM_INDEX_H
#define NGRAM_INDEX_H

#include "llama.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <array>
#include <fstream>

// Default maximum n-gram length
#define NGRAM_INDEX_MAX_DEFAULT 6

// Maximum buffer size for storing locations of n-gram occurrences
#define NGRAM_LOCATION_BUFFER_SIZE 3

/**
 * Key for the n-gram index
 * Represents an n-gram sequence as a fixed-size array of tokens
 */
struct ngram_index_key {
    llama_token tokens[NGRAM_INDEX_MAX_DEFAULT];
    int size;

    // Create a key from a sequence of tokens
    ngram_index_key(const llama_token* tokens_in, int size_in) {
        size = std::min(size_in, NGRAM_INDEX_MAX_DEFAULT);
        std::memcpy(tokens, tokens_in, size * sizeof(llama_token));
    }

    // Default constructor for map creation
    ngram_index_key() : size(0) {
        for (int i = 0; i < NGRAM_INDEX_MAX_DEFAULT; ++i) {
            tokens[i] = 0;  // Initialize with zeros
        }
    }

    // Equality comparison for map lookup
    bool operator==(const ngram_index_key& other) const {
        if (size != other.size) return false;
        return std::memcmp(tokens, other.tokens, size * sizeof(llama_token)) == 0;
    }
};

/**
 * Hash function for n-gram index keys
 */
struct ngram_index_key_hash {
    std::size_t operator()(const ngram_index_key& key) const {
        std::size_t hash = 0;
        // Simple hash combining each token with size
        for (int i = 0; i < key.size; ++i) {
            hash = hash * 31 + key.tokens[i];
        }
        hash = hash * 31 + key.size;
        return hash;
    }
};

/**
 * Represents the location of an n-gram sequence
 * Can point to a virtual prompt (in-memory) or a file on disk
 */
struct ngram_location {
    enum class StorageType {
        VIRTUAL_PROMPT,  // In-memory virtual prompt
        DISK_FILE        // On-disk file storage
    };

    StorageType type;    // Type of storage
    int source_id;       // ID of the source (prompt index or file ID)
    size_t position;     // Position within the source

    // Default constructor to fix std::array initialization
    ngram_location() 
        : type(StorageType::VIRTUAL_PROMPT), source_id(0), position(0) {}

    ngram_location(StorageType type_in, int source_id_in, size_t position_in)
        : type(type_in), source_id(source_id_in), position(position_in) {}
};

/**
 * Fixed-size circular buffer to store n-gram locations
 * Uses FIFO replacement strategy
 */
class location_buffer {
private:
    std::array<ngram_location, NGRAM_LOCATION_BUFFER_SIZE> locations;
    int next_index;
    int count;

public:
    // Initialize with default-constructed ngram_location objects
    location_buffer() : next_index(0), count(0) {
        // No additional initialization needed as ngram_location now has a default constructor
    }

    // Add a new location to the buffer (FIFO replacement)
    void add(const ngram_location& loc) {
        locations[next_index] = loc;
        next_index = (next_index + 1) % NGRAM_LOCATION_BUFFER_SIZE;
        if (count < NGRAM_LOCATION_BUFFER_SIZE) {
            count++;
        }
    }

    // Get the number of locations stored
    int size() const {
        return count;
    }

    // Check if the buffer is empty
    bool empty() const {
        return count == 0;
    }

    // Get the locations in order (most recently added last)
    std::vector<ngram_location> get_locations() const {
        std::vector<ngram_location> result;
        result.reserve(count);
        
        // Start with oldest entry
        int start = count < NGRAM_LOCATION_BUFFER_SIZE
                  ? 0
                  : next_index;
                  
        for (int i = 0; i < count; ++i) {
            int idx = (start + i) % NGRAM_LOCATION_BUFFER_SIZE;
            result.push_back(locations[idx]);
        }
        
        return result;
    }
};

/**
 * Manager for virtual prompts and on-disk sources
 * Handles loading, accessing, and caching source data
 */
class ngram_source_manager {
private:
    std::vector<std::vector<llama_token>> virtual_prompts;
    // File handling will be added in a future implementation

public:
    // Add a new virtual prompt and return its ID
    int add_virtual_prompt(const std::vector<llama_token>& tokens) {
        int id = virtual_prompts.size();
        virtual_prompts.push_back(tokens);
        return id;
    }
    
    // Append a token to an existing virtual prompt
    // Returns true if successful, false if prompt ID is invalid
    bool append_token_to_prompt(int prompt_id, llama_token token) {
        if (prompt_id >= 0 && (size_t)prompt_id < virtual_prompts.size()) {
            virtual_prompts[prompt_id].push_back(token);
            return true;
        }
        return false;
    }

    // Get token at a specific location
    llama_token get_token_at_location(const ngram_location& loc) {
        if (loc.type == ngram_location::StorageType::VIRTUAL_PROMPT) {
            if (loc.source_id >= 0 && (size_t)loc.source_id < virtual_prompts.size() &&
                loc.position < virtual_prompts[loc.source_id].size()) {
                return virtual_prompts[loc.source_id][loc.position];
            }
        }
        // File access will be implemented later
        
        return LLAMA_TOKEN_NULL;
    }

    // Get context around a location - retrieve tokens at and after a location
    std::vector<llama_token> get_tokens_at_location(const ngram_location& loc, size_t n) {
        std::vector<llama_token> result;
        
        if (loc.type == ngram_location::StorageType::VIRTUAL_PROMPT) {
            if (loc.source_id >= 0 && (size_t)loc.source_id < virtual_prompts.size()) {
                auto& tokens = virtual_prompts[loc.source_id];
                size_t end = std::min(loc.position + n, tokens.size());
                
                for (size_t i = loc.position; i < end; ++i) {
                    result.push_back(tokens[i]);
                }
            }
        }
        // File access will be implemented later
        
        return result;
    }

    // Clear all virtual prompts
    void clear_virtual_prompts() {
        virtual_prompts.clear();
    }

    // Get the number of virtual prompts
    size_t get_virtual_prompt_count() const {
        return virtual_prompts.size();
    }

    // Get total size of all tokens in all virtual prompts
    size_t get_total_virtual_prompt_size() const {
        size_t total = 0;
        for (const auto& prompt : virtual_prompts) {
            total += prompt.size();
        }
        return total;
    }
    
    // Get the size of a specific virtual prompt
    size_t get_virtual_prompt_size(int prompt_id) const {
        if (prompt_id >= 0 && (size_t)prompt_id < virtual_prompts.size()) {
            return virtual_prompts[prompt_id].size();
        }
        return 0;
    }
};

/**
 * Main n-gram index class
 * Maps n-gram sequences to their locations for efficient retrieval
 */
class NGramIndex {
private:
    int ngram_min;  // Minimum n-gram size to index
    int ngram_max;  // Maximum n-gram size to index
    
    // Main index mapping n-gram keys to their locations
    std::unordered_map<ngram_index_key, location_buffer, ngram_index_key_hash> index;
    
    // Source data manager
    ngram_source_manager source_manager;

public:
    // Constructor
    NGramIndex(int min_n = 1, int max_n = NGRAM_INDEX_MAX_DEFAULT);

    // Index a virtual prompt
    int index_virtual_prompt(const std::vector<llama_token>& tokens);

    // Add a single token to the index (for incremental updating)
    void add_token(const llama_token* context, int context_size, llama_token new_token);

    // Draft using the n-gram index (returns a single token)
    llama_token draft(const llama_token* tokens, int n_tokens, llama_context* ctx);
    
    // Draft multiple tokens using the n-gram index
    // Returns the number of tokens added to the draft
    int draft_multiple(const llama_token* tokens, int n_tokens, int n_draft, 
                       std::vector<llama_token>& drafted_tokens, llama_context* ctx);

    // Save the index to a file
    bool save(const std::string& filename);

    // Load the index from a file
    bool load(const std::string& filename);

    // Merge with another index
    void merge(const NGramIndex& other);

    // Print statistics about the index
    void print_stats() const;

    // Clear the index
    void clear();

    // Access the raw index (for direct manipulation)
    std::unordered_map<ngram_index_key, location_buffer, ngram_index_key_hash>& get_index() {
        return index;
    }

    // Get read-only access to the raw index
    const std::unordered_map<ngram_index_key, location_buffer, ngram_index_key_hash>& get_index() const {
        return index;
    }
};

// Global function to draft using an n-gram index (single token)
llama_token draft_with_ngram_index(const NGramIndex& index, const llama_token* tokens, int n_tokens, llama_context* ctx);

// Global function to draft multiple tokens at once
int draft_multiple_with_ngram_index(const NGramIndex& index, const llama_token* tokens, int n_tokens, 
                                    int n_draft, std::vector<llama_token>& drafted_tokens, llama_context* ctx);

#endif // NGRAM_INDEX_H 