#ifndef NGRAM_INDEX_H
#define NGRAM_INDEX_H

#include "llama.h"

#include <unordered_map>
#include <string>
#include <vector>

// Default configuration parameters
#define LLAMA_NGRAM_MIN         1      // Minimum n-gram size to index
#define LLAMA_NGRAM_MAX         6      // Maximum n-gram size to index 
#define LLAMA_LOCATION_MAX      3      // Maximum locations to store per n-gram

/**
 * N-gram key structure
 * Represents an n-gram sequence as fixed-size array of tokens
 */
struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];

    // Default constructor
    common_ngram() {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    // Create from token sequence
    common_ngram(const llama_token * input, const int ngram_size) {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    // Equality operator for map lookup
    bool operator==(const common_ngram & other) const {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * Hash function for tokens
 */
struct common_token_hash_function {
    size_t operator()(const llama_token token) const {
        // Fibonacci hashing
        return token * 11400714819323198485llu;
    }
};

/**
 * Hash function for n-gram keys
 */
struct common_ngram_hash_function {
    size_t operator()(const common_ngram & ngram) const {
        size_t hash = common_token_hash_function{}(ngram.tokens[0]);
        for (int i = 1; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= common_token_hash_function{}(ngram.tokens[i]);
        }
        return hash;
    }
};

/**
 * Location in the prompt where a specific n-gram occurs
 */
struct ngram_location {
    int32_t index;  // Position in the token array

    ngram_location() : index(-1) {}
    ngram_location(int32_t idx) : index(idx) {}
};

/**
 * Fixed-size circular buffer to store n-gram locations
 * Uses FIFO replacement strategy
 */
class location_buffer {
private:
    std::vector<ngram_location> locations;
    size_t max_locations;
    size_t next_index;

public:
    location_buffer(size_t max_size = LLAMA_LOCATION_MAX) 
        : max_locations(max_size), next_index(0) {}

    // Add a new location to the buffer (FIFO replacement)
    void add(const ngram_location& loc) {
        if (locations.size() < max_locations) {
            locations.push_back(loc);
        } else {
            locations[next_index] = loc;
        }
        next_index = (next_index + 1) % max_locations;
    }

    // Get all stored locations
    const std::vector<ngram_location>& get_locations() const {
        return locations;
    }

    // Check if the buffer is empty
    bool empty() const {
        return locations.empty();
    }

    // Get the number of locations stored
    size_t size() const {
        return locations.size();
    }
};

/**
 * Main n-gram index class
 * Maps n-gram sequences to their locations for efficient retrieval
 */
class NGramIndex {
private:
    // The actual token sequence
    std::vector<llama_token> prompt_tokens;
    
    // Mapping from n-grams to their locations in prompt_tokens
    std::unordered_map<common_ngram, location_buffer, common_ngram_hash_function> index;
    
    // Configuration
    int ngram_min;
    int ngram_max;
    int location_max;

public:
    /**
     * Constructor
     * @param min_n Minimum n-gram size to index
     * @param max_n Maximum n-gram size to index
     * @param loc_max Maximum locations to store per n-gram
     */
    NGramIndex(int min_n = LLAMA_NGRAM_MIN, int max_n = LLAMA_NGRAM_MAX, int loc_max = LLAMA_LOCATION_MAX);
    
    /**
     * Initialize index with a sequence of tokens
     * @param tokens The tokens to index
     */
    void index_prompt(const std::vector<llama_token>& tokens);
    
    /**
     * Add a single token to the index incrementally
     * @param context The context tokens including the new token at the end
     * @param context_size Number of tokens in the context
     */
    void add_token(const llama_token* context, int context_size);
    
    /**
    * Draft tokens based on context
    * @param inp The context tokens generated so far
    * @param draft The token sequence to draft (expected to initially contain the previously sampled token)
    * @param n_draft Desired number of tokens to add to draft
    * @return A pair of (match_n, tokens_added) where match_n is the n-gram size that was matched
    *         and tokens_added is the number of tokens added to the draft
    */
    std::pair<int, int> draft(const std::vector<llama_token>& inp, std::vector<llama_token>& draft, int n_draft);
    
    /**
     * Save the index to a file
     * @param file_path Path to save the index
     * @return True if successful, false otherwise
     */
    bool save(const std::string& file_path) const;
    
    /**
     * Load the index from a file
     * @param file_path Path to load the index from
     * @return True if successful, false otherwise
     */
    bool load(const std::string& file_path);

    /**
     * Print statistics about the index
     */
    void print_stats() const;
    
    /**
     * Clear the index
     */
    void clear();
    
    /**
     * Access the prompt tokens
     * @return Reference to the token vector
     */
    const std::vector<llama_token>& get_prompt_tokens() const {
        return prompt_tokens;
    }
    
    /**
     * Access the index
     * @return Reference to the index map
     */
    const std::unordered_map<common_ngram, location_buffer, common_ngram_hash_function>& get_index() const {
        return index;
    }
};

#endif // NGRAM_INDEX_H