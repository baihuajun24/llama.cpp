#pragma once

#include "llama.h"

#include <unordered_map>
#include <string>
#include <vector>

// Default max n-gram size, can be adjusted via command line
#define NGRAM_INDEX_MAX_DEFAULT 4

// Data structures for n-gram index lookup

struct ngram_index_key {
    llama_token* tokens;
    int size;

    ngram_index_key(int max_size) {
        size = max_size;
        tokens = new llama_token[max_size];
        for (int i = 0; i < max_size; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    ngram_index_key(const llama_token* input, int ngram_size, int max_size) {
        size = max_size;
        tokens = new llama_token[max_size];
        for (int i = 0; i < max_size; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    ~ngram_index_key() {
        delete[] tokens;
    }

    // Copy constructor
    ngram_index_key(const ngram_index_key& other) {
        size = other.size;
        tokens = new llama_token[size];
        for (int i = 0; i < size; ++i) {
            tokens[i] = other.tokens[i];
        }
    }

    // Move constructor
    ngram_index_key(ngram_index_key&& other) noexcept {
        size = other.size;
        tokens = other.tokens;
        other.tokens = nullptr;
        other.size = 0;
    }

    // Copy assignment
    ngram_index_key& operator=(const ngram_index_key& other) {
        if (this != &other) {
            delete[] tokens;
            size = other.size;
            tokens = new llama_token[size];
            for (int i = 0; i < size; ++i) {
                tokens[i] = other.tokens[i];
            }
        }
        return *this;
    }

    // Move assignment
    ngram_index_key& operator=(ngram_index_key&& other) noexcept {
        if (this != &other) {
            delete[] tokens;
            size = other.size;
            tokens = other.tokens;
            other.tokens = nullptr;
            other.size = 0;
        }
        return *this;
    }

    bool operator==(const ngram_index_key& other) const {
        if (size != other.size) return false;
        for (int i = 0; i < size; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

struct ngram_index_key_hash {
    size_t operator()(const ngram_index_key& key) const {
        size_t hash = 0;
        for (int i = 0; i < key.size; ++i) {
            // Combine hashes using FNV-1a
            hash ^= std::hash<llama_token>{}(key.tokens[i]);
            hash *= 16777619;
        }
        return hash;
    }
};

// Value type: vector of indices where this n-gram appears in the prompt
typedef std::vector<int> ngram_index_positions;

// n-gram -> list of positions where this n-gram appears
typedef std::unordered_map<ngram_index_key, ngram_index_positions, ngram_index_key_hash> ngram_index;

// Build an n-gram index from a token sequence
// token_sequence: the input token sequence to index
// ngram_min/ngram_max: the min/max size of n-grams to index
// max_indices_per_ngram: limit the number of indices stored per n-gram (0 = no limit)
// print_progress: whether to print progress information
ngram_index build_ngram_index(
    const std::vector<llama_token>& token_sequence,
    int ngram_min,
    int ngram_max,
    int max_indices_per_ngram = 0,
    bool print_progress = false);

// Save an n-gram index to a file
void save_ngram_index(ngram_index& index, const std::string& filename);

// Load an n-gram index from a file
ngram_index load_ngram_index(const std::string& filename);

// Draft tokens using the n-gram index
// inp: the current input sequence
// draft: the draft sequence (initially contains one token)
// n_draft: maximum number of tokens to draft
// prompt: the original prompt used to build the index
// index: the n-gram index
// ngram_min/ngram_max: the min/max size of n-grams to use for lookup
// selection_strategy: how to select from multiple positions (0=first, 1=random)
void draft_with_ngram_index(
    const std::vector<llama_token>& inp,
    std::vector<llama_token>& draft,
    int n_draft,
    const std::vector<llama_token>& prompt,
    const ngram_index& index,
    int ngram_min,
    int ngram_max,
    int selection_strategy = 0);

// Print statistics about an n-gram index
void print_ngram_index_stats(const ngram_index& index);