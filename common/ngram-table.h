#pragma once

#include "llama.h"
#include "ngram-cache.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#define LLAMA_NGRAM_MAX 6

// Future token prediction with frequency count
struct FutureToken {
    llama_token token;
    uint32_t frequency;
};

class NGramTable {
public:
    NGramTable();
    ~NGramTable() = default;

    // Load table from binary file
    bool load(const std::string& filename);

    // Draft next tokens based on input sequence
    void draft(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
               int n_draft, int min_n, int max_n);

    // Get table statistics
    struct Stats {
        size_t total_ngrams;
        uint32_t min_n;
        uint32_t max_n;
        uint32_t horizon;
    };
    Stats get_stats() const;

private:
    static constexpr const char MAGIC[9] = "NGRAMTBL";
    static constexpr uint32_t VERSION = 1;
    
    uint32_t min_n;
    uint32_t max_n;
    uint32_t horizon;

    // Map from n-gram to its future token predictions
    std::unordered_map<common_ngram, std::vector<FutureToken>, common_ngram_hash_function> table;

    // Helper functions
    bool read_header(std::ifstream& file);
    bool read_ngram_entry(std::ifstream& file);
}; 