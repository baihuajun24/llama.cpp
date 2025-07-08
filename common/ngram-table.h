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
    ~NGramTable();

    // Original in-memory loading
    bool load(const std::string& filename);

    // Memory-mapped file loading
    bool load_mmap(const std::string& filename);
    void unload_mmap();
    bool is_mmap_loaded() const { return mapped_data != nullptr; }

    // Draft next tokens based on input sequence
    void draft(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
               int n_draft, int min_n, int max_n);

    // Memory-mapped draft
    void draft_mmap(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
                    int n_draft, int min_n, int max_n);

    // Interleaved draft: first search in prompt history, then static table
    std::pair<int, char> interleave_draft(const std::vector<llama_token>& input, 
                                          std::vector<llama_token>& draft, 
                                          int n_draft, int min_n, int max_n);

    // Get table statistics
    struct Stats {
        size_t total_ngrams;
        uint32_t min_n;
        uint32_t max_n;
        uint32_t horizon;
    };
    Stats get_stats() const;

    // Debug function to show sample entries
    void debug_show_entries(int max_entries = 5) const;

private:
    static constexpr const char MAGIC[9] = "NGRAMTBL";
    static constexpr uint32_t VERSION = 1;
    
    uint32_t min_n;
    uint32_t max_n;
    uint32_t horizon;

    // Map from n-gram to its future token predictions
    std::unordered_map<common_ngram, std::vector<FutureToken>, common_ngram_hash_function> table;

    // Memory-mapped data
    void* mapped_data;
    size_t file_size;
    std::unordered_map<common_ngram, size_t, common_ngram_hash_function> mmap_index; // ngram -> file offset

    // Helper functions
    bool read_header(std::ifstream& file);
    bool read_ngram_entry(std::ifstream& file);

    // Memory-mapped helper functions
    bool read_header_mmap(const char* data, size_t& offset);
    bool build_mmap_index();
    std::vector<FutureToken> lookup_mmap(const common_ngram& key);
    
    // Platform-specific memory mapping
    bool map_file(const std::string& filename);
    void unmap_file();
}; 