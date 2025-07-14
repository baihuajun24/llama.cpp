#pragma once

#include "llama.h"
#include "ngram-cache.h"

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

class SuffixArray {
public:
    SuffixArray();
    ~SuffixArray();

    // Loading and saving
    bool load_mmap(const std::string& filename);
    void unload_mmap();
    bool is_mmap_loaded() const { return mapped_data != nullptr; }
    
    // Building (for preprocessing)
    bool build_from_corpus(const std::vector<llama_token>& tokens);
    bool save(const std::string& filename);
    
    // Loading raw tokens (for preprocessing)
    bool load_raw_tokens(const std::string& filename);
    bool load_raw_tokens_with_header(const std::string& filename);
    
    // Complete build pipeline (preprocess + save)
    bool build_and_save(const std::string& input_file, const std::string& output_file);
    bool build_and_save_sampled(const std::string& input_file, const std::string& output_file,
                                double sample_fraction, int random_seed = 42);

    // Search operations
    std::pair<size_t, size_t> find_range(const std::vector<llama_token>& pattern) const;
    std::vector<llama_token> get_continuations(const std::vector<llama_token>& pattern, int max_continuations) const;
    std::vector<llama_token> get_coherent_draft(const std::vector<llama_token>& pattern, int n_draft) const;
    
    // Draft generation
    void draft_suffix(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
                      int n_draft, int min_n, int max_n);
    
    // Interleaved draft: first search in prompt history, then suffix array
    std::pair<int, char> interleave_draft(const std::vector<llama_token>& input, 
                                          std::vector<llama_token>& draft, 
                                          int n_draft, int min_n, int max_n);

    // Statistics
    struct Stats {
        size_t corpus_size;
        size_t file_size_bytes;
        size_t memory_usage_bytes;
    };
    Stats get_stats() const;

    // Debug function
    void debug_show_entries(int max_entries = 5) const;

private:
    static constexpr const char MAGIC[9] = "SUFFIXAR";
    static constexpr uint32_t VERSION = 1;

    // File header structure
    struct Header {
        char magic[8];
        uint32_t version;
        uint64_t corpus_size;
        uint64_t reserved[4];
    };

    // In-memory corpus and suffix array (for building)
    std::vector<llama_token> corpus;
    std::vector<size_t> suffix_array;

    // Memory-mapped data
    void* mapped_data;
    size_t file_size;
    const llama_token* corpus_data;    // Points to corpus in mapped file
    const size_t* suffix_data;         // Points to suffix array in mapped file
    size_t corpus_size;

    // Helper functions for file I/O
    bool read_header_mmap(const char* data, size_t& offset);
    bool write_header(std::ofstream& file);

    // Binary search functions
    int compare_suffix(size_t suffix_pos, const std::vector<llama_token>& pattern) const;
    size_t binary_search_left(const std::vector<llama_token>& pattern) const;
    size_t binary_search_right(const std::vector<llama_token>& pattern) const;

    // Suffix array construction
    void build_suffix_array();
    static bool suffix_compare(const std::vector<llama_token>& corpus, size_t a, size_t b);

    // Platform-specific memory mapping
    bool map_file(const std::string& filename);
    void unmap_file();

    // Prompt history search (for interleaved approach)
    std::pair<int, char> search_prompt_history(const std::vector<llama_token>& input, 
                                               std::vector<llama_token>& draft, 
                                               int n_draft, int min_n, int max_n);
};