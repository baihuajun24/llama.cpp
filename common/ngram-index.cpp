#include "ngram-index.h"
#include "log.h"

#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>

ngram_index build_ngram_index(
    const std::vector<llama_token>& token_sequence,
    int ngram_min,
    int ngram_max,
    int max_indices_per_ngram,
    bool print_progress)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    ngram_index index;
    
    const int sequence_size = token_sequence.size();
    
    // Limit max_ngram to sequence size
    ngram_max = std::min(ngram_max, sequence_size - 1);
    
    LOG_INF("Building n-gram index for sequence of %d tokens (n-gram sizes %d-%d)\n", 
            sequence_size, ngram_min, ngram_max);
    
    size_t total_ngrams = 0;
    
    // For each n-gram size
    for (int n = ngram_min; n <= ngram_max; ++n) {
        int ngrams_for_this_size = 0;
        
        // For each position in the sequence where an n-gram can start
        for (int pos = 0; pos <= sequence_size - n; ++pos) {
            // Create n-gram key
            ngram_index_key key(&token_sequence[pos], n, ngram_max);
            
            // Find or create the positions vector for this key
            auto& positions = index[key];
            
            // Add the position (only if there's room)
            if (max_indices_per_ngram == 0 || positions.size() < (size_t)max_indices_per_ngram) {
                positions.push_back(pos);
                ngrams_for_this_size++;
            }
        }
        
        if (print_progress) {
            LOG_INF("  %d-grams: %d unique patterns\n", n, ngrams_for_this_size);
        }
        
        total_ngrams += ngrams_for_this_size;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    LOG_INF("Built n-gram index with %zu unique patterns in %.2f seconds\n", 
            index.size(), duration / 1000.0f);
    
    return index;
}

void save_ngram_index(ngram_index& index, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }
    
    // Write magic header and version
    const char magic[] = "NGRIDX";
    file.write(magic, 6);
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write number of entries
    uint64_t num_entries = index.size();
    file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    
    // For each entry in the index
    for (const auto& entry : index) {
        const ngram_index_key& key = entry.first;
        const ngram_index_positions& positions = entry.second;
        
        // Write key size
        uint32_t key_size = key.size;
        file.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        
        // Write key tokens
        for (int i = 0; i < key.size; ++i) {
            int32_t token = key.tokens[i];
            file.write(reinterpret_cast<const char*>(&token), sizeof(token));
        }
        
        // Write number of positions
        uint32_t num_positions = positions.size();
        file.write(reinterpret_cast<const char*>(&num_positions), sizeof(num_positions));
        
        // Write positions
        for (int pos : positions) {
            int32_t position = pos;
            file.write(reinterpret_cast<const char*>(&position), sizeof(position));
        }
    }
    
    LOG_INF("Saved n-gram index to %s (%zu entries)\n", filename.c_str(), num_entries);
}

ngram_index load_ngram_index(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG_ERR("Failed to open file for reading: %s\n", filename.c_str());
        return ngram_index();
    }
    
    ngram_index index;
    
    // Read and verify magic header
    char magic[7] = {0};
    file.read(magic, 6);
    if (std::strcmp(magic, "NGRIDX") != 0) {
        LOG_ERR("Invalid file format: %s\n", filename.c_str());
        return index;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        LOG_ERR("Unsupported version %u in file: %s\n", version, filename.c_str());
        return index;
    }
    
    // Read number of entries
    uint64_t num_entries;
    file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    
    // For each entry
    for (uint64_t i = 0; i < num_entries; ++i) {
        // Read key size
        uint32_t key_size;
        file.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        
        // Create key
        ngram_index_key key(key_size);
        
        // Read key tokens
        for (uint32_t j = 0; j < key_size; ++j) {
            int32_t token;
            file.read(reinterpret_cast<char*>(&token), sizeof(token));
            key.tokens[j] = token;
        }
        
        // Read number of positions
        uint32_t num_positions;
        file.read(reinterpret_cast<char*>(&num_positions), sizeof(num_positions));
        
        // Read positions
        ngram_index_positions positions;
        positions.reserve(num_positions);
        
        for (uint32_t j = 0; j < num_positions; ++j) {
            int32_t position;
            file.read(reinterpret_cast<char*>(&position), sizeof(position));
            positions.push_back(position);
        }
        
        // Add to index
        index[key] = std::move(positions);
    }
    
    LOG_INF("Loaded n-gram index from %s (%zu entries)\n", filename.c_str(), index.size());
    
    return index;
}

void draft_with_ngram_index(
    const std::vector<llama_token>& inp,
    std::vector<llama_token>& draft,
    int n_draft,
    const std::vector<llama_token>& prompt,
    const ngram_index& index,
    int ngram_min,
    int ngram_max,
    int selection_strategy) 
{
    // Add debugging
    LOG_DBG("Starting draft_with_ngram_index: inp.size=%zu, draft.size=%zu, prompt.size=%zu\n", 
            inp.size(), draft.size(), prompt.size());
    
    // Validate input 
    if (inp.empty() || draft.empty() || prompt.empty()) {
        LOG_DBG("Empty input sequences, cannot draft\n");
        return;
    }
    
    // Start with the largest n-gram size and work down
    for (int n = ngram_max; n >= ngram_min; --n) {
        // Get the last n tokens from inp
        if ((int)inp.size() < n) {
            LOG_DBG("Input too short for %d-gram context\n", n);
            continue;
        }
        
        std::vector<llama_token> context;
        context.reserve(n);
        
        // Get tokens from inp
        for (int i = inp.size() - n; i < (int)inp.size(); ++i) {
            context.push_back(inp[i]);
        }
        
        // Create key from context
        ngram_index_key key(context.data(), n, ngram_max);
        
        // Debug info about context
        std::string context_str = "";
        for (auto t : context) {
            context_str += std::to_string(t) + " ";
        }
        LOG_DBG("Looking up %d-gram context: %s\n", n, context_str.c_str());
        
        // Look up in index
        auto it = index.find(key);
        if (it == index.end()) {
            LOG_DBG("No match found for this context\n");
            continue;  // No match, try smaller n-gram
        }
        
        const ngram_index_positions& positions = it->second;
        if (positions.empty()) {
            LOG_DBG("Empty positions list for this context\n");
            continue;  // No positions, try smaller n-gram
        }
        
        LOG_DBG("Found %zu positions for this context\n", positions.size());
        
        // Select a position based on strategy
        int selected_pos;
        if (selection_strategy == 1) {  // Random
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, positions.size() - 1);
            selected_pos = positions[dis(gen)];
        } else {  // First
            selected_pos = positions[0];
        }
        
        LOG_DBG("Selected position: %d\n", selected_pos);
        
        // Continuation position
        int continuation_pos = selected_pos + n;
        
        // Check if there's enough tokens after this position
        if (continuation_pos >= (int)prompt.size()) {
            LOG_DBG("Not enough tokens after position %d\n", continuation_pos);
            continue;  // Not enough tokens, try smaller n-gram
        }
        
        // Ensure first draft token matches what we already have
        if (draft.size() == 1 && prompt[continuation_pos] != draft[0]) {
            LOG_DBG("First token mismatch: prompt[%d]=%d, draft[0]=%d\n", 
                    continuation_pos, prompt[continuation_pos], draft[0]);
            continue;  // First token doesn't match, try smaller n-gram
        }
        
        // Clear existing draft tokens after the first one
        if (draft.size() > 1) {
            draft.resize(1);
        }
        
        // Add tokens from the continuation
        int tokens_to_add = std::min(n_draft, (int)prompt.size() - continuation_pos);
        
        for (int i = 0; i < tokens_to_add; ++i) {
            // If we're at the start, skip the first token as it's already in draft
            if (i == 0 && draft.size() == 1) {
                continue;
            }
            
            draft.push_back(prompt[continuation_pos + i]);
        }
        
        LOG_DBG("Found continuation at position %d with n-gram size %d, added %zu tokens\n", 
                selected_pos, n, draft.size() - 1);
        
        // Successfully found and added continuation
        return;
    }
    
    // If we got here, didn't find any matching n-grams
    LOG_DBG("No matching n-grams found for drafting\n");
}

void print_ngram_index_stats(const ngram_index& index) {
    // Count n-grams by size
    std::unordered_map<int, int> sizes;
    size_t total_positions = 0;
    int max_positions = 0;
    std::string most_frequent_pattern;
    
    for (const auto& entry : index) {
        const ngram_index_key& key = entry.first;
        const ngram_index_positions& positions = entry.second;
        
        // Count actual tokens in key (excluding nulls)
        int actual_size = 0;
        for (int i = 0; i < key.size; ++i) {
            if (key.tokens[i] != LLAMA_TOKEN_NULL) {
                actual_size++;
            }
        }
        
        sizes[actual_size]++;
        total_positions += positions.size();
        
        if ((int)positions.size() > max_positions) {
            max_positions = positions.size();
            
            // Create a string representation of the most frequent pattern
            most_frequent_pattern = "";
            for (int i = 0; i < key.size; ++i) {
                if (key.tokens[i] != LLAMA_TOKEN_NULL) {
                    most_frequent_pattern += std::to_string(key.tokens[i]) + " ";
                }
            }
        }
    }
    
    LOG_INF("N-gram index statistics:\n");
    LOG_INF("  Total unique patterns: %zu\n", index.size());
    LOG_INF("  Total positions: %zu\n", total_positions);
    LOG_INF("  Average positions per pattern: %.2f\n", 
            index.size() > 0 ? (float)total_positions / index.size() : 0);
    LOG_INF("  Maximum positions for a pattern: %d\n", max_positions);
    LOG_INF("  Most frequent pattern: %s\n", most_frequent_pattern.c_str());
    
    LOG_INF("  Patterns by n-gram size:\n");
    for (const auto& s : sizes) {
        LOG_INF("    %d-grams: %d patterns\n", s.first, s.second);
    }
}