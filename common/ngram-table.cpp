#include "ngram-table.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

NGramTable::NGramTable() : min_n(0), max_n(0), horizon(0) {}

bool NGramTable::read_header(std::ifstream& file) {
    char magic[9];
    file.read(magic, 8);
    magic[8] = '\0';
    if (std::strncmp(magic, MAGIC, 8) != 0) {
        throw std::runtime_error("Invalid magic bytes in ngram table file");
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != VERSION) {
        throw std::runtime_error("Unsupported version in ngram table file");
    }

    file.read(reinterpret_cast<char*>(&min_n), sizeof(min_n));
    file.read(reinterpret_cast<char*>(&max_n), sizeof(max_n));
    file.read(reinterpret_cast<char*>(&horizon), sizeof(horizon));

    return file.good();
}

bool NGramTable::read_ngram_entry(std::ifstream& file) {
    if (!file.good()) {
        return false;
    }

    // Read n-gram length
    uint32_t n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n > LLAMA_NGRAM_MAX || n == 0) {
        return false;
    }

    // Read tokens in the n-gram
    std::vector<llama_token> tokens(n);
    for (uint32_t i = 0; i < n; i++) {
        file.read(reinterpret_cast<char*>(&tokens[i]), sizeof(llama_token));
    }

    // Create common_ngram from tokens
    common_ngram key(tokens.data(), n);

    // Read frequency (this was missing in the original code!)
    uint32_t frequency;
    file.read(reinterpret_cast<char*>(&frequency), sizeof(frequency));

    // Read future tokens data for each horizon position
    std::vector<FutureToken> all_predictions;
    
    for (uint32_t pos = 0; pos < horizon; pos++) {
        // Read number of future tokens for this position
        uint32_t num_tokens;
        file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
        
        // Read each token and its frequency for this position
        for (uint32_t i = 0; i < num_tokens; i++) {
            llama_token token;
            uint32_t count;
            file.read(reinterpret_cast<char*>(&token), sizeof(token));
            file.read(reinterpret_cast<char*>(&count), sizeof(count));
            
            // Add to predictions list
            all_predictions.push_back({token, count});
        }
    }

    // Store in table
    table.emplace(key, std::move(all_predictions));
    return true;
}

bool NGramTable::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    try {
        if (!read_header(file)) {
            return false;
        }

        // Read table size (FIXED: was reading uint32_t, should be uint64_t!)
        uint64_t n_entries;
        file.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));

        for (uint64_t i = 0; i < n_entries; i++) {
            if (!read_ngram_entry(file)) {
                continue; // Skip invalid entries
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading ngram table: " << e.what() << std::endl;
        return false;
    }
}

void NGramTable::draft(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
                      int n_draft, int min_n, int max_n) {
    // Start with the largest possible n-gram and work down
    for (int n = std::min(max_n, static_cast<int>(input.size())); n >= min_n; n--) {
        if (static_cast<int>(input.size()) < n) continue;

        // Create common_ngram from the last n tokens
        common_ngram key(input.data() + input.size() - n, n);
        
        // Look up in table
        auto it = table.find(key);
        if (it != table.end()) {
            // Found a match - add predictions to draft
            const auto& predictions = it->second;
            for (const auto& pred : predictions) {
                if (static_cast<int>(draft.size()) >= n_draft + 1) break; // +1 because draft already has 1 token
                draft.push_back(pred.token);
            }
            break; // Stop after finding first match
        }
    }
}

NGramTable::Stats NGramTable::get_stats() const {
    return Stats{
        table.size(),
        min_n,
        max_n,
        horizon
    };
}

void NGramTable::debug_show_entries(int max_entries) const {
    std::cout << "\n=== NGramTable Debug Info ===" << std::endl;
    std::cout << "Total entries: " << table.size() << std::endl;
    std::cout << "Configuration: min_n=" << min_n << ", max_n=" << max_n << ", horizon=" << horizon << std::endl;
    
    int count = 0;
    for (const auto& entry : table) {
        if (count >= max_entries) break;
        
        const auto& ngram = entry.first;
        const auto& predictions = entry.second;
        
        // Print the n-gram tokens
        std::cout << "\nEntry " << (count + 1) << ": N-gram [";
        bool first = true;
        for (int i = 0; i < LLAMA_NGRAM_MAX; i++) {
            if (ngram.tokens[i] == LLAMA_TOKEN_NULL) break;
            if (!first) std::cout << ", ";
            std::cout << ngram.tokens[i];
            first = false;
        }
        std::cout << "]" << std::endl;
        
        // Print future token predictions
        std::cout << "  Future tokens (" << predictions.size() << "): ";
        for (size_t i = 0; i < std::min(predictions.size(), size_t(10)); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << predictions[i].token << ":" << predictions[i].frequency;
        }
        if (predictions.size() > 10) {
            std::cout << " ... (+" << (predictions.size() - 10) << " more)";
        }
        std::cout << std::endl;
        
        count++;
    }
    std::cout << "=========================" << std::endl;
} 