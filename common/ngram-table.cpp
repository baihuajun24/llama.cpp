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

    // Read tokens
    std::vector<llama_token> tokens(n);
    file.read(reinterpret_cast<char*>(tokens.data()), n * sizeof(llama_token));

    // Create common_ngram from tokens
    common_ngram key(tokens.data(), n);

    // Read future token predictions
    uint32_t n_predictions;
    file.read(reinterpret_cast<char*>(&n_predictions), sizeof(n_predictions));

    std::vector<FutureToken> predictions(n_predictions);
    file.read(reinterpret_cast<char*>(predictions.data()), n_predictions * sizeof(FutureToken));

    // Store in table
    table.emplace(key, std::move(predictions));
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

        // Read n-gram entries
        uint32_t n_entries;
        file.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));

        for (uint32_t i = 0; i < n_entries; i++) {
            if (!read_ngram_entry(file)) {
                continue; // Skip invalid entries
            }
        }

        return true;
    } catch (const std::exception& e) {
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