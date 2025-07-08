#include "ngram-table.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Platform-specific includes for memory mapping
#ifdef _WIN32
#define NOMINMAX  // Prevent Windows from defining min/max macros
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

NGramTable::NGramTable() : min_n(0), max_n(0), horizon(0), mapped_data(nullptr), file_size(0) {}

NGramTable::~NGramTable() {
    unload_mmap();
}

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

// Platform-specific memory mapping functions
bool NGramTable::map_file(const std::string& filename) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, 
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }
    file_size = fileSize.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        return false;
    }

    mapped_data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    
    CloseHandle(hMapping);
    CloseHandle(hFile);
    
    return mapped_data != nullptr;
#else
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        return false;
    }
    file_size = st.st_size;

    mapped_data = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    
    return mapped_data != MAP_FAILED;
#endif
}

void NGramTable::unmap_file() {
    if (mapped_data) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data);
#else
        munmap(mapped_data, file_size);
#endif
        mapped_data = nullptr;
        file_size = 0;
    }
}

// Memory-mapped header reading
bool NGramTable::read_header_mmap(const char* data, size_t& offset) {
    if (offset + 8 > file_size) return false;
    
    // Check magic
    if (std::strncmp(data + offset, MAGIC, 8) != 0) {
        return false;
    }
    offset += 8;
    
    if (offset + sizeof(uint32_t) > file_size) return false;
    uint32_t version = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    
    if (version != VERSION) return false;
    
    if (offset + 3 * sizeof(uint32_t) > file_size) return false;
    min_n = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    max_n = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    horizon = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t);
    
    return true;
}

// Build index from memory-mapped data
bool NGramTable::build_mmap_index() {
    if (!mapped_data || file_size == 0) return false;
    
    const char* data = static_cast<const char*>(mapped_data);
    size_t offset = 0;
    
    // Read header
    if (!read_header_mmap(data, offset)) return false;
    
    // Read table size
    if (offset + sizeof(uint64_t) > file_size) return false;
    uint64_t n_entries = *reinterpret_cast<const uint64_t*>(data + offset);
    offset += sizeof(uint64_t);
    
    // Build index
    for (uint64_t i = 0; i < n_entries; i++) {
        size_t entry_start = offset;
        
        // Read n-gram length
        if (offset + sizeof(uint32_t) > file_size) return false;
        uint32_t n = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        if (n > LLAMA_NGRAM_MAX || n == 0) return false;
        
        // Read tokens
        if (offset + n * sizeof(llama_token) > file_size) return false;
        const llama_token* tokens = reinterpret_cast<const llama_token*>(data + offset);
        offset += n * sizeof(llama_token);
        
        // Create key
        common_ngram key(tokens, n);
        
        // Store offset in index
        mmap_index[key] = entry_start;
        
        // Skip the rest of the entry (frequency + future tokens)
        if (offset + sizeof(uint32_t) > file_size) return false;
        offset += sizeof(uint32_t); // frequency
        
        // Skip future tokens for each horizon position
        for (uint32_t pos = 0; pos < horizon; pos++) {
            if (offset + sizeof(uint32_t) > file_size) return false;
            uint32_t num_tokens = *reinterpret_cast<const uint32_t*>(data + offset);
            offset += sizeof(uint32_t);
            
            // Skip tokens and their counts
            offset += num_tokens * (sizeof(llama_token) + sizeof(uint32_t));
            if (offset > file_size) return false;
        }
    }
    
    return true;
}

// Memory-mapped lookup
std::vector<FutureToken> NGramTable::lookup_mmap(const common_ngram& key) {
    auto it = mmap_index.find(key);
    if (it == mmap_index.end()) {
        return {};
    }
    
    const char* data = static_cast<const char*>(mapped_data);
    size_t offset = it->second;
    
    // Skip n-gram length and tokens
    if (offset + sizeof(uint32_t) > file_size) return {};
    uint32_t n = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += sizeof(uint32_t) + n * sizeof(llama_token);
    
    // Skip frequency
    offset += sizeof(uint32_t);
    
    // Read future tokens
    std::vector<FutureToken> predictions;
    for (uint32_t pos = 0; pos < horizon; pos++) {
        if (offset + sizeof(uint32_t) > file_size) return {};
        uint32_t num_tokens = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        for (uint32_t i = 0; i < num_tokens; i++) {
            if (offset + sizeof(llama_token) + sizeof(uint32_t) > file_size) return {};
            llama_token token = *reinterpret_cast<const llama_token*>(data + offset);
            offset += sizeof(llama_token);
            uint32_t count = *reinterpret_cast<const uint32_t*>(data + offset);
            offset += sizeof(uint32_t);
            
            predictions.push_back({token, count});
        }
    }
    
    return predictions;
}

// Main memory-mapped loading function
bool NGramTable::load_mmap(const std::string& filename) {
    // Clean up any existing mapping
    unload_mmap();
    
    // Map the file
    if (!map_file(filename)) {
        return false;
    }
    
    // Build index
    if (!build_mmap_index()) {
        unload_mmap();
        return false;
    }
    
    return true;
}

void NGramTable::unload_mmap() {
    mmap_index.clear();
    unmap_file();
}

// Memory-mapped draft function
void NGramTable::draft_mmap(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
                           int n_draft, int min_n, int max_n) {
    if (!is_mmap_loaded()) return;
    
    // Start with the largest possible n-gram and work down
    for (int n = std::min(max_n, static_cast<int>(input.size())); n >= min_n; n--) {
        if (static_cast<int>(input.size()) < n) continue;

        // Create common_ngram from the last n tokens
        common_ngram key(input.data() + input.size() - n, n);
        
        // Look up in memory-mapped data
        std::vector<FutureToken> predictions = lookup_mmap(key);
        if (!predictions.empty()) {
            // Found a match - add predictions to draft
            for (const auto& pred : predictions) {
                if (static_cast<int>(draft.size()) >= n_draft) break;
                draft.push_back(pred.token);
            }
            break; // Stop after finding first match
        }
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

std::pair<int, char> NGramTable::interleave_draft(const std::vector<llama_token>& input, 
                                                  std::vector<llama_token>& draft, 
                                                  int n_draft, int min_n, int max_n) {
    const int inp_size = input.size();
    
    if (inp_size < min_n) {
        return {0, 'X'};
    }

    // Create search space from prompt history (excluding last min_n tokens to avoid overlap)
    std::vector<llama_token> search_space;
    if (inp_size > min_n) {
        search_space = std::vector<llama_token>(input.begin(), input.end() - min_n);
    }

    const int max_n_actual = std::min(max_n, inp_size);

    for (int n = max_n_actual; n >= min_n; --n) {
        if (inp_size < n) continue;

        // First, try to find match in prompt history (search_space)
        if (!search_space.empty() && (int)search_space.size() >= n) {
            // Take the last n tokens of input as query
            std::vector<llama_token> suffix(input.end() - n, input.end());
            
            for (int i = 0; i <= (int)search_space.size() - n; ++i) {
                bool match = true;
                for (int j = 0; j < n; ++j) {
                    if (search_space[i + j] != suffix[j]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    // Draft up to n_draft tokens after the match from prompt history
                    for (int j = 0; j < n_draft && (i + n + j) < (int)search_space.size(); ++j) {
                        draft.push_back(search_space[i + n + j]);
                    }
                    return {n, 'P'}; // Found in prompt
                }
            }
        }

        // If not found in prompt history, try static ngram table
        // Create common_ngram from the last n tokens
        common_ngram key(input.data() + input.size() - n, n);
        
        // Look up in table
        auto it = table.find(key);
        if (it != table.end()) {
            // Found a match - add predictions to draft
            const auto& predictions = it->second;
            for (const auto& pred : predictions) {
                if (static_cast<int>(draft.size()) >= n_draft) break;
                draft.push_back(pred.token);
            }
            return {n, 'S'}; // Found in static table
        }
    }

    return {0, 'X'}; // No match found in either source
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