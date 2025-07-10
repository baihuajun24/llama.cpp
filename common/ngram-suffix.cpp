#include "ngram-suffix.h"
#include "llama.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <set>

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

SuffixArray::SuffixArray() : mapped_data(nullptr), file_size(0), corpus_data(nullptr), 
                             suffix_data(nullptr), corpus_size(0) {}

SuffixArray::~SuffixArray() {
    unload_mmap();
}

bool SuffixArray::build_from_corpus(const std::vector<llama_token>& tokens) {
    corpus = tokens;
    corpus_size = tokens.size();
    
    std::cout << "Building suffix array from corpus of " << corpus_size << " tokens..." << std::endl;
    build_suffix_array();
    std::cout << "Suffix array construction complete." << std::endl;
    
    return true;
}

void SuffixArray::build_suffix_array() {
    suffix_array.clear();
    suffix_array.reserve(corpus_size);
    
    // Initialize suffix array with all positions
    for (size_t i = 0; i < corpus_size; i++) {
        suffix_array.push_back(i);
    }
    
    // Sort suffixes using custom comparator
    std::sort(suffix_array.begin(), suffix_array.end(), 
              [this](size_t a, size_t b) { return suffix_compare(corpus, a, b); });
}

bool SuffixArray::suffix_compare(const std::vector<llama_token>& corpus, size_t a, size_t b) {
    const size_t size = corpus.size();
    while (a < size && b < size) {
        if (corpus[a] < corpus[b]) return true;
        if (corpus[a] > corpus[b]) return false;
        a++;
        b++;
    }
    return a >= size && b < size;  // Shorter suffix comes first
}

bool SuffixArray::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        return false;
    }

    if (!write_header(file)) {
        return false;
    }

    // Write corpus
    file.write(reinterpret_cast<const char*>(corpus.data()), 
               corpus.size() * sizeof(llama_token));
    
    // Write suffix array
    file.write(reinterpret_cast<const char*>(suffix_array.data()), 
               suffix_array.size() * sizeof(size_t));

    if (!file.good()) {
        std::cerr << "Error writing suffix array data" << std::endl;
        return false;
    }

    std::cout << "Saved suffix array to " << filename 
              << " (corpus: " << corpus.size() << " tokens, "
              << "file size: " << file.tellp() << " bytes)" << std::endl;
    
    return true;
}

bool SuffixArray::write_header(std::ofstream& file) {
    Header header = {};
    std::strncpy(header.magic, MAGIC, 8);
    header.version = VERSION;
    header.corpus_size = corpus_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    return file.good();
}

// Platform-specific memory mapping functions
bool SuffixArray::map_file(const std::string& filename) {
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

void SuffixArray::unmap_file() {
    if (mapped_data) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data);
#else
        munmap(mapped_data, file_size);
#endif
        mapped_data = nullptr;
        file_size = 0;
        corpus_data = nullptr;
        suffix_data = nullptr;
    }
}

bool SuffixArray::read_header_mmap(const char* data, size_t& offset) {
    if (offset + sizeof(Header) > file_size) return false;
    
    const Header* header = reinterpret_cast<const Header*>(data + offset);
    offset += sizeof(Header);
    
    // Check magic
    if (std::strncmp(header->magic, MAGIC, 8) != 0) {
        std::cerr << "Invalid magic bytes in suffix array file" << std::endl;
        return false;
    }
    
    if (header->version != VERSION) {
        std::cerr << "Unsupported version in suffix array file" << std::endl;
        return false;
    }
    
    corpus_size = header->corpus_size;
    return true;
}

bool SuffixArray::load_mmap(const std::string& filename) {
    // Clean up any existing mapping
    unload_mmap();
    
    // Map the file
    if (!map_file(filename)) {
        std::cerr << "Failed to map file: " << filename << std::endl;
        return false;
    }
    
    const char* data = static_cast<const char*>(mapped_data);
    size_t offset = 0;
    
    // Read header
    if (!read_header_mmap(data, offset)) {
        unload_mmap();
        return false;
    }
    
    // Set up pointers to corpus and suffix array in mapped memory
    corpus_data = reinterpret_cast<const llama_token*>(data + offset);
    offset += corpus_size * sizeof(llama_token);
    
    suffix_data = reinterpret_cast<const size_t*>(data + offset);
    
    std::cout << "Loaded suffix array from " << filename 
              << " (corpus: " << corpus_size << " tokens)" << std::endl;
    
    return true;
}

void SuffixArray::unload_mmap() {
    unmap_file();
    corpus_size = 0;
}

int SuffixArray::compare_suffix(size_t suffix_pos, const std::vector<llama_token>& pattern) const {
    const size_t pattern_size = pattern.size();
    
    for (size_t i = 0; i < pattern_size; i++) {
        if (suffix_pos + i >= corpus_size) {
            return -1;  // Suffix is shorter than pattern
        }
        
        llama_token corpus_token = corpus_data[suffix_pos + i];
        llama_token pattern_token = pattern[i];
        
        if (corpus_token < pattern_token) return -1;
        if (corpus_token > pattern_token) return 1;
    }
    
    return 0;  // Perfect match
}

size_t SuffixArray::binary_search_left(const std::vector<llama_token>& pattern) const {
    size_t left = 0, right = corpus_size;
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        size_t suffix_pos = suffix_data[mid];
        
        if (compare_suffix(suffix_pos, pattern) < 0) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

size_t SuffixArray::binary_search_right(const std::vector<llama_token>& pattern) const {
    size_t left = 0, right = corpus_size;
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        size_t suffix_pos = suffix_data[mid];
        
        if (compare_suffix(suffix_pos, pattern) <= 0) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

std::pair<size_t, size_t> SuffixArray::find_range(const std::vector<llama_token>& pattern) const {
    if (!is_mmap_loaded() || pattern.empty()) {
        return {0, 0};
    }
    
    size_t left_bound = binary_search_left(pattern);
    size_t right_bound = binary_search_right(pattern);
    
    return {left_bound, right_bound};
}

std::vector<llama_token> SuffixArray::get_continuations(const std::vector<llama_token>& pattern, 
                                                        int max_continuations) const {
    auto [left, right] = find_range(pattern);
    
    std::vector<llama_token> continuations;
    std::set<llama_token> seen_tokens;  // Avoid duplicates
    
    const size_t pattern_len = pattern.size();
    
    for (size_t i = left; i < right && (int)continuations.size() < max_continuations; i++) {
        size_t suffix_pos = suffix_data[i];
        
        // Check if there's a continuation token after the pattern
        if (suffix_pos + pattern_len < corpus_size) {
            llama_token next_token = corpus_data[suffix_pos + pattern_len];
            
            // Add if not seen before
            if (seen_tokens.find(next_token) == seen_tokens.end()) {
                seen_tokens.insert(next_token);
                continuations.push_back(next_token);
            }
        }
    }
    
    return continuations;
}

std::vector<llama_token> SuffixArray::get_coherent_draft(const std::vector<llama_token>& pattern, int n_draft) const {
    auto [left, right] = find_range(pattern);
    
    if (left >= right) {
        return {};  // No matches found
    }
    
    // Take the first match found (fastest approach)
    size_t suffix_pos = suffix_data[left];
    size_t start_pos = suffix_pos + pattern.size();
    
    std::vector<llama_token> draft;
    
    // Draft consecutive tokens from this one match
    for (int i = 0; i < n_draft && start_pos + i < corpus_size; i++) {
        draft.push_back(corpus_data[start_pos + i]);
    }
    
    return draft;
}

void SuffixArray::draft_suffix(const std::vector<llama_token>& input, std::vector<llama_token>& draft, 
                               int n_draft, int min_n, int max_n) {
    if (!is_mmap_loaded()) return;
    
    // Try n-grams from max_n down to min_n
    for (int n = std::min(max_n, (int)input.size()); n >= min_n; n--) {
        if ((int)input.size() < n) continue;
        
        // Extract last n tokens as query
        std::vector<llama_token> query(input.end() - n, input.end());
        
        // Get continuations
        std::vector<llama_token> continuations = get_continuations(query, n_draft);
        
        if (!continuations.empty()) {
            // Add continuations to draft
            for (const auto& token : continuations) {
                if ((int)draft.size() >= n_draft) break;
                draft.push_back(token);
            }
            return; // Stop at first match (longest n-gram)
        }
    }
}

std::pair<int, char> SuffixArray::search_prompt_history(const std::vector<llama_token>& input, 
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

        // First, try to find match in prompt history
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
    }

    return {0, 'X'}; // No match found in prompt history
}

std::pair<int, char> SuffixArray::interleave_draft(const std::vector<llama_token>& input, 
                                                   std::vector<llama_token>& draft, 
                                                   int n_draft, int min_n, int max_n) {
    // First, try prompt history search
    auto prompt_result = search_prompt_history(input, draft, n_draft, min_n, max_n);
    if (prompt_result.second == 'P') {
        return prompt_result;  // Found in prompt history
    }

    // If not found in prompt, try suffix array
    const int inp_size = input.size();
    const int max_n_actual = std::min(max_n, inp_size);

    for (int n = max_n_actual; n >= min_n; --n) {
        if (inp_size < n) continue;

        // Extract last n tokens as query
        std::vector<llama_token> query(input.end() - n, input.end());
        
        // Get continuations from suffix array
        std::vector<llama_token> continuations = get_continuations(query, n_draft);
        
        if (!continuations.empty()) {
            // Add continuations to draft
            for (const auto& token : continuations) {
                if ((int)draft.size() >= n_draft) break;
                draft.push_back(token);
            }
            return {n, 'S'}; // Found in suffix array
        }
    }
    
    return {0, 'X'}; // No match found
}

SuffixArray::Stats SuffixArray::get_stats() const {
    Stats stats = {};
    stats.corpus_size = corpus_size;
    stats.file_size_bytes = file_size;
    stats.memory_usage_bytes = is_mmap_loaded() ? file_size : 
                               (corpus.size() * sizeof(llama_token) + suffix_array.size() * sizeof(size_t));
    return stats;
}

void SuffixArray::debug_show_entries(int max_entries) const {
    if (!is_mmap_loaded()) {
        std::cout << "Suffix array not loaded" << std::endl;
        return;
    }
    
    std::cout << "=== First " << max_entries << " suffix array entries ===" << std::endl;
    for (int i = 0; i < std::min(max_entries, (int)corpus_size); i++) {
        size_t suffix_pos = suffix_data[i];
        std::cout << "Entry " << i << ": suffix_pos=" << suffix_pos << ", tokens: ";
        
        // Show first 5 tokens of this suffix
        for (int j = 0; j < 5 && suffix_pos + j < corpus_size; j++) {
            std::cout << corpus_data[suffix_pos + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool SuffixArray::load_raw_tokens(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open raw token file: " << filename << std::endl;
        return false;
    }
    
    // Get file size
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of tokens
    size_t num_tokens = file_size / sizeof(llama_token);
    std::cout << "Reading " << num_tokens << " tokens from " << filename << std::endl;
    
    // Read all tokens
    corpus.resize(num_tokens);
    if (!file.read(reinterpret_cast<char*>(corpus.data()), file_size)) {
        std::cerr << "Failed to read token data" << std::endl;
        return false;
    }
    
    corpus_size = num_tokens;
    std::cout << "Successfully loaded " << corpus_size << " tokens" << std::endl;
    return true;
}

bool SuffixArray::load_raw_tokens_with_header(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open token file with header: " << filename << std::endl;
        return false;
    }
    
    // Try to read a potential header
    struct TokenFileHeader {
        char magic[8];
        uint64_t num_tokens;
        uint64_t reserved[4];
    };
    
    TokenFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Check if this looks like a header
    bool has_header = (std::strncmp(header.magic, "LLMATOKN", 8) == 0);
    
    if (!has_header) {
        // No header, treat as raw tokens
        file.seekg(0, std::ios::beg);
        return load_raw_tokens(filename);
    }
    
    // Has header
    std::cout << "Found header, reading " << header.num_tokens << " tokens" << std::endl;
    
    corpus.resize(header.num_tokens);
    if (!file.read(reinterpret_cast<char*>(corpus.data()), 
                   header.num_tokens * sizeof(llama_token))) {
        std::cerr << "Failed to read token data" << std::endl;
        return false;
    }
    
    corpus_size = header.num_tokens;
    std::cout << "Successfully loaded " << corpus_size << " tokens" << std::endl;
    return true;
}

bool SuffixArray::build_and_save(const std::string& input_file, const std::string& output_file) {
    std::cout << "=== Building Suffix Array ===" << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    
    // Step 1: Load raw tokens
    if (!load_raw_tokens_with_header(input_file)) {
        return false;
    }
    
    // Step 2: Build suffix array
    if (!build_from_corpus(corpus)) {
        return false;
    }
    
    // Step 3: Save to file
    if (!save(output_file)) {
        return false;
    }
    
    std::cout << "=== Build Complete ===" << std::endl;
    return true;
}