// rest-index.h
#ifndef REST_INDEX_H
#define REST_INDEX_H

#include <vector>
#include <string>
#include "llama.h"
#include "draftretriever/draftretriever.h" // Include the DraftRetriever header

// Forward declare Python types
struct _object;
typedef _object PyObject;

// Define a structure to hold a candidate sequence
struct RESTCandidate {
    std::vector<llama_token> tokens;
    // Could add score or other metadata here if needed
};

class RESTIndex {
private:
    // Change from rust::Box<Reader> to a pointer
    rust::Box<Reader>* reader;
    bool loaded;

public:
    RESTIndex();
    ~RESTIndex();
    
    // Prevent copying
    RESTIndex(const RESTIndex&) = delete;
    RESTIndex& operator=(const RESTIndex&) = delete;
    
    // Create and save an index from tokens (not implemented)
    bool buildIndex(const std::vector<llama_token>& tokens, const std::string& indexPath);
    
    // Load an existing index
    bool loadIndex(const std::string& indexPath);
    
    // Search for continuations given a prefix - return multiple candidates
    std::vector<RESTCandidate> searchCandidates(const std::vector<llama_token>& prefix, int choices = 64);
    
    // Original search that returns a single vector (for backward compatibility)
    std::vector<llama_token> search(const std::vector<llama_token>& prefix, int choices = 64);
    
    // Check if an index is loaded
    bool isLoaded() const;
    
    // Helper to print token information for candidates
    void printCandidates(const std::vector<RESTCandidate>& candidates, 
                         llama_context* ctx, int max_candidates = 3, int max_tokens = 20);
};

// Function to draft tokens using REST approach
void rest_draft(
    const std::vector<llama_token>& inp,
    std::vector<llama_token>& draft,
    int n_draft,
    RESTIndex& restIndex,
    int max_prefix_len = 6,
    int min_prefix_len = 2
);

#endif // REST_INDEX_H