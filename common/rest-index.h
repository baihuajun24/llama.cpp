// rest-index.h
#ifndef REST_INDEX_H
#define REST_INDEX_H

#include <vector>
#include <string>
#include "llama.h"

// Forward declare Python types
struct _object;
typedef _object PyObject;

class RESTIndex {
private:
    PyObject* py_module;
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
    
    // Search for continuations given a prefix
    std::vector<llama_token> search(const std::vector<llama_token>& prefix, int choices = 64);
    
    // Check if an index is loaded
    bool isLoaded() const;
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