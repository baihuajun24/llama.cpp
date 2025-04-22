// rest-index.cpp
#include "rest-index.h"
#include "log.h"
#include <stdexcept>
#include <Python.h>
#include <iostream>

// Helper function to convert C++ vector to Python list
PyObject* vectorToList(const std::vector<llama_token>& vec) {
    PyObject* list = PyList_New(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong((long)vec[i]));
    }
    return list;
}

// Helper function to convert Python list to C++ vector
std::vector<llama_token> listToVector(PyObject* list) {
    std::vector<llama_token> vec;
    if (!PyList_Check(list)) return vec;
    
    Py_ssize_t size = PyList_Size(list);
    vec.reserve(size);
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(list, i);
        vec.push_back((llama_token)PyLong_AsLong(item));
    }
    
    return vec;
}

RESTIndex::RESTIndex() : py_module(nullptr), loaded(false) {
    // Initialize Python interpreter if not already running
    if (!Py_IsInitialized()) {
        Py_Initialize();
        PyRun_SimpleString("import sys; sys.path.append('.')"); // Add current directory to path
    }
    
    // Import the bridge module
    PyObject* moduleName = PyUnicode_FromString("rest_bridge");
    py_module = PyImport_Import(moduleName);
    Py_DECREF(moduleName);
    
    if (!py_module) {
        LOG_ERR("Failed to import rest_bridge module. Make sure rest_bridge.py is in your path.\n");
        PyErr_Print();
    }
}

RESTIndex::~RESTIndex() {
    Py_XDECREF(py_module);
}

bool RESTIndex::buildIndex(const std::vector<llama_token>& tokens, const std::string& indexPath) {
    LOG_ERR("Building index is not implemented for Python bridge. Please build index using Python directly.\n");
    return false;
}

bool RESTIndex::loadIndex(const std::string& indexPath) {
    if (!py_module) {
        LOG_ERR("Python bridge module not loaded\n");
        return false;
    }
    
    // Get the load_index function
    PyObject* loadFunc = PyObject_GetAttrString(py_module, "load_index");
    if (!loadFunc || !PyCallable_Check(loadFunc)) {
        LOG_ERR("Cannot find function 'load_index' in rest_bridge module\n");
        Py_XDECREF(loadFunc);
        return false;
    }
    
    // Call the function with the path
    PyObject* pathObj = PyUnicode_FromString(indexPath.c_str());
    PyObject* result = PyObject_CallFunctionObjArgs(loadFunc, pathObj, NULL);
    Py_DECREF(pathObj);
    Py_DECREF(loadFunc);
    
    if (!result) {
        LOG_ERR("Error calling load_index function\n");
        PyErr_Print();
        return false;
    }
    
    // Check if it returned True
    loaded = PyObject_IsTrue(result);
    Py_DECREF(result);
    
    LOG_INF("REST index %s loaded: %s\n", indexPath.c_str(), loaded ? "success" : "failed");
    return loaded;
}

std::vector<llama_token> RESTIndex::search(const std::vector<llama_token>& prefix, int choices) {
    if (!py_module || !loaded) {
        return {};
    }
    
    // Get the search function
    PyObject* searchFunc = PyObject_GetAttrString(py_module, "search");
    if (!searchFunc || !PyCallable_Check(searchFunc)) {
        LOG_ERR("Cannot find function 'search' in rest_bridge module\n");
        Py_XDECREF(searchFunc);
        return {};
    }
    
    // Convert prefix to Python list
    PyObject* prefixList = vectorToList(prefix);
    
    // Call the function
    PyObject* choicesObj = PyLong_FromLong(choices);
    PyObject* resultObj = PyObject_CallFunctionObjArgs(searchFunc, prefixList, choicesObj, NULL);
    Py_DECREF(prefixList);
    Py_DECREF(choicesObj);
    Py_DECREF(searchFunc);
    
    if (!resultObj) {
        LOG_ERR("Error calling search function\n");
        PyErr_Print();
        return {};
    }
    
    // Convert result to vector
    std::vector<llama_token> result = listToVector(resultObj);
    Py_DECREF(resultObj);
    
    LOG_DBG("REST search found %zu continuation tokens\n", result.size());
    return result;
}

bool RESTIndex::isLoaded() const {
    return loaded && py_module != nullptr;
}

void rest_draft(
    const std::vector<llama_token>& inp,
    std::vector<llama_token>& draft,
    int n_draft,
    RESTIndex& restIndex,
    int max_prefix_len,
    int min_prefix_len
) {
    // Validate inputs
    if (inp.empty() || draft.empty() || !restIndex.isLoaded()) {
        LOG_DBG("Invalid inputs for REST drafting\n");
        return;
    }
    
    // The first token in draft should match the last token in inp
    llama_token last_token = inp.back();
    if (draft[0] != last_token) {
        LOG_DBG("Draft first token doesn't match input last token\n");
        return;
    }
    
    // Try decreasing prefix lengths
    for (int prefix_len = std::min(max_prefix_len, (int)inp.size()); 
         prefix_len >= min_prefix_len; --prefix_len) {
        
        // Extract the prefix
        std::vector<llama_token> prefix;
        prefix.reserve(prefix_len);
        
        for (int i = inp.size() - prefix_len; i < (int)inp.size(); ++i) {
            prefix.push_back(inp[i]);
        }
        
        LOG_DBG("Trying REST search with prefix length %d\n", prefix_len);
        
        // Search for continuations
        std::vector<llama_token> continuations = restIndex.search(prefix, 64);
        
        if (!continuations.empty()) {
            LOG_DBG("Found %zu continuation tokens with prefix length %d\n", 
                   continuations.size(), prefix_len);
            
            // Clear draft except for the first token
            draft.resize(1);
            
            // Add continuation tokens (up to n_draft)
            size_t tokens_to_add = std::min(continuations.size(), static_cast<size_t>(n_draft));
            for (size_t i = 0; i < tokens_to_add; ++i) {
                draft.push_back(continuations[i]);
            }
            
            LOG_DBG("Added %zu tokens to draft from REST search\n", tokens_to_add);
            return;
        }
    }
    
    LOG_DBG("No continuations found with REST search\n");
}