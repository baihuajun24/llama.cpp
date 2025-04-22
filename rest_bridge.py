import draftretriever
import os
import sys

class RESTBridge:
    """Bridge class to connect C++ to draftretriever"""
    
    def __init__(self):
        self.reader = None
        
    def load_index(self, index_path):
        """Load a REST index file"""
        try:
            self.reader = draftretriever.Reader(index_file_path=index_path)
            return True
        except Exception as e:
            print(f"Error loading REST index: {e}", file=sys.stderr)
            return False
            
    def search(self, prefix_tokens, choices=64):
        """Search for continuations given a prefix"""
        if self.reader is None:
            print("Reader not initialized", file=sys.stderr)
            return []
            
        try:
            # Convert to Python list if passed as array
            prefix_list = list(prefix_tokens)
            
            # Call the search method
            retrieved_token_list, _, _, _, _ = self.reader.search(
                prefix_list, 
                choices=choices
            )
            
            # If we found results, return the first continuation sequence
            if retrieved_token_list and len(retrieved_token_list) > 0:
                # Filter out padding tokens (-2)
                return [token for token in retrieved_token_list[0] if token != -2]
            return []
            
        except Exception as e:
            print(f"Error in search: {e}", file=sys.stderr)
            return []

# Create a global instance for easier access from C++
_bridge = RESTBridge()

# Export functions to be called from C++
def load_index(path):
    return _bridge.load_index(path)
    
def search(prefix_tokens, choices=64):
    return _bridge.search(prefix_tokens, choices)
