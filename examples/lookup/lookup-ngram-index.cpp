#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-index.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

struct ngram_index_params {
    // Input file with prompt to index
    std::string index_file;
    
    // Output file to save the index
    std::string save_index;
    
    // Path to a saved index to load
    std::string load_index;
    
    // Minimum n-gram size
    int ngram_min = 1;
    
    // Maximum n-gram size
    int ngram_max = 4;
    
    // Maximum positions to store per n-gram (0 = unlimited)
    int max_positions = 0;
    
    // Selection strategy: 0 = first, 1 = random
    int selection_strategy = 0;
};

static bool parse_params(int argc, char** argv, common_params& params, ngram_index_params& idx_params) {
    bool valid_args = true;
    
    // Set default values first
    idx_params.ngram_min = 1;
    idx_params.ngram_max = 4;
    idx_params.max_positions = 0;
    idx_params.selection_strategy = 0;
    
    // Read from environment variables if set
    if (const char* env_min = std::getenv("NGRAM_MIN")) {
        try {
            idx_params.ngram_min = std::stoi(env_min);
            LOG_INF("Using NGRAM_MIN=%d from environment variable\n", idx_params.ngram_min);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid NGRAM_MIN value in environment: %s\n", env_min);
        }
    }
    
    if (const char* env_max = std::getenv("NGRAM_MAX")) {
        try {
            idx_params.ngram_max = std::stoi(env_max);
            LOG_INF("Using NGRAM_MAX=%d from environment variable\n", idx_params.ngram_max);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid NGRAM_MAX value in environment: %s\n", env_max);
        }
    }
    
    if (const char* env_pos = std::getenv("MAX_POSITIONS")) {
        try {
            idx_params.max_positions = std::stoi(env_pos);
            LOG_INF("Using MAX_POSITIONS=%d from environment variable\n", idx_params.max_positions);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid MAX_POSITIONS value in environment: %s\n", env_pos);
        }
    }
    
    if (const char* env_sel = std::getenv("SELECTION_STRATEGY")) {
        try {
            idx_params.selection_strategy = std::stoi(env_sel);
            LOG_INF("Using SELECTION_STRATEGY=%d from environment variable\n", idx_params.selection_strategy);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid SELECTION_STRATEGY value in environment: %s\n", env_sel);
        }
    }
    
    // Process command line arguments for parameters that are recognized by common_params_parse
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--index-file" && i + 1 < argc) {
            idx_params.index_file = argv[++i];
        } else if (arg == "--save-index" && i + 1 < argc) {
            idx_params.save_index = argv[++i];
        } else if (arg == "--load-index" && i + 1 < argc) {
            idx_params.load_index = argv[++i];
        }
        // No longer parsing --ngram-min, --ngram-max, --max-positions, --selection-strategy
        // from command line to avoid conflicts with common_params_parse
    }
    
    // Validate parameters
    if (idx_params.index_file.empty() && idx_params.load_index.empty()) {
        // If no index file specified, use the prompt file as the index
        if (!params.prompt_file.empty()) {
            idx_params.index_file = params.prompt_file;
            LOG_INF("Using prompt file as index: %s\n", idx_params.index_file.c_str());
        } else {
            LOG_ERR("Either --index-file, --load-index, or -f/--file must be specified\n");
            valid_args = false;
        }
    }
    
    if (idx_params.ngram_min < 1) {
        LOG_ERR("ngram-min must be at least 1\n");
        valid_args = false;
    }
    
    if (idx_params.ngram_max < idx_params.ngram_min) {
        LOG_ERR("ngram-max must be greater than or equal to ngram-min\n");
        valid_args = false;
    }
    
    if (idx_params.selection_strategy < 0 || idx_params.selection_strategy > 1) {
        LOG_ERR("selection-strategy must be 0 (first) or 1 (random)\n");
        valid_args = false;
    }
    
    return valid_args;
}

static void print_usage() {
    printf("usage: llama-lookup-ngram-index [options]\n\n");
    printf("options:\n");
    printf("  -h, --help                  show this help message and exit\n");
    printf("  -m FNAME, --model FNAME     model path\n");
    printf("  -f FNAME, --file FNAME      prompt file path\n");
    printf("  -n N, --n-predict N         number of tokens to predict\n");
    printf("  -t N, --threads N           number of threads\n");
    printf("  --index-file FNAME          text file to index for n-grams\n");
    printf("  --save-index FNAME          save built index to this file\n");
    printf("  --load-index FNAME          load index from this file\n");
    printf("  --ngram-min N               minimum n-gram size (default: 1)\n");
    printf("  --ngram-max N               maximum n-gram size (default: 4)\n");
    printf("  --max-positions N           maximum positions per n-gram (default: 0 = unlimited)\n");
    printf("  --selection-strategy N      strategy for selecting continuations (0=first, 1=random)\n");
    printf("  --draft N                   number of tokens to draft (default: 5)\n");
    printf("  -o FNAME, --output FNAME    output file for generated text\n");
}

int main(int argc, char** argv) {
    common_params params;
    ngram_index_params idx_params;
    
    // Parse common parameters
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        print_usage();
        return 1;
    }
    
    // Parse n-gram index specific parameters
    if (!parse_params(argc, argv, params, idx_params)) {
        print_usage();
        return 1;
    }
    
    common_init();
    
    // For storing generated text
    std::stringstream generated_text;
    bool write_to_file = !params.out_file.empty();
    
    // Max number of tokens to draft
    const int n_draft = params.speculative.n_max;
    
    // Initialize llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load the model
    common_init_result llama_init = common_init_from_params(params);
    
    llama_model* model = llama_init.model.get();
    llama_context* ctx = llama_init.context.get();
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    // Tokenize prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);
    
    // Tokenize the index file for n-gram indexing
    std::vector<llama_token> index_tokens;
    ngram_index nindex;
    
    if (!idx_params.load_index.empty()) {
        // Load existing index
        nindex = load_ngram_index(idx_params.load_index);
    } else if (!idx_params.index_file.empty()) {
        // Create new index from file
        std::ifstream index_file(idx_params.index_file);
        if (!index_file) {
            LOG_ERR("Failed to open index file: %s\n", idx_params.index_file.c_str());
            return 1;
        }
        
        std::string index_text((std::istreambuf_iterator<char>(index_file)),
                               std::istreambuf_iterator<char>());
        index_file.close();
        
        // Tokenize index text
        index_tokens = common_tokenize(ctx, index_text, true, true);
        
        // Build n-gram index
        nindex = build_ngram_index(
            index_tokens,
            idx_params.ngram_min,
            idx_params.ngram_max,
            idx_params.max_positions,
            true);
        
        // Print index statistics
        print_ngram_index_stats(nindex);
        
        // Save index if requested
        if (!idx_params.save_index.empty()) {
            save_ngram_index(nindex, idx_params.save_index);
        }

        // Replace the sample printing section (around line 230-265) with this:
        // Print 5 samples of 3-gram patterns
        LOG_INF("===== 3-gram Pattern Samples =====\n");
        int samples_printed = 0;
        for (const auto& pair : nindex) {
            // Check if this is a 3-gram pattern by its size
            if (pair.first.size == 3) {
                // Get the token IDs
                std::string ids_str = "Token IDs: [";
                for (int i = 0; i < pair.first.size; i++) {
                    ids_str += std::to_string(pair.first.tokens[i]);
                    if (i < pair.first.size - 1) {
                        ids_str += ", ";
                    }
                }
                ids_str += "]";
                LOG_INF("%s\n", ids_str.c_str());
                
                // Decode the tokens to display as text
                std::string text_str = "Text: '";
                for (int i = 0; i < pair.first.size; i++) {
                    text_str += common_token_to_piece(ctx, pair.first.tokens[i]);
                }
                text_str += "'";
                LOG_INF("%s\n", text_str.c_str());
                
                // Show the positions where this n-gram occurs
                std::string pos_str = "Positions: [";
                for (size_t i = 0; i < std::min(pair.second.size(), size_t(5)); i++) {
                    pos_str += std::to_string(pair.second[i]);
                    if (i < std::min(pair.second.size(), size_t(5)) - 1) {
                        pos_str += ", ";
                    }
                }
                if (pair.second.size() > 5) {
                    pos_str += ", ...";
                }
                pos_str += "]";
                LOG_INF("%s\n", pos_str.c_str());
                
                LOG_INF("------------------------------\n");
                
                samples_printed++;
                if (samples_printed >= 5) {
                    break; // Stop after printing 5 samples
                }
            }
        }

        // Tokenize and query the string "for i in" in nindex, print the value retrieved
        std::vector<llama_token> query_tokens = common_tokenize(ctx, "def has_close_elements", true, true);
        if (query_tokens.size() > 0) {
            // Create a key from the tokens
            ngram_index_key key(query_tokens.data(), std::min((int)query_tokens.size(), idx_params.ngram_max), idx_params.ngram_max);
            
            // Find the key in the index
            auto it = nindex.find(key);
            if (it != nindex.end()) {
                LOG_INF("0421 CHECK: 'def has_close_elements' found in index\n");
                
                // Get the first position
                if (!it->second.empty()) {
                    int position = it->second[0];
                    LOG_INF("First position: %d\n", position);
                    
                    // Print 10 tokens starting at this position (if available)
                    std::string retrieved_text = "";
                    int end_pos = std::min(position + 10, (int)index_tokens.size());
                    for (int i = position; i < end_pos; i++) {
                        retrieved_text += common_token_to_piece(ctx, index_tokens[i]);
                    }
                    LOG_INF("Retrieved text: %s\n", retrieved_text.c_str());
                } else {
                    LOG_INF("No positions stored for this n-gram\n");
                }
            } else {
                LOG_INF("0421 CHECK: 'def has_close_elements' not found in index\n");
            }
        } else {
            LOG_INF("0421 CHECK: Failed to tokenize 'def has_close_elements'\n");
        }
    }
    
    // Check context limits
    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;
    
    if ((int)inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }
    
    // Print prompt
    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }
    fflush(stderr);
    
    const int n_input = inp.size();
    
    // Track performance metrics
    const auto t_enc_start = ggml_time_us();
    
    // Encode prompt
    llama_decode(ctx, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(), 1));
    
    const auto t_enc_end = ggml_time_us();
    
    // Generation variables
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept = 0;
    std::vector<int> n_accept_list;
    
    int n_past = inp.size();
    bool has_eos = false;
    
    struct common_sampler* smpl = common_sampler_init(model, params.sampling);
    std::vector<llama_token> draft;
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);
    
    const auto t_dec_start = ggml_time_us();
    
    // Main generation loop
    while (true) {
        // Clear previous draft
        draft.clear();
        
        int i_dft = 0;
        int accept_length = 0;
        
        // Sample first token
        llama_token id = common_sampler_sample(smpl, ctx, i_dft);
        common_sampler_accept(smpl, id, true);
        
        const std::string token_str = common_token_to_piece(ctx, id);
        
        LOG("%s", token_str.c_str());
        
        if (llama_vocab_is_eog(vocab, id)) {
            has_eos = true;
        }
        
        ++n_predict;
        
        // Initialize draft with this token
        draft.push_back(id);
        
        // Draft additional tokens using n-gram index
        if (!nindex.empty()) {
            LOG_DBG("Before drafting: inp.size=%zu, draft.size=%zu\n", 
                    inp.size(), draft.size());
            
            // Count existing drafted tokens
            int pre_draft_size = draft.size();
            
            draft_with_ngram_index(
                inp,
                draft,
                n_draft,
                index_tokens.empty() ? inp : index_tokens, // Use prompt if no index tokens
                nindex,
                idx_params.ngram_min,
                idx_params.ngram_max,
                idx_params.selection_strategy);
            
            // Count how many new tokens were added
            int newly_drafted = draft.size() - pre_draft_size;
            LOG_DBG("Drafted %d additional tokens\n", newly_drafted);
            
            // Track total drafts for statistics
            n_drafted += newly_drafted;
        }
        
        // Record first token
        if (write_to_file) {
            generated_text << token_str;
        }
        
        // Add first token to input
        inp.push_back(id);
        
        // If we drafted additional tokens, verify them
        if (draft.size() > 1) {
            // Prepare batch for first token
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);
            
            // Decode first token
            llama_decode(ctx, batch_tgt);
            n_past++;
            
            // Reset acceptance counter for this round
            accept_length = 0;
            
            // Process drafted tokens (starting from index 1)
            for (size_t i = 1; i < draft.size(); ++i) {
                // Sample from model
                id = common_sampler_sample(smpl, ctx, i_dft);
                common_sampler_accept(smpl, id, true);
                
                const std::string drafted_token_str = common_token_to_piece(ctx, id);
                LOG("%s", drafted_token_str.c_str());
                
                ++n_predict;
                
                // Check if token matches our drafted token
                if (id == draft[i]) {
                    // Match! Accept the token
                    LOG_DBG("the sampled target token matches the drafted token (%d, '%s') - accepted\n", 
                            id, drafted_token_str.c_str());
                    ++n_accept;
                    ++accept_length;
                    ++n_past;
                    ++i_dft;
                    inp.push_back(id);
                    
                    if (write_to_file) {
                        generated_text << drafted_token_str;
                    }
                    
                    // Prepare batch for next token (if any)
                    if (i + 1 < draft.size()) {
                        common_batch_clear(batch_tgt);
                        common_batch_add(batch_tgt, draft[i+1], n_past, { 0 }, true);
                        llama_decode(ctx, batch_tgt);
                    }
                    
                    // Continue with next drafted token
                    continue;
                } else {
                    // No match, stop drafting
                    LOG_DBG("the sampled target token (%d, '%s') did not match the drafted token (%d)\n", 
                            id, drafted_token_str.c_str(), draft[i]);
                    
                    if (write_to_file) {
                        generated_text << drafted_token_str;
                    }
                    
                    // Reset draft with this token
                    draft.clear();
                    draft.push_back(id);
                    inp.push_back(id);
                    break;
                }
            }
            
            // Record acceptance statistics
            n_accept_list.push_back(std::max(accept_length, 1));
        } else {
            // No drafting occurred, just record a single acceptance
            n_accept_list.push_back(1);
        }
        
        // Check if we've generated enough tokens or reached EOS
        if ((params.n_predict > 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
        
        // Reset for next token
        n_past = inp.size();
    }
    
    auto t_dec_end = ggml_time_us();
    
    LOG("\n\n");
    
    // Print performance statistics
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", 
            n_input, (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", 
            n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));
    
    // Calculate accept length average
    float sum = 0;
    for (size_t i = 0; i < n_accept_list.size(); i++) {
        sum += n_accept_list[i];
    }
    float average = n_accept_list.empty() ? 0 : sum / n_accept_list.size();
    
    // Format the vector as a string
    std::string accept_list_str = "[";
    for (size_t i = 0; i < n_accept_list.size(); i++) {
        accept_list_str += std::to_string(n_accept_list[i]);
        if (i < n_accept_list.size() - 1) {
            accept_list_str += ", ";
        }
    }
    accept_list_str += "]";
    
    LOG_INF("Accept lengths: %s\n", accept_list_str.c_str());
    LOG_INF("Average accept length: %.2f\n", average);
    LOG_INF("Total drafted tokens: %d\n", n_drafted);
    LOG_INF("Total accepted tokens: %d\n", n_accept);
    if (n_drafted > 0) {
        LOG_INF("Accept rate: %.2f%%\n", 100.0f * n_accept / n_drafted);
    }
    
    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            // First write the statistics as header lines
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                      << (n_predict / ((t_dec_end - t_dec_start) / 1e6f)) << " t/s\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Accept lengths: " << accept_list_str << "\n";
            output_file << "# N-gram index parameters: min=" << idx_params.ngram_min 
                      << ", max=" << idx_params.ngram_max << "\n";
            
            // Then write the generated text
            output_file << generated_text.str();
            output_file.close();
            
            LOG_INF("Generated text written to: %s\n", params.out_file.c_str());
        }
    }
    
    // Clean up
    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_backend_free();
    
    return 0;
}