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


        // Tokenize and query a string in nindex
        const std::string query_string = "def has_close_elements";
        // get the token id of first token in query_string
        llama_token first_token_id = common_tokenize(ctx, query_string, true, true)[1]; // 0 -> <|begin_of_text|>
        LOG_INF("0421 CHECK: first token id: %d\n", first_token_id);
        std::vector<llama_token> tokens_with_bos = common_tokenize(ctx, query_string, true, true);
        std::vector<llama_token> query_tokens(tokens_with_bos.begin() + 1, tokens_with_bos.end());
        // check length of query_tokens
        LOG_INF("0421 CHECK: query_tokens size: %zu\n", query_tokens.size());
        // decode query_tokens to text
        std::string query_text = "";
        for (int i = 0; i < query_tokens.size(); i++) {
            LOG_INF("0421 CHECK: query_text: %s\n", common_token_to_piece(ctx, query_tokens[i]).c_str());
        }
        
        
        if (query_tokens.size() > 0) {
            // Create a key from the tokens
            ngram_index_key key(query_tokens.data(), std::min((int)query_tokens.size(), idx_params.ngram_max), idx_params.ngram_max);
            
            // Find the key in the index
            auto it = nindex.find(key);
            if (it != nindex.end()) {
                LOG_INF("0421 CHECK: '%s' found in index\n", query_string.c_str());
                
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
                LOG_INF("0421 CHECK: '%s' not found in index\n", query_string.c_str());
            }
        } else {
            LOG_INF("0421 CHECK: Failed to tokenize '%s'\n", query_string.c_str());
        }

        
        // Print all patterns starting with first_token_id
        LOG_INF("===== Patterns starting with token %d =====\n", first_token_id);
        for (const auto& pair : nindex) {
            // Check if pattern starts with first_token_id
            if (pair.first.tokens[0] == first_token_id) {
                // Get the token IDs
                std::string ids_str = "Token IDs: [";
                int actual_size = 0;
                for (int i = 0; i < pair.first.size; i++) {
                    if (pair.first.tokens[i] != LLAMA_TOKEN_NULL) {
                        ids_str += std::to_string(pair.first.tokens[i]);
                        actual_size++;
                        if (i < pair.first.size-1 && pair.first.tokens[i+1] != LLAMA_TOKEN_NULL) {
                            ids_str += ", ";
                        }
                    }
                }
                ids_str += "]";
                LOG_INF("%s\n", ids_str.c_str());
                
                // Decode the pattern tokens to text
                std::string text_str = "Pattern text: '";
                for (int i = 0; i < pair.first.size; i++) {
                    if (pair.first.tokens[i] != LLAMA_TOKEN_NULL) {
                        text_str += common_token_to_piece(ctx, pair.first.tokens[i]);
                    }
                }
                text_str += "'";
                LOG_INF("%s\n", text_str.c_str());
                
                // Show positions where pattern occurs
                std::string pos_str = "Positions: [";
                for (size_t i = 0; i < pair.second.size(); i++) {
                    pos_str += std::to_string(pair.second[i]);
                    if (i < pair.second.size() - 1) {
                        pos_str += ", ";
                    }
                }
                pos_str += "]";
                LOG_INF("%s\n", pos_str.c_str());

                // For each position, show continuation
                LOG_INF("Continuations:\n");
                for (int pos : pair.second) {
                    // Get 10 tokens after pattern
                    std::string cont_ids = "  Next token IDs: [";
                    std::string cont_text = "  Continuation text: '";
                    
                    int pattern_len = actual_size;
                    int end_pos = std::min(pos + pattern_len + 10, (int)index_tokens.size());
                    
                    for (int i = pos + pattern_len; i < end_pos; i++) {
                        cont_ids += std::to_string(index_tokens[i]);
                        cont_text += common_token_to_piece(ctx, index_tokens[i]);
                        if (i < end_pos - 1) {
                            cont_ids += ", ";
                        }
                    }
                    cont_ids += "]";
                    cont_text += "'";
                    
                    LOG_INF("%s\n", cont_ids.c_str());
                    LOG_INF("%s\n", cont_text.c_str());
                }
                
                LOG_INF("------------------------------\n");
            }
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
    
    int64_t t_draft_us = 0;
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
    
    while (true) {

        int i_dft = 0;
        int accept_length = 0;
        int debug_count = 0; // for test
        while (debug_count < 1000) {
            debug_count++;
            // sample from the target model
            llama_token id = common_sampler_sample(smpl, ctx, i_dft);

            common_sampler_accept(smpl, id, true);

            const std::string token_str = common_token_to_piece(ctx, id);

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches the draft
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                LOG_INF("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                ++n_accept;
                accept_length += 1;
                ++n_past;
                ++i_dft;
                inp.push_back(id);
                {
                    // Update context ngram cache with the newly accepted token:
                    const int64_t t_start_draft_us = ggml_time_us();
                    ngram_index_update(nindex, inp, params.ngram_min, params.ngram_max, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }

                if (write_to_file) {
                    generated_text << token_str;
                }

                continue;
            }

            if (write_to_file) {
                generated_text << token_str;
            }


            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            {
                // Update context ngram cache with the newly accepted token:
                const int64_t t_start_draft_us = ggml_time_us();
                ngram_index_update(nindex, inp, params.ngram_min, params.ngram_max, 1, false);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }
        n_accept_list.push_back(std::max(accept_length, 1));
        
        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

        // Draft already contains a single token sampled from the model:
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        const int64_t t_start_draft_us = ggml_time_us();

        // In main generation loop
        ngram_index_draft(inp, draft, n_draft, idx_params.ngram_min, idx_params.ngram_max, nindex, index_tokens, idx_params.selection_strategy);    
        //ngram_index_draft(inp, draft, n_draft, params.ngram_min, params.ngram_max, nindex, 0);
        // check content of draft if not empty, len of draf
        if (!draft.empty()) {
            // Create a string to hold the concatenated tokens
            std::string draft_content = "";
            for (size_t i = 0; i < draft.size(); i++) {
                draft_content += common_token_to_piece(ctx, draft[i]);
            }
            LOG_INF("0421 CHECK: draft [len=%zu]: '%s'\n", draft.size(), draft_content.c_str());
        }

        for (size_t i = 1; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        }

        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;

        llama_decode(ctx, batch_tgt);
        ++n_past;

        draft.erase(draft.begin());
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