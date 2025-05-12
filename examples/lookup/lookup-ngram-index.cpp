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
#include <iomanip>  // for std::setprecision

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
};

static bool parse_params(int argc, char** argv, common_params& params, ngram_index_params& idx_params) {
    bool valid_args = true;
    
    // Set default values first
    idx_params.ngram_min = 1;
    idx_params.ngram_max = 4;
    
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

    // Add this near the other environment variable checks
    if (const char* env_index_file = std::getenv("INDEX_FILE")) {
        idx_params.index_file = env_index_file;
        LOG_INF("Using INDEX_FILE=%s from environment variable\n", idx_params.index_file.c_str());
    }
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--index-file" && i + 1 < argc) {
            idx_params.index_file = argv[++i];
        } else if (arg == "--save-index" && i + 1 < argc) {
            idx_params.save_index = argv[++i];
        } else if (arg == "--load-index" && i + 1 < argc) {
            idx_params.load_index = argv[++i];
        } else if (arg == "--ngram-min" && i + 1 < argc) {
            idx_params.ngram_min = std::stoi(argv[++i]);
        } else if (arg == "--ngram-max" && i + 1 < argc) {
            idx_params.ngram_max = std::stoi(argv[++i]);
        }
    }
    
    // Validate parameters
    if (idx_params.ngram_min < 1) {
        LOG_ERR("ngram-min must be at least 1\n");
        valid_args = false;
    }
    
    if (idx_params.ngram_max < idx_params.ngram_min) {
        LOG_ERR("ngram-max must be greater than or equal to ngram-min\n");
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
    printf("  --index-file FNAME          text file to index for n-grams (not used in first iteration)\n");
    printf("  --save-index FNAME          save built index to this file (not used in first iteration)\n");
    printf("  --load-index FNAME          load index from this file (not used in first iteration)\n");
    printf("  --ngram-min N               minimum n-gram size (default: 1)\n");
    printf("  --ngram-max N               maximum n-gram size (default: 4)\n");
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
    
    // Debug flag
    const bool dump_kv_cache = params.dump_kv_cache;
    
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
    
    // Create a new n-gram index with the specified parameters
    NGramIndex nindex(idx_params.ngram_min, idx_params.ngram_max);

    // if load index is not empty, load the index from file
    NGramIndex nindex_static(idx_params.ngram_min, idx_params.ngram_max);
    
    // 0505 testing: HARDCODE for now
    bool static_index_loaded = true;
    if (static_index_loaded) {
        std::string INDEX_PATH;
        
        // If index_file is provided via command line, use it
        if (!idx_params.index_file.empty()) {
            INDEX_PATH = idx_params.index_file;
        } else {
            // print a msg and throw an error
            LOG_INF("No index file provided via command line\n");
            // throw std::runtime_error("No index file provided via command line");
            // Default fallback paths
            #ifdef _WIN32
                INDEX_PATH = "C:\\Users\\Administrator\\Documents\\ngram-spec\\cache\\llama3_magpie_info_50k.bin"; // Default for Windows
            #else
                INDEX_PATH = "/Users/baihuajun/Documents/llama.cpp/ngram-spec/cache/llama3_magpie_info_50k.bin"; // Default for Mac/Linux
            #endif
        }
        
        LOG_INF("Loading static n-gram index from %s", INDEX_PATH.c_str());
        
        static_index_loaded = nindex_static.load(INDEX_PATH);
        nindex_static.print_stats();

    }
    
    // Initialize performance metrics
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;
    
    {
        // Fill up n-gram index with tokens from user input
        const int64_t t_start_draft_us = ggml_time_us();
        
        // Use the initial prompt to build the index
        if (!inp.empty()) {
            // Index the entire prompt at once
            nindex.index_prompt(inp);
        }
        
        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
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
    int n_accept = 1;
    int n_no_draft_forward = 0;
    std::vector<int> n_accept_list;
    
    int n_past = inp.size();
    bool has_eos = false;
    
    struct common_sampler* smpl = common_sampler_init(model, params.sampling);
    std::vector<llama_token> draft;
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);
    
    // Debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);
    
    const auto t_dec_start = ggml_time_us();
    
    while (true) {
        // Debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }
        
        int i_dft = 0;
        int accept_length = 1; // Start from 1 to match lookup.cpp behavior
        
        while (true) {
            // Sample from the target model
            llama_token id = common_sampler_sample(smpl, ctx, i_dft);
            
            common_sampler_accept(smpl, id, true);
            
            const std::string token_str = common_token_to_piece(ctx, id);
            
            if (!params.use_color) {
                LOG("%s", token_str.c_str());
            }
            
            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }
            
            ++n_predict;
            
            // Check if the target token matches the draft
            if (i_dft < (int)draft.size() && id == draft[i_dft]) {
                ++n_accept;
                accept_length += 1;
                ++n_past;
                ++i_dft;
                inp.push_back(id);
                {
                    // Update n-gram index with the newly accepted token
                    const int64_t t_start_draft_us = ggml_time_us();
                    // Add the new token to the index (context now includes the new token)
                    // nindex.add_token(inp.data(), inp.size());
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
                
                if (write_to_file) {
                    generated_text << token_str;
                }
                
                continue;
            } else {
                if ((int)draft.size() == 0) {
                    n_no_draft_forward += 1;
                    // LOG_INF("[Check] no draft caused accept length = %d\n", accept_length);
                }
            }
            
            if (write_to_file) {
                generated_text << token_str;
            }
            
            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            {
                // Update n-gram index with the newly accepted token
                const int64_t t_start_draft_us = ggml_time_us();
                // Add the new token to the index (context now includes the new token)
                //nindex.add_token(inp.data(), inp.size());
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }
        
        n_accept_list.push_back(accept_length);
        
        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
        
        // KV cache management
        // Clean the cache of draft tokens that weren't accepted
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);
        
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);
        
        // Draft already contains a single token sampled from the model
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        
        const int64_t t_start_draft_us = ggml_time_us();
        
        // Use the n-gram index to draft tokens
        int n_drafted_tokens = nindex.draft(inp, draft, n_draft);
        
        // If no tokens were drafted from the main index and static index is loaded, try it
        if (n_drafted_tokens == 0 && static_index_loaded) {
            n_drafted_tokens = nindex_static.draft(inp, draft, n_draft);
            if (n_drafted_tokens > 0) {
                LOG_INF("0505 Check: Drafted %d tokens from static index\n", n_drafted_tokens);
            }
        }
                
        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;
        
        // Add drafted tokens to the batch for the next forward pass
        for (size_t i = 1; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        }
        
        llama_decode(ctx, batch_tgt);
        ++n_past;
        
        draft.erase(draft.begin());
    }
    
    auto t_dec_end = ggml_time_us();
    
    LOG("\n\n");
    
    // Print performance statistics
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", 
            n_input, (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    const float decoding_speed = n_predict / ((t_dec_end - t_dec_start) / 1e6f);
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", 
            n_predict, (t_dec_end - t_dec_start) / 1e6f, decoding_speed);
    
    // Print the n-gram index statistics
    nindex.print_stats();

    // Calculate memory usage for nindex
    size_t nindex_mem_tokens = nindex.get_prompt_tokens().size() * sizeof(llama_token);
    size_t nindex_mem_index = nindex.get_index().size() * (sizeof(common_ngram) + sizeof(location_buffer));
    size_t nindex_mem_total = nindex_mem_tokens + nindex_mem_index + sizeof(NGramIndex);
    LOG_INF("Memory usage for nindex: %zu KB (Tokens: %zu KB, Index: %zu KB)\n", 
            nindex_mem_total / 1024, nindex_mem_tokens / 1024, nindex_mem_index / 1024);

    // Declare these variables outside any if blocks, but initialize them to 0
    size_t static_mem_tokens = 0;
    size_t static_mem_index = 0;
    size_t static_mem_total = 0;

    // Print static index stats if loaded
    if (static_index_loaded) {
        nindex_static.print_stats();
        
        // Calculate memory usage for nindex_static
        static_mem_tokens = nindex_static.get_prompt_tokens().size() * sizeof(llama_token);
        static_mem_index = nindex_static.get_index().size() * (sizeof(common_ngram) + sizeof(location_buffer));
        static_mem_total = static_mem_tokens + static_mem_index + sizeof(NGramIndex);
        LOG_INF("Memory usage for nindex_static: %zu KB (Tokens: %zu KB, Index: %zu KB)\n", 
                static_mem_total / 1024, static_mem_tokens / 1024, static_mem_index / 1024);
    }
    
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
    
    LOG_INF("0420 Check: len is %d, n_accept_list = %s\n", (int)n_accept_list.size(), accept_list_str.c_str());
    LOG_INF("0420 Check: accept length average      = %.3f\n", average);
    LOG_INF("0505 Check: add_token is not used -> nindex is not updated; but nindex_static is used\n");
    LOG_INF("0428 Check: no draft is suppiled forward times = %d\n", n_no_draft_forward);
    
    LOG_INF("\n");
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_INF("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_INF("n_accept     = %d\n", n_accept);
    LOG_INF("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);
    
    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            // First write the statistics as header lines            
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                      << decoding_speed << " t/s, # Forward: " 
                      << n_accept_list.size() << ", # No draft forward: " 
                      << n_no_draft_forward << "\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Accept length list: " << accept_list_str << "\n";

            // Add memory usage statistics
            size_t nindex_mem_total = nindex_mem_tokens + nindex_mem_index + sizeof(NGramIndex);
            output_file << "# Memory for nindex: " << nindex_mem_total / 1024 << " KB (Tokens: " 
                    << nindex_mem_tokens / 1024 << " KB, Index: " 
                    << nindex_mem_index / 1024 << " KB)\n";
            
            if (static_index_loaded) {
                size_t static_mem_total = static_mem_tokens + static_mem_index + sizeof(NGramIndex);
                output_file << "# Memory for nindex_static: " << static_mem_total / 1024 << " KB (Tokens: " 
                        << static_mem_tokens / 1024 << " KB, Index: " 
                        << static_mem_index / 1024 << " KB)\n";
            }

            // Then write the generated text
            output_file << generated_text.str();
            output_file.close();
            
            LOG_INF("Generated text written to: %s\n", params.out_file.c_str());
        }
    }
    
    LOG_INF("\ntarget:\n\n");
    common_perf_print(ctx, smpl);
    
    // Clean up
    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_backend_free();
    
    LOG("\n\n");
    
    return 0;
}