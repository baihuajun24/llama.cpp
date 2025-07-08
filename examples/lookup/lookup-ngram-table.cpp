#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-cache.h"
#include "ngram-table.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

struct ngram_table_params {
    // Path to a saved ngram table to load
    std::string load_table;
    
    // Minimum n-gram size
    int ngram_min = 1;
    
    // Maximum n-gram size
    int ngram_max = 6;
};

static bool parse_params(int argc, char** argv, common_params& params, ngram_table_params& table_params) {
    bool valid_args = true;
    
    // Set default values first
    table_params.ngram_min = 1;
    table_params.ngram_max = 6;
    
    // Read from environment variables if set
    if (const char* env_min = std::getenv("NGRAM_MIN")) {
        try {
            table_params.ngram_min = std::stoi(env_min);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid NGRAM_MIN value in environment: %s\n", env_min);
        }
    }
    
    if (const char* env_max = std::getenv("NGRAM_MAX")) {
        try {
            table_params.ngram_max = std::stoi(env_max);
        } catch (const std::exception& e) {
            LOG_ERR("Invalid NGRAM_MAX value in environment: %s\n", env_max);
        }
    }

    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--load-table" && i + 1 < argc) {
            table_params.load_table = argv[++i];
        } else if (arg == "--ngram-min" && i + 1 < argc) {
            table_params.ngram_min = std::stoi(argv[++i]);
        } else if (arg == "--ngram-max" && i + 1 < argc) {
            table_params.ngram_max = std::stoi(argv[++i]);
        }
    }
    
    // Validate parameters
    if (table_params.ngram_min < 1) {
        LOG_ERR("ngram-min must be at least 1\n");
        valid_args = false;
    }
    
    if (table_params.ngram_max < table_params.ngram_min) {
        LOG_ERR("ngram-max must be greater than or equal to ngram-min\n");
        valid_args = false;
    }
    
    return valid_args;
}

static void print_usage() {
    printf("usage: llama-lookup-ngram-table [options]\n\n");
    printf("options:\n");
    printf("  -h, --help                  show this help message and exit\n");
    printf("  -m FNAME, --model FNAME     model path\n");
    printf("  -f FNAME, --file FNAME      prompt file path\n");
    printf("  -n N, --n-predict N         number of tokens to predict\n");
    printf("  -t N, --threads N           number of threads\n");
    printf("  --load-table FNAME          load ngram table from this file\n");
    printf("  --ngram-min N               minimum n-gram size (default: 1)\n");
    printf("  --ngram-max N               maximum n-gram size (default: 6)\n");
    printf("  --draft N                   number of tokens to draft (default: 5)\n");
    printf("  -o FNAME, --output FNAME    output file for generated text\n");
}

int main(int argc, char** argv) {
    common_params params;
    ngram_table_params table_params;
    
    // Parse parameters
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        print_usage();
        return 1;
    }
    
    if (!parse_params(argc, argv, params, table_params)) {
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
    
    // Initialize ngram cache for PLD
    common_ngram_cache ngram_cache_context;
    
    // Load static ngram table
    NGramTable ng_table;
    bool table_loaded = false;
    if (!table_loaded) {
        // const std::string table_path = "C:/Users/Administrator/Documents/ngram-spec/cache/llama3_ngram_code_50k.bin";
        // const std::string table_path = "C:/Users/Administrator/Downloads/llama3_ngram_coding_debugging_all.bin";
        const std::string table_path = "C:/Users/Administrator/Downloads/llama3_ngram_merged.bin";
        LOG_INF("0707 Loading ngram table(mmap) from %s\n", table_path.c_str());
        // table_loaded = ng_table.load_mmap(table_path);
        table_loaded = ng_table.load(table_path);
        if (table_loaded) {
            auto stats = ng_table.get_stats();
            LOG_INF("0630 Loaded ngram table with %zu n-grams (min_n=%u, max_n=%u, horizon=%u)\n",
                    stats.total_ngrams, stats.min_n, stats.max_n, stats.horizon);
            
            // [Added on 0701] Debug: Show a few sample entries to verify correct loading
            ng_table.debug_show_entries(5);
        } else {
            LOG_ERR("0630 Failed to load ngram table from %s\n", table_path.c_str());
        }
    }
    
    // Performance tracking
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;
    
    // Update context ngram cache with prompt
    {
        const int64_t t_start_draft_us = ggml_time_us();
        common_ngram_cache_update(ngram_cache_context, table_params.ngram_min, table_params.ngram_max, 
                                inp, inp.size(), false);
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
    
    // Encode prompt
    const auto t_enc_start = ggml_time_us();
    llama_decode(ctx, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(), 1));
    const auto t_enc_end = ggml_time_us();
    
    // Generation variables
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_no_draft_forward = 0;
    std::vector<std::pair<int, int>> verify_list;
    
    int n_past = inp.size();
    bool has_eos = false;
    
    struct common_sampler* smpl = common_sampler_init(model, params.sampling);
    std::vector<llama_token> draft;
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);
    
    const auto t_dec_start = ggml_time_us();
    
    while (true) {
        int i_dft = 0;
        int accept_length = 1;
        
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
                    // Update context ngram cache with the newly accepted token
                    const int64_t t_start_draft_us = ggml_time_us();
                    common_ngram_cache_update(ngram_cache_context, table_params.ngram_min, 
                                           table_params.ngram_max, inp, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
                
                if (write_to_file) {
                    generated_text << token_str;
                }
                
                continue;
            } else {
                if ((int)draft.size() == 0) {
                    n_no_draft_forward += 1;
                }
            }
            
            if (write_to_file) {
                generated_text << token_str;
            }
            
            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            {
                // Update context ngram cache with the newly accepted token
                const int64_t t_start_draft_us = ggml_time_us();
                common_ngram_cache_update(ngram_cache_context, table_params.ngram_min, 
                                       table_params.ngram_max, inp, 1, false);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }
        
        verify_list.push_back({0, accept_length});  // 0 for match_n since we're not tracking it yet
        
        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
        
        // KV cache management
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);
        
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);
        
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        
        const int64_t t_start_draft_us = ggml_time_us();
        
        // Try to get draft tokens from the static ngram table
        if (table_loaded) {
            ng_table.draft(inp, draft, n_draft, table_params.ngram_min, table_params.ngram_max); // commented out on 0703, this method only uses static table for draft
            // auto result = ng_table.interleave_draft(inp, draft, n_draft, table_params.ngram_min, table_params.ngram_max);
            //ng_table.draft_mmap(inp, draft, n_draft, table_params.ngram_min, table_params.ngram_max);
        }
        
        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;
        
        for (size_t i = 1; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        }
        
        llama_decode(ctx, batch_tgt);
        ++n_past;
        
        draft.erase(draft.begin());
    }
    
    auto t_dec_end = ggml_time_us();
    
    // Calculate statistics
    float sum = 0;
    for (const auto& pair : verify_list) {
        sum += pair.second;
    }
    float average = verify_list.empty() ? 0 : sum / verify_list.size();
    
    // Format verify_list as string
    std::string verify_list_str = "[";
    for (size_t i = 0; i < verify_list.size(); i++) {
        verify_list_str += "(" + std::to_string(verify_list[i].first) + "," + 
                          std::to_string(verify_list[i].second) + ")";
        if (i < verify_list.size() - 1) {
            verify_list_str += ", ";
        }
    }
    verify_list_str += "]";
    
    const float decoding_speed = n_predict / ((t_dec_end - t_dec_start) / 1e6f);
    
    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                      << decoding_speed << " t/s, # Forward: " 
                      << verify_list.size() << ", # No draft forward: " 
                      << n_no_draft_forward << "\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Verify list (match_n, accept_length): " << verify_list_str << "\n";
            output_file << "# Average draft time per forward: " 
                      << (verify_list.empty() ? 0.0 : (double)t_draft_us/verify_list.size()) 
                      << " us\n";
            
            // Write the generated text
            output_file << generated_text.str();
            output_file.close();
        }
    }
    
    // Clean up
    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_backend_free();
    
    return 0;
} 