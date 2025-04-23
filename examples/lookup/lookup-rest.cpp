#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "rest-index.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

struct rest_params {
    // Path to a saved index to load
    std::string load_index;
    
    // Maximum prefix length for searching
    int max_prefix_len = 6;
    
    // Minimum prefix length for searching
    int min_prefix_len = 2;
    
    // Number of choices to consider
    int choices = 64;
    
    // Selection strategy: 0 = first, 1 = random
    int selection_strategy = 0;
};

static bool parse_params(int argc, char** argv, common_params& params, rest_params& rest_params) {
    // Similar to ngram_index's parse_params
    // Set defaults, read environment variables, etc.
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--load-index" && i + 1 < argc) {
            rest_params.load_index = argv[++i];
        } else if (arg == "--max-prefix-len" && i + 1 < argc) {
            rest_params.max_prefix_len = std::stoi(argv[++i]);
        } else if (arg == "--min-prefix-len" && i + 1 < argc) {
            rest_params.min_prefix_len = std::stoi(argv[++i]);
        } else if (arg == "--choices" && i + 1 < argc) {
            rest_params.choices = std::stoi(argv[++i]);
        } else if (arg == "--selection-strategy" && i + 1 < argc) {
            rest_params.selection_strategy = std::stoi(argv[++i]);
        }
    }
    
    // // Validate parameters
    // if (rest_params.load_index.empty()) {
    //     LOG_ERR("--load-index must be specified\n");
    //     return false;
    // }
    
    if (rest_params.max_prefix_len < rest_params.min_prefix_len) {
        LOG_ERR("max-prefix-len must be greater than or equal to min-prefix-len\n");
        return false;
    }
    
    return true;
}

static void print_usage() {
    printf("usage: llama-lookup-rest [options]\n\n");
    printf("options:\n");
    printf("  -h, --help                  show this help message and exit\n");
    printf("  -m FNAME, --model FNAME     model path\n");
    printf("  -f FNAME, --file FNAME      prompt file path\n");
    printf("  -n N, --n-predict N         number of tokens to predict\n");
    printf("  -t N, --threads N           number of threads\n");
    printf("  --load-index FNAME          load REST index from this file\n");
    printf("  --max-prefix-len N          maximum prefix length to consider (default: 6)\n");
    printf("  --min-prefix-len N          minimum prefix length to consider (default: 2)\n");
    printf("  --choices N                 number of choices to consider (default: 64)\n");
    printf("  --selection-strategy N      strategy for selecting continuations (0=first, 1=random)\n");
    printf("  --draft N                   number of tokens to draft (default: 5)\n");
    printf("  -o FNAME, --output FNAME    output file for generated text\n");
}

int main(int argc, char** argv) {
    common_params params;
    rest_params rparams;
    
    // Parse parameters
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        print_usage();
        return 1;
    }
    
    if (!parse_params(argc, argv, params, rparams)) {
        print_usage();
        return 1;
    }

    // Hardcode for testing
    rparams.load_index = "C:\\Users\\Administrator\\Documents\\REST\\datastore\\datastore_stack_small.idx";

    rparams.max_prefix_len = 6;
    rparams.min_prefix_len = 2;
    rparams.choices = 64;
    
    // From here on, follow the structure of lookup-ngram-index.cpp's main function:
    // 1. Initialize common components
    // 2. Load the model
    // 3. Load the REST index
    // 4. Process the prompt and generate text
    // 5. Track performance metrics and output the results
    
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
    
    // Initialize REST index
    RESTIndex rest_index;
    
    if (!rest_index.loadIndex(rparams.load_index)) {
        LOG_ERR("Failed to load REST index from %s\n", rparams.load_index.c_str());
        return 1;
    }
    
    LOG_INF("Successfully loaded REST index from %s\n", rparams.load_index.c_str());
    
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
                ++n_accept;
                accept_length += 1;
                ++n_past;
                ++i_dft;
                inp.push_back(id);

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

        // HERE IS THE KEY CHANGE: Call rest_draft instead of ngram_index_draft
        rest_draft(inp, draft, n_draft, rest_index, rparams.max_prefix_len, rparams.min_prefix_len);
        
        // Log draft information
        if (!draft.empty()) {
            std::string draft_content = "";
            for (size_t i = 0; i < draft.size(); i++) {
                draft_content += common_token_to_piece(ctx, draft[i]);
            }
            if (draft.size() > 1) {
                LOG_INF("REST draft [len=%zu]: '%s'\n", draft.size(), draft_content.c_str());
            }
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
    
    // Calculate accept length average and print statistics
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
            output_file << "# REST parameters: min_prefix=" << rparams.min_prefix_len 
                      << ", max_prefix=" << rparams.max_prefix_len << "\n";
            
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
