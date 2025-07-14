#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ngram-suffix.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <random>
#include <chrono>

#include <iostream>
#include <iomanip>  // for std::fixed, std::setprecision

// Draft from suffix array only
static std::pair<int, char> draft_from_suffix_array(const SuffixArray& suffix_array,
                                             const std::vector<llama_token>& input,
                                             std::vector<llama_token>& draft,
                                             int n_draft,
                                             int min_n,
                                             int max_n) {
    if (!suffix_array.is_mmap_loaded()) {
        return {0, 'X'};
    }
    
    const int inp_size = input.size();
    const int max_n_actual = std::min(max_n, inp_size);

    for (int n = max_n_actual; n >= min_n; --n) {
        if (inp_size < n) continue;

        // Extract last n tokens as query
        std::vector<llama_token> query(input.end() - n, input.end());
        
        // Get coherent draft from suffix array
        std::vector<llama_token> coherent_draft = suffix_array.get_coherent_draft(query, n_draft);
        
        if (!coherent_draft.empty()) {
            // Add coherent draft tokens
            for (const auto& token : coherent_draft) {
                draft.push_back(token);
            }
            return {n, 'S'}; // Found in suffix array
        }
    }
    
    return {0, 'X'}; // No match found
}

// Draft from prompt history first, then suffix array (interleaved)
static std::pair<int, char> draft_interleaved(const SuffixArray& suffix_array,
                                       const std::vector<llama_token>& input,
                                       std::vector<llama_token>& draft,
                                       int n_draft,
                                       int min_n,
                                       int max_n) {
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

        // If not found in prompt history, try suffix array
        if (suffix_array.is_mmap_loaded()) {
            // Extract last n tokens as query
            std::vector<llama_token> query(input.end() - n, input.end());
            
            // Get coherent draft from suffix array
            std::vector<llama_token> coherent_draft = suffix_array.get_coherent_draft(query, n_draft);
            
            if (!coherent_draft.empty()) {
                // Add coherent draft tokens
                for (const auto& token : coherent_draft) {
                    draft.push_back(token);
                }
                return {n, 'S'}; // Found in suffix array
            }
        }
    }
    
    return {0, 'X'}; // No match found
}

static void test_suffix_array_search(const SuffixArray& suffix_array, llama_context* ctx) {
    // Define the test string
    std::string test_string = "for i in";

    // Tokenize the input string
    std::vector<llama_token> test_ids = common_tokenize(ctx, test_string, true, true);
    test_ids.erase(test_ids.begin());  // remove first element (BOS token)
    LOG_INF("Test string: \"%s\"\n", test_string.c_str());

    // Print token IDs
    LOG_INF("Encoded token IDs: [");
    for (size_t i = 0; i < test_ids.size(); ++i) {
        LOG("%d", test_ids[i]);
        if (i < test_ids.size() - 1) LOG(", ");
    }
    LOG("]\n");

    // Search suffix array with test_ids as input
    std::vector<llama_token> draft;
    auto result = draft_from_suffix_array(suffix_array, test_ids, draft, /*n_draft=*/10, /*min_n=*/2, /*max_n=*/6);
    int match_n = result.first;
    char source = result.second;

    LOG_INF("Match n-gram length: %d, source: %c\n", match_n, source);

    // Print draft tokens
    if (draft.empty()) {
        LOG_INF("No draft generated.\n");
    } else {
        LOG_INF("Drafted token IDs: [");
        for (size_t i = 0; i < draft.size(); ++i) {
            LOG("%d", draft[i]);
            if (i < draft.size() - 1) LOG(", ");
        }
        LOG("]\n");

        // Print decoded draft text
        std::string decoded_draft;
        for (llama_token tok : draft) {
            decoded_draft += common_token_to_piece(ctx, tok);
        }
        LOG_INF("Decoded draft: \"%s\"\n", decoded_draft.c_str());
    }
}

int main(int argc, char** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    common_init();

    std::stringstream generated_text;
    bool write_to_file = !params.out_file.empty();

    // max. number of additional tokens to draft if match is found
    const int n_draft = params.speculative.n_max;

    const bool dump_kv_cache = params.dump_kv_cache;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model* model = llama_init.model.get();
    llama_context* ctx = llama_init.context.get();

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);

    // Load suffix array based on size key
    SuffixArray suffix_array;
    std::string suffix_file;

    // Get suffix array size from environment variable or default to "full"
    std::string size_key = "01pct";
    const char* size_env = std::getenv("SUFFIX_SIZE");
    if (size_env) {
        size_key = std::string(size_env);
    }

    // Map size key to file path
    std::map<std::string, std::string> size_to_file = {
        {"0001pct", "C:/Users/Administrator/Documents/ngram-spec/outputs/code/magpie-all-suffix-0001pct.bin"},
        {"01pct",   "C:/Users/Administrator/Documents/ngram-spec/outputs/code/magpie-all-suffix-01pct.bin"},
        {"10pct",   "C:/Users/Administrator/Documents/ngram-spec/outputs/code/magpie-all-suffix-10pct.bin"},
        {"full",    "C:/Users/Administrator/Documents/ngram-spec/outputs/code/magpie-all-suffix-full.bin"}
    };

    // Check if size key is valid
    if (size_to_file.find(size_key) == size_to_file.end()) {
        LOG_ERR("Invalid suffix array size key: %s\n", size_key.c_str());
        LOG_ERR("Valid sizes: 0001pct, 01pct, 10pct, full\n");
        return 1;
    }

    suffix_file = size_to_file[size_key];
    LOG_INF("Loading suffix array: %s (size: %s)\n", suffix_file.c_str(), size_key.c_str());

    if (!suffix_array.load_mmap(suffix_file)) {
        LOG_ERR("Failed to load suffix array from: %s\n", suffix_file.c_str());
        return 1;
    }

    // Test suffix array search
    // test_suffix_array_search(suffix_array, ctx);
    // return 0;

    // Determine draft strategy based on parameters
    bool use_interleaved = true;  // Default to interleaved mode
    
    // Check if user specified a specific mode via environment variable or parameter
    const char* mode_env = std::getenv("SUFFIX_MODE");
    if (mode_env && std::string(mode_env) == "static") {
        use_interleaved = false;
    }

    LOG_INF("Using %s draft strategy\n", use_interleaved ? "interleaved" : "suffix-only");

    // Fill up context with tokens from user input
    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    llama_decode(ctx, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(), 1));

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept = 0;
    
    // Enhanced verify list to track retrieval time and draft length
    std::vector<std::tuple<int, int, int64_t, int64_t, int64_t, std::string>> verify_list;

    int n_past = inp.size();
    bool has_eos = false;

    struct common_sampler* smpl = common_sampler_init(model, params.sampling);

    std::vector<llama_token> draft;
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    // Counter for no drafted forward times
    int n_no_draft_forward = 0;

    int match_n = 0;
    std::string source = "X";
    std::string last_source = "X";

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        int i_dft = 0;
        int accept_length = 1;

        // Track verification timing
        const auto t_verify_start = ggml_time_us();

        while (true) {
            // sample from the target model
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

            if (i_dft < (int)draft.size() && id == draft[i_dft]) {
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
            else {
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
            break;
        }

        // End verification timing
        const auto t_verify_end = ggml_time_us();
        const int64_t verify_time_us = t_verify_end - t_verify_start;

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

        // Draft already contains a single token sampled from the model
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        const size_t original_draft_size = draft.size();

        // Draft retrieval timing
        const int64_t t_start_draft_us = ggml_time_us();
        
        last_source = source;

        // Use the specified draft strategy
        std::pair<int, char> result;
        if (use_interleaved) {
            result = draft_interleaved(suffix_array, inp, draft, n_draft, params.ngram_min, params.ngram_max);
        } else {
            result = draft_from_suffix_array(suffix_array, inp, draft, n_draft, params.ngram_min, params.ngram_max);
        }

        match_n = result.first;
        source = std::string(1, result.second);

        const int64_t t_end_draft_us = ggml_time_us();
        const int64_t draft_time_us = t_end_draft_us - t_start_draft_us;

        // Record metrics: match_n, accept_length, draft_size, draft_time_us, verify_time_us, source
        verify_list.emplace_back(match_n, accept_length, draft.size() - original_draft_size, 
                                draft_time_us, verify_time_us, last_source);

        for (size_t i = 1; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        }

        n_drafted += draft.size() - 1;

        llama_decode(ctx, batch_tgt);
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    // Compute statistics
    float sum = 0;
    float sum_prompt = 0;
    float sum_static = 0;
    int count_prompt = 0;
    int count_static = 0;

    int64_t total_draft_time = 0;
    int64_t total_verify_time = 0;
    int64_t total_draft_size = 0;

    for (const auto& entry : verify_list) {
        int accept_len = std::get<1>(entry);
        int64_t draft_size = std::get<2>(entry);
        int64_t draft_time = std::get<3>(entry);
        int64_t verify_time = std::get<4>(entry);
        const std::string& source = std::get<5>(entry);

        sum += accept_len;
        total_draft_time += draft_time;
        total_verify_time += verify_time;
        total_draft_size += draft_size;

        if (source == "P") {
            sum_prompt += accept_len;
            count_prompt++;
        } else if (source == "S") {
            sum_static += accept_len;
            count_static++;
        }
    }

    float average = verify_list.empty() ? 0 : sum / verify_list.size();
    float average_prompt = count_prompt > 0 ? sum_prompt / count_prompt : 0;
    float average_static = count_static > 0 ? sum_static / count_static : 0;

    // Convert to milliseconds for readability
    float avg_draft_time = verify_list.empty() ? 0 : (float)total_draft_time / verify_list.size() / 1000.0f;
    float avg_verify_time = verify_list.empty() ? 0 : (float)total_verify_time / verify_list.size() / 1000.0f;
    float avg_draft_size = verify_list.empty() ? 0 : (float)total_draft_size / verify_list.size();

    std::string verify_list_str = "[";
    for (size_t i = 0; i < verify_list.size(); ++i) {
        auto [match_n, accept_len, draft_size, draft_time_us, verify_time_us, source] = verify_list[i];
        verify_list_str += "(" + std::to_string(match_n) + "," + std::to_string(accept_len) + "," + std::to_string(draft_size) + "," + std::to_string(draft_time_us) + "," + std::to_string(verify_time_us) + "," + source + ")";
        if (i < verify_list.size() - 1) {
            verify_list_str += ", ";
        }
    }
    verify_list_str += "]";

    LOG_INF("Draft strategy: %s\n", use_interleaved ? "interleaved" : "suffix-only");
    LOG_INF("Accept length average: %.3f\n", average);
    LOG_INF("Prompt avg: %.3f, Static avg: %.3f\n", average_prompt, average_static);
    LOG_INF("Timing Analysis: avg draft time = %.3f ms, avg verify time = %.3f ms, avg draft size = %.1f\n", 
            avg_draft_time, avg_verify_time, avg_draft_size);
    LOG_INF("No draft forward times: %d\n", n_no_draft_forward);
    LOG_INF("Verify list: %s\n", verify_list_str.c_str());

    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            // Format to match parse_perf.py expectations
            output_file << "# Tokens: " << n_predict << ", Speed: "
                        << (n_predict / ((t_dec_end - t_dec_start) / 1e6f)) << " t/s, # Forward: "
                        << verify_list.size() << ", # No draft forward: " << n_no_draft_forward << "\n";
            output_file << "Accept length average: " << average << "\n";
            output_file << "Verify list (match_n, accept_length, draft_size, draft_time_us, verify_time_us): " << verify_list_str << "\n";
            output_file << "Average draft time per forward: " << avg_draft_time * 1000 << " us\n";
            output_file << generated_text.str();
            output_file.close();
            LOG_INF("Generated text written to: %s\n", params.out_file.c_str());
        }
    }

    LOG_INF("\n");
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("n_accept     = %d\n", n_accept);
    LOG_INF("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\ntarget:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_backend_free();

    LOG("\n\n");

    return 0;
} 