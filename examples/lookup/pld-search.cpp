#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>

#include <iostream>
#include <iomanip>  // for std::fixed, std::setprecision

// Base version // don't do draft.clear() !!!!
int common_ngram_cache_draft(const std::vector<llama_token> & search_space,
                              const std::vector<llama_token> & match_key,
                              std::vector<llama_token> & draft,
                              int n_draft,
                              int ngram_min,
                              int ngram_max) {
    const int search_size = search_space.size();
    const int match_size = match_key.size();

    if (match_size < ngram_min || search_size < ngram_min) {
        return 0;
    }

    const int max_n = std::min(ngram_max, match_size);

    for (int n = max_n; n >= ngram_min; --n) {
        // Take the last n tokens of match_key as query
        std::vector<llama_token> suffix(match_key.end() - n, match_key.end());

        for (int i = 0; i <= search_size - n; ++i) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (search_space[i + j] != suffix[j]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                // Draft up to n_draft tokens after the match
                for (int j = 0; j < n_draft && (i + n + j) < search_size; ++j) {
                    draft.push_back(search_space[i + n + j]);
                }

                return n;
            }
        }
    }

    return 0; // No match found
}


// Advanced version with frequency-based ranking
// Helper function to select best candidate based on token frequency ranking
std::vector<llama_token> select_best_candidate(const std::vector<llama_token>& search_space,
                                             const std::vector<int>& match_positions,
                                             int n,
                                             int n_draft) {
    if (match_positions.empty()) {
        return {};
    }

    const int search_size = search_space.size();
    
    // Find valid match positions (those with at least one continuation token)
    std::vector<std::pair<int, int>> extend_match_positions; // [pos, count]
    for (int pos : match_positions) {
        if (pos + n < search_size) {
            extend_match_positions.push_back({pos, 0}); // count will be updated
        }
    }

    if (extend_match_positions.empty()) {
        return {};
    }

    if (extend_match_positions.size() == 1) {
        // Single candidate, extract directly
        std::vector<llama_token> result;
        int pos = extend_match_positions[0].first;
        for (int j = 0; j < n_draft && (pos + n + j) < search_size; ++j) {
            result.push_back(search_space[pos + n + j]);
        }
        return result;
    }

    // Iteratively filter by token frequency at each position
    for (int offset = 1; offset <= n_draft && extend_match_positions.size() > 1; ++offset) {
        // Count frequency of tokens at current offset position
        std::unordered_map<llama_token, int> token_freq;
        
        for (auto& pair : extend_match_positions) {
            int pos = pair.first;
            int actual_pos = pos + n + offset - 1; // offset-1 because we start from offset=1
            if (actual_pos < search_size) {
                token_freq[search_space[actual_pos]]++;
            }
        }

        if (token_freq.empty()) {
            break; // No more tokens to compare
        }

        // Update counts in extend_match_positions
        for (auto& pair : extend_match_positions) {
            int pos = pair.first;
            int actual_pos = pos + n + offset - 1;
            if (actual_pos < search_size) {
                pair.second = token_freq[search_space[actual_pos]];
            } else {
                pair.second = 0; // No token at this position
            }
        }

        // Find the maximum frequency (1st place)
        int max_freq = 0;
        for (const auto& pair : extend_match_positions) {
            max_freq = std::max(max_freq, pair.second);
        }

        // Remove all non-1st items (keep only those with max frequency)
        std::vector<std::pair<int, int>> filtered_positions;
        for (const auto& pair : extend_match_positions) {
            if (pair.second == max_freq) {
                filtered_positions.push_back(pair);
            }
        }

        extend_match_positions = filtered_positions;
    }

    // Extract result from the single remaining position
    if (!extend_match_positions.empty()) {
        int single_pos_left = extend_match_positions[0].first;
        std::vector<llama_token> result;
        for (int j = 0; j < n_draft && (single_pos_left + n + j) < search_size; ++j) {
            result.push_back(search_space[single_pos_left + n + j]);
        }
        return result;
    }
    // Fallback (should not reach here)
    return {};
}

int common_ngram_cache_draft_advanced(const std::vector<llama_token> & search_space,
                                    const std::vector<llama_token> & match_key,
                                    std::vector<llama_token> & draft,
                                    int n_draft,
                                    int ngram_min,
                                    int ngram_max) {
    const int search_size = search_space.size();
    const int match_size = match_key.size();

    if (match_size < ngram_min || search_size < ngram_min) {
        return 0;
    }

    const int max_n = std::min(ngram_max, match_size);

    for (int n = max_n; n >= ngram_min; --n) {
        // Take the last n tokens of match_key as query
        std::vector<llama_token> suffix(match_key.end() - n, match_key.end());

        // Find all match positions for this n-gram
        std::vector<int> match_positions;
        for (int i = 0; i <= search_size - n; ++i) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (search_space[i + j] != suffix[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                match_positions.push_back(i);
            }
        }

        if (match_positions.empty()) {
            continue; // Try shorter n-gram
        }

        // Check if any match position has valid continuation
        bool has_valid_continuation = false;
        for (int pos : match_positions) {
            if (pos + n < search_size) {
                has_valid_continuation = true;
                break;
            }
        }

        if (!has_valid_continuation) {
            continue; // No valid continuations found
        }

        // Find the best candidate using frequency-based ranking
        std::vector<llama_token> best_candidate = select_best_candidate(search_space, match_positions, n, n_draft);
        
        // Copy the best candidate to draft
        draft.insert(draft.end(), best_candidate.begin(), best_candidate.end());
        return n;
    }

    return 0; // No match found
}

// Interleaved version that searches prompt first, then static cache for each n-gram length
// Returns a pair: {match_n, source} where source is 'P' for prompt, 'S' for static, 'X' for no match
std::pair<int, char> common_ngram_cache_interleave(const std::vector<llama_token> & static_cache,
                                                  const std::vector<llama_token> & inp,
                                                  std::vector<llama_token> & draft,
                                                  int n_draft,
                                                  int ngram_min,
                                                  int ngram_max) {
    const int inp_size = inp.size();
    const int static_size = static_cache.size();

    if (inp_size < ngram_min || static_size < ngram_min) {
        return {0, 'X'};
    }

    // Create search space from prompt history (excluding last ngram_min tokens to avoid overlap)
    std::vector<llama_token> search_space;
    if (inp_size > ngram_min) {
        search_space = std::vector<llama_token>(inp.begin(), inp.end() - ngram_min);
    }

    const int max_n = std::min(ngram_max, inp_size);

    for (int n = max_n; n >= ngram_min; --n) {
        // Take the last n tokens of inp as query
        std::vector<llama_token> suffix(inp.end() - n, inp.end());

        // First, try to find match in prompt history (search_space)
        if (!search_space.empty() && (int)search_space.size() >= n) {
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

        // If not found in prompt history, try static cache with advanced selection
        if (static_size >= n) {
            // Find all match positions in static cache
            std::vector<int> match_positions;
            for (int i = 0; i <= static_size - n; ++i) {
                bool match = true;
                for (int j = 0; j < n; ++j) {
                    if (static_cache[i + j] != suffix[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    match_positions.push_back(i);
                }
            }

            if (!match_positions.empty()) {
                // Check if any match position has valid continuation
                bool has_valid_continuation = false;
                for (int pos : match_positions) {
                    if (pos + n < static_size) {
                        has_valid_continuation = true;
                        break;
                    }
                }

                if (has_valid_continuation) {
                    // Use advanced selection for static cache
                    std::vector<llama_token> best_candidate = select_best_candidate(static_cache, match_positions, n, n_draft);
                    
                    // Copy the best candidate to draft
                    draft.insert(draft.end(), best_candidate.begin(), best_candidate.end());
                    return {n, 'S'}; // Found in static cache
                }
            }
        }
    }

    return {0, 'X'}; // No match found in either source
}


std::vector<llama_token> load_static_token_cache() {
    LOG_INF("[0528 start load_static_token_cache]");
    const std::string file_path = "C:/Users/Administrator/Documents/ngram-spec/outputs/code/magpie-all-tokens.bin"; // enlarge dataset
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        LOG_ERR("[load_static_token_cache] Failed to open static token cache: %s\n", file_path.c_str());
        exit(1);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % sizeof(int32_t) != 0) {
        LOG_ERR("[load_static_token_cache] Unexpected file size (%ld): not divisible by sizeof(int32_t)\n", static_cast<long>(file_size));
        exit(1);
    }

    size_t num_tokens = file_size / sizeof(int32_t);
    std::vector<llama_token> tokens(num_tokens);

    file.read(reinterpret_cast<char*>(tokens.data()), file_size);
    if (!file) {
        LOG_ERR("[load_static_token_cache] Failed to read full token data from file: %s\n", file_path.c_str());
        exit(1);
    }

    double disk_mb = file_size / (1024.0 * 1024.0);
    double ram_mb  = (tokens.size() * sizeof(llama_token)) / (1024.0 * 1024.0);

    LOG_INF("[load_static_token_cache] Static token cache loaded\n");
    LOG_INF("  Total tokens     : %zu\n", num_tokens);
    LOG_INF("  Disk file size   : %.2f MB\n", disk_mb);
    LOG_INF("  RAM usage        : %.2f MB\n", ram_mb);
    LOG_INF("  File path        : %s\n", file_path.c_str());

    return tokens;
}

void test_pld_search(const std::vector<llama_token> & static_cache, llama_context * ctx) {
    // Define the test string
    std::string test_string = "Given a list of temperatures";

    // Tokenize the input string
    std::vector<llama_token> test_ids = common_tokenize(ctx, test_string, true, true);
    test_ids.erase(test_ids.begin());  // remove first element, fixed as 128000
    LOG_INF("Test string: \"%s\"\n", test_string.c_str());

    // Print token IDs
    LOG_INF("Encoded token IDs: [");
    for (size_t i = 0; i < test_ids.size(); ++i) {
        LOG("%d", test_ids[i]);
        if (i < test_ids.size() - 1) LOG(", ");
    }
    LOG("]\n");

    // Search static cache with test_ids as input
    std::vector<llama_token> draft;
    int match_n = common_ngram_cache_draft(static_cache, test_ids, draft, /*n_draft=*/10, /*ngram_min=*/2, /*ngram_max=*/6);

    LOG_INF("Match n-gram length: %d\n", match_n);

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

    std::string target_string = "Given a list of temperatures in a week, determine the number of days it takes for the temperature to increase from a specific temperature to a higher temperature.";
    std::vector<llama_token> target_ids = common_tokenize(ctx, target_string, true, true);
    LOG_INF("Target string: \"%s\"\n", target_string.c_str());
    // Print token IDs
    LOG_INF("[target retrieval] Encoded token IDs (%zu tokens):", target_ids.size());
    for (size_t i = 0; i < target_ids.size(); ++i) {
        LOG("%d ", target_ids[i]);
        if ((i + 1) % 20 == 0) LOG("\n");  // Newline every 20 tokens
    }
    LOG("\n");
}


int main(int argc, char ** argv){
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

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);

    // load the static cache
    std::vector<llama_token> static_cache = load_static_token_cache();

    // This is for testing
    // LOG_INF("V2: first 1000 tokens decode:\n");
    // std::string preview_text;
    // for (size_t i = 0; i < std::min(static_cache.size(), size_t(1000)); ++i) {
    //     preview_text += common_token_to_piece(ctx, static_cache[i]);
    // }
    // LOG("%s\n", preview_text.c_str());

    // test_pld_search(static_cache, ctx);
    // return 0;

    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    // Fill up context ngram cache with tokens from user input:

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1));

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    // a list for n_accept
    std::vector<std::tuple<int, int, std::string>> verify_list;

    int n_past = inp.size();

    bool has_eos = false;

    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    std::vector<llama_token> draft;

    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    // a new counter for no drafted forward times
    int n_no_draft_forward = 0;

    int match_n = 0;
    int last_match_n = -1;
    std::string source = "X";
    std::string last_source = "X";

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        // print current draft sequence
        // LOG_DBG("drafted %s\n", string_from(ctx, draft).c_str());

        int i_dft = 0;
        int accept_length = 1;

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
            // for debug usage

            // int start_ngram = std::max(0, (int)inp.size() - params.ngram_max);
            // std::string recent_tokens = "[";
            // for (int i = start_ngram; i < (int)inp.size(); ++i) {
            //     recent_tokens += std::to_string(inp[i]);
            //     if (i < (int)inp.size() - 1) recent_tokens += ", ";
            // }
            // recent_tokens += "]";
            // if (i_dft < (int) draft.size()){
            //     LOG_INF("[0528 verify check] Last %d tokens in inp: %s; current id: %d; draft[%d] = %d\n",
            //             params.ngram_max, recent_tokens.c_str(), id, i_dft, draft[i_dft]);
            // }

            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                //LOG_DBG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());

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
                if ((int) draft.size() == 0) {
                    n_no_draft_forward += 1;
                    LOG_INF("[0428 Check] no draft caused accept length = %d\n", accept_length);
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
        last_match_n = match_n;
        last_source = source;
        // source = "X";
        // std::vector<llama_token> search_space;

        // Use interleaved search: prompt first, then static cache for each n-gram length
        auto result = common_ngram_cache_interleave(static_cache, inp, draft, n_draft, params.ngram_min, params.ngram_max); // use a larger ngram_max to test performance
        match_n = result.first;
        source = std::string(1, result.second); 

        // // Search draft in prompt+answer first
        // if (inp.size() > params.ngram_min) {
        //     search_space = std::vector<llama_token>(inp.begin(), inp.end() - params.ngram_min);
        //     match_n = common_ngram_cache_draft(search_space, inp, draft, n_draft, params.ngram_min, params.ngram_max);
        // }
        
        // if (draft.size() == 1) {
        //     // LOG_INF("[draft fallback] Draft size is 1, falling back to static_cache\n");
        //     match_n = common_ngram_cache_draft_advanced(static_cache, inp, draft, n_draft, params.ngram_min, params.ngram_max); // use a larger ngram_max for static?
        //     // match_n = common_ngram_cache_draft(static_cache, inp, draft, n_draft, params.ngram_min, params.ngram_max);

        //     std::string draft_str = "[";
        //     for (size_t i = 0; i < draft.size(); ++i) {
        //         draft_str += std::to_string(draft[i]);
        //         if (i < draft.size() - 1) {
        //             draft_str += ", ";
        //         }
        //     }
        //     draft_str += "]";

        //     // LOG_INF("[0528 Draft from Static]: %s\n", draft_str.c_str());
        //     source = match_n > 0 ? "S" : "X";
        // } else {
        //     source = "P";
        // }

        verify_list.emplace_back(last_match_n, accept_length, last_source);

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

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    // Compute overall average accept length
    float sum = 0;
    float sum_prompt = 0;
    float sum_static = 0;
    int count_prompt = 0;
    int count_static = 0;

    for (const auto &entry : verify_list) {
        int accept_len = std::get<1>(entry);
        const std::string &source = std::get<2>(entry);

        sum += accept_len;

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

    std::string verify_list_str = "[";
    for (size_t i = 0; i < verify_list.size(); ++i) {
        auto [match_n, accept_len, source] = verify_list[i];
        verify_list_str += "(" + std::to_string(match_n) + "," + std::to_string(accept_len) + "," + source + ")";
        if (i < verify_list.size() - 1) {
            verify_list_str += ", ";
        }
    }
    verify_list_str += "]";

    LOG_INF("0420 Check: len is %zu, verify_list = %s\n", verify_list.size(), verify_list_str.c_str());
    LOG_INF("0420 Check: accept length average     = %.3f\n", average);
    LOG_INF("0420 Check: prompt avg                = %.3f, static avg = %.3f\n", average_prompt, average_static);
    LOG_INF("0428 Check: no draft is supplied forward times = %d\n", n_no_draft_forward);

    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            output_file << "# Tokens: " << n_predict << ", Speed: "
                        << (n_predict / ((t_dec_end - t_dec_start) / 1e6f)) << " t/s, # Forward: "
                        << verify_list.size() << "\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Accept Length for Prompt: " << std::fixed << std::setprecision(2)
                        << average_prompt << "; Accept Length for Static: " << average_static << "\n";
            output_file << "# Verify list (match_n, accept_length, source): " << verify_list_str << "\n";
            output_file << generated_text.str();
            output_file.close();
            LOG_INF("Generated text written to: %s\n", params.out_file.c_str());
        }
    }

    LOG_INF("\n");
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_INF("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
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