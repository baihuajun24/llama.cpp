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

// Base version
// void common_ngram_cache_draft(const std::vector<llama_token> & inp,
//                                std::vector<llama_token> & draft,
//                                int n_draft,
//                                int ngram_min,
//                                int ngram_max) {

//     const int inp_size = inp.size();
//     LOG_INF("[draft] inp_size = %d, n_draft = %d, ngram_min = %d, ngram_max = %d\n", inp_size, n_draft, ngram_min, ngram_max);

//     if (inp_size < ngram_min) {
//         LOG_INF("[draft] Early return: inp_size < ngram_min\n");
//         return;
//     }

//     const int max_n = std::min(ngram_max, inp_size);
//     LOG_INF("[draft] max_n = %d\n", max_n);

//     for (int n = max_n; n >= ngram_min; --n) {
//         const int start_pos = inp_size - n;
//         LOG_INF("[draft] Trying n = %d, start_pos = %d\n", n, start_pos);

//         std::vector<llama_token> suffix(inp.begin() + start_pos, inp.end());

//         LOG_INF("[draft] Suffix:");
//         for (auto t : suffix) {
//             LOG_INF(" %d", t);
//         }
//         LOG_INF("\n");

//         for (int i = 0; i < start_pos; ++i) {
//             bool match = true;
//             for (int j = 0; j < n; ++j) {
//                 if (inp[i + j] != suffix[j]) {
//                     match = false;
//                     break;
//                 }
//             }

//             if (match) {
//                 LOG_INF("[draft] Match found at i = %d for n = %d\n", i, n);
//                 int collected = 0;
//                 for (int j = 0; j < n_draft && (i + n + j) < inp_size; ++j) {
//                     llama_token token = inp[i + n + j];
//                     draft.push_back(token);
//                     LOG_INF("[draft] Drafted token[%d] = %d\n", j, token);
//                     collected++;
//                 }
//                 LOG_INF("[draft] Total %d tokens drafted\n", collected);
//                 return;
//             }
//         }
//     }

//     LOG_INF("[draft] No match found, draft remains empty\n");
// }

void common_ngram_cache_draft(const std::vector<llama_token> & inp,
                               std::vector<llama_token> & draft,
                               int n_draft,
                               int ngram_min,
                               int ngram_max) {
                                
    const int inp_size = inp.size();
    LOG_INF("[draft] inp_size = %d, n_draft = %d, ngram_min = %d, ngram_max = %d\n", inp_size, n_draft, ngram_min, ngram_max);

    if (inp_size < ngram_min) {
        LOG_INF("[draft] Early return: inp_size < ngram_min\n");
        return;
    }

    const int max_n = std::min(ngram_max, inp_size);
    LOG_INF("[draft] max_n = %d\n", max_n);

    std::unordered_map<llama_token, int> freq;
    std::unordered_map<llama_token, std::vector<int>> indices;

    for (int n = max_n; n >= ngram_min; --n) {
        const int start_pos = inp_size - n;
        std::vector<llama_token> suffix(inp.begin() + start_pos, inp.end());

        LOG_INF("[draft] Trying n = %d, start_pos = %d\n", n, start_pos);
        LOG_INF("[draft] Suffix:");
        for (auto t : suffix) LOG_INF(" %d", t);
        LOG_INF("\n");

        for (int i = 0; i < start_pos; ++i) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (inp[i + j] != suffix[j]) {
                    match = false;
                    break;
                }
            }

            if (match && (i + n) < inp_size) {
                llama_token next_token = inp[i + n];
                freq[next_token]++;
                indices[next_token].push_back(i + n);  // record the starting index of continuation

                LOG_INF("[draft] Match at i = %d: first draft token = %d, freq = %d\n",
                        i, next_token, freq[next_token]);
            }
        }

        if (!freq.empty()) break;  // stop early once we get candidates for some n
    }

    if (freq.empty()) {
        LOG_INF("[draft] No match found, draft remains empty\n");
        return;
    }

    // Find the most frequent first token(s)
    int max_freq = 0;
    std::vector<llama_token> top_tokens;
    for (const auto &kv : freq) {
        if (kv.second > max_freq) {
            max_freq = kv.second;
            top_tokens.clear();
            top_tokens.push_back(kv.first);
        } else if (kv.second == max_freq) {
            top_tokens.push_back(kv.first);
        }
    }

    // Random tie-breaking
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, top_tokens.size() - 1);
    llama_token chosen_token = top_tokens[dis(gen)];

    LOG_INF("[draft] Chosen first token = %d with freq = %d\n", chosen_token, max_freq);

    // Randomly select one of the continuation indices for the chosen token
    const std::vector<int>& starts = indices[chosen_token];
    if (!starts.empty()) {
        std::uniform_int_distribution<> dis2(0, starts.size() - 1);
        int chosen_start = starts[dis2(gen)];

        for (int j = 0; j < n_draft && (chosen_start + j) < inp_size; ++j) {
            draft.push_back(inp[chosen_start + j]);
            LOG_INF("[draft] Final drafted token[%d] = %d\n", j, inp[chosen_start + j]);
        }
    }
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
    std::vector<int> n_accept_list;

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
            // use LOG_INF to check the target token and the draft
            // LOG_INF("0428 Check: target string = %s, draft = %s\n", token_str.c_str(), common_token_to_piece(ctx, draft[i_dft]).c_str());
            // check if the target token matches the draft
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
        n_accept_list.push_back(accept_length);
        
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

        common_ngram_cache_draft(inp, draft, n_draft, params.ngram_min, params.ngram_max);
        
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

    // print n_accept_list and average
    float sum = 0;
    for (size_t i = 0; i < n_accept_list.size(); i++) {
        sum += n_accept_list[i];
    }
    float average = sum / n_accept_list.size();

    // Format the vector as a string
    std::string accept_list_str = "[";
    for (size_t i = 0; i < n_accept_list.size(); i++) {
        accept_list_str += std::to_string(n_accept_list[i]);
        if (i < n_accept_list.size() - 1) {
            accept_list_str += ", ";
        }
    }
    accept_list_str += "]";
    LOG_INF("0420 Check: len is %d, n_accept_list = %s\n", n_accept_list.size(), accept_list_str.c_str());
    LOG_INF("0420 Check: accept length average      = %.3f\n", average);
    LOG_INF("0428 Check: no draft is suppiled forward times = %d\n", n_no_draft_forward);

    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            // First write the statistics as the first 3 lines
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                    << (n_predict  / ((t_dec_end - t_dec_start) / 1e6f)) << " t/s, # Forward: " 
                    << n_accept_list.size() << "\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Accept length list: " << accept_list_str << "\n";
            // Write all collected text at once
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