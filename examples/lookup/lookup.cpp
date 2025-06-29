#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-cache.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

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

    common_ngram_cache ngram_cache_context; // prompt + generated text
    common_ngram_cache ngram_cache_dynamic; // save to dynamic after this run, possible for reload and reuse for next run   
    common_ngram_cache ngram_cache_static; // static cache built from external database
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        // Fill up context ngram cache with tokens from user input:
        const int64_t t_start_draft_us = ggml_time_us();
        common_ngram_cache_update(ngram_cache_context, params.ngram_min, params.ngram_max, inp, inp.size(), false);

        if (!params.lookup_cache_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(params.lookup_cache_static);
            } catch (std::ifstream::failure const &) {
                LOG_ERR("failed to open static lookup cache: %s", params.lookup_cache_static.c_str());
                exit(1);
            }
        }

        if (!params.lookup_cache_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(params.lookup_cache_dynamic);
                // LOG_INF("0428 Check: loaded ngram_cache from this file: %s\n", params.lookup_cache_dynamic.c_str());
            } catch (std::ifstream::failure const &) {} // if the file does not exist it will simply be created at the end of the program
        }

        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
    }

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

    // Add a verification list to track (match_n, accept_length, draft_size, draft_time_us, verify_time_us)
    std::vector<std::tuple<int, int, int64_t, int64_t, int64_t>> verify_list;

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
                {
                    // Update context ngram cache with the newly accepted token:
                    const int64_t t_start_draft_us = ggml_time_us();
                    common_ngram_cache_update(ngram_cache_context, params.ngram_min, params.ngram_max, inp, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }

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
                else{
                    // LOG_INF("[0428 Check] draft not matched. should be %d, but is %d. caused accept length = %d\n", id, draft[i_dft], accept_length);
                }
            }

            if (write_to_file) {
                generated_text << token_str;
            }


            // LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            {
                // Update context ngram cache with the newly accepted token:
                const int64_t t_start_draft_us = ggml_time_us();
                common_ngram_cache_update(ngram_cache_context, params.ngram_min, params.ngram_max, inp, 1, false);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }
        
        // End verification timing
        const auto t_verify_end = ggml_time_us();
        const int64_t verify_time_us = t_verify_end - t_verify_start;
        
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
        const size_t original_draft_size = draft.size(); // Store original draft size before ngram_cache_draft
        common_ngram_cache_draft(inp, draft, n_draft, params.ngram_min, params.ngram_max, ngram_cache_context, ngram_cache_dynamic, ngram_cache_static);
        const int64_t draft_time_us = ggml_time_us() - t_start_draft_us;

        // Record to verify_list: (match_n, accept_length, draft_size, draft_time_us, verify_time_us)
        // For lookup, match_n represents the number of tokens that matched (i_dft)
        int match_n = i_dft;
        verify_list.push_back(std::make_tuple(match_n, accept_length, draft.size() - original_draft_size, draft_time_us, verify_time_us));

        // LOG_INF to check last ngram_max tokens of inp and first ngram_max tokens of draft
        {
            // Create a vector of the last ngram_max tokens from inp
            std::vector<llama_token> last_inp_tokens;
            size_t start_idx = (inp.size() >= params.ngram_max) ? (inp.size() - params.ngram_max) : 0;
            last_inp_tokens.insert(last_inp_tokens.end(), inp.begin() + start_idx, inp.end());
            // LOG_INF("0427 Check: last %d tokens of inp\n", (int)(inp.size() - start_idx));
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

    // Update dynamic ngram cache with context ngram cache and save it to disk if path supplied
    if (!params.lookup_cache_dynamic.empty()) {
        common_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_context);
        LOG_INF("0428 Check: writting ngram_cache to this file:  %s\n", params.lookup_cache_dynamic.c_str());
        common_ngram_cache_save(ngram_cache_dynamic, params.lookup_cache_dynamic);
    }

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

    // Calculate accept length average from verify_list (for consistency with speculative-simple)
    float verify_sum = 0;
    for (size_t i = 0; i < verify_list.size(); i++) {
        verify_sum += std::get<1>(verify_list[i]); // Access the accept_length part
    }
    float verify_average = verify_list.empty() ? 0 : verify_sum / verify_list.size();

    // Format the verify_list as a string
    std::string verify_list_str = "[";
    for (size_t i = 0; i < verify_list.size(); i++) {
        verify_list_str += "(" + std::to_string(std::get<0>(verify_list[i])) + "," + 
                          std::to_string(std::get<1>(verify_list[i])) + "," +
                          std::to_string(std::get<2>(verify_list[i])) + "," +
                          std::to_string(std::get<3>(verify_list[i])) + "," +
                          std::to_string(std::get<4>(verify_list[i])) + ")";
        if (i < verify_list.size() - 1) {
            verify_list_str += ", ";
        }
    }
    verify_list_str += "]";

    LOG_INF("0420 Check: len is %d, verify_list = %s\n", (int)verify_list.size(), verify_list_str.c_str());
    LOG_INF("0420 Check: verify accept length average = %.3f\n", verify_average);

    // Calculate average draft size
    float total_draft_size = 0;
    for (const auto& item : verify_list) {
        total_draft_size += std::get<2>(item);
    }
    float avg_draft_size = verify_list.empty() ? 0 : total_draft_size / verify_list.size();
    LOG_INF("avg_draft_size = %.3f\n", avg_draft_size);

    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", params.out_file.c_str());
        } else {
            // First write the statistics as the first lines
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                    << (n_predict  / ((t_dec_end - t_dec_start) / 1e6f)) << " t/s, # Forward: " 
                    << verify_list.size() << ", n_draft: " << n_draft << "\n";
            output_file << "# Accept length average: " << verify_average << "\n";
            output_file << "# Verify list (match_n, accept_length, draft_size, draft_time_us, verify_time_us): " << verify_list_str << "\n";
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