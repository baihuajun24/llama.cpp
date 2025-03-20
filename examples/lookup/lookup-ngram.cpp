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

int main(int argc, char ** argv){
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    common_init();

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

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        // Fill up context ngram cache with tokens from user input:
        const int64_t t_start_draft_us = ggml_time_us();
        common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size(), false);

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

    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1));  // First decode: Initial encoding of all tokens in the prompt except the last one
    // MY NOTE: this is weird because it breaks prefill into n-1 and 1
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1));  // Second decode: Process just the last token of the prompt separately (likely for proper KV cache setup)


    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    bool has_eos = false;

    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    std::vector<llama_token> draft;

    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    while (true) { // this loop ends when n_predict > params.n_predict or has_eos is true
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        // print current draft sequence
        LOG_DBG("drafted %s\n", string_from(ctx, draft).c_str());

        int i_dft = 0;
        while (true) { // This loop checks each draft token against the target model's output
            // Sample from the target model to get the "true" next token
            llama_token id = common_sampler_sample(smpl, ctx, i_dft);

            common_sampler_accept(smpl, id, true);

            const std::string token_str = common_token_to_piece(ctx, id);

            // Print the token if colors are disabled
            if (!params.use_color) {
                LOG("%s", token_str.c_str());
            }

            // Check for end of generation token
            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            // TOKEN VERIFICATION: Check if the current token matches our draft
            // Example: If draft is [A,B,C,D] and target model generates [A,B,E,F]
            // Then A and B will match, but E won't match C and we need to break
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                // TOKEN MATCHED: The model's prediction matches our draft token
                LOG_DBG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                
                // Increment acceptance stats
                ++n_accept;
                
                // Update KV cache position
                ++n_past;
                
                // Move to next draft token for checking
                ++i_dft;
                
                // Add this token to input history
                inp.push_back(id);
                {
                    // Update n-gram cache with this token
                    const int64_t t_start_draft_us = ggml_time_us();
                    common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }

                // Visual indication for matched token
                if (params.use_color) {
                    LOG("\033[34m%s\033[0m", token_str.c_str());
                    fflush(stdout);
                }
                
                // Continue checking the next token in sequence
                continue;
            } else {
                // TOKEN MISMATCH or END OF DRAFT: We got a token that doesn't match our draft
                // For example: If draft=[A,B,C,D] and model generates [A,B,E,...], we'll hit this
                // when comparing E vs C. We accept E (the model's token) and break the sequence.
                
                // Print the non-matching token
                if (params.use_color) {
                    LOG("%s", token_str.c_str());
                }
                fflush(stdout);

                // Log the mismatch
                LOG_DBG("the sampled target token (%d, '%s') did not match draft token %d - stopping draft\n", 
                       id, token_str.c_str(), i_dft < (int)draft.size() ? draft[i_dft] : -1);

                // We ALWAYS use the target model's token (id) in case of mismatch
                // Start a new draft with this token
                draft.clear();
                draft.push_back(id);
                inp.push_back(id);
                {
                    // Update n-gram cache with this new token
                    const int64_t t_start_draft_us = ggml_time_us();
                    common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
                
                // Exit the inner loop - we'll start a new draft sequence
                break;
            }
        }

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);

        // Important: At this point, draft[0] already contains either:
        // 1. A token that was just accepted from the target model output (in case of mismatch)
        // 2. The next token to process after accepting all draft tokens
        // In both cases, we don't need to recompute this token - we use it directly

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

        // Draft already contains a single token sampled from the model:
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        const int64_t t_start_draft_us = ggml_time_us();

        common_ngram_cache_draft(inp, draft, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ngram_cache_context, ngram_cache_dynamic, ngram_cache_static);

        for (size_t i = 1; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
        }

        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;

        llama_decode(ctx, batch_tgt);  // Third decode: Verify the draft tokens against the language model - this is the critical verification step
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_dec_end = ggml_time_us();

    // Update dynamic ngram cache with context ngram cache and save it to disk:
    common_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_context);
    common_ngram_cache_save(ngram_cache_dynamic, params.lookup_cache_dynamic);

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

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
