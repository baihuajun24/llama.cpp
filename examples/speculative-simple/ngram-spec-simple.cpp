#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "ngram-cache.h"

# include <map>
# include <algorithm>
# include <string>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric> // For std::accumulate

static const int N_RETRIEVAL = 4;

static llama_tokens generate_draft_from_ngram(std::vector<llama_token>& prompt_tgt, common_ngram_cache& ngram_cache) {
    llama_tokens draft;
    
    // Try different n-gram lengths, from longest to shortest
    for (size_t n = N_RETRIEVAL; n > 0; --n) {
        if (prompt_tgt.size() >= n) {
            // Get the last n tokens from the prompt
            std::vector<llama_token> last_n_tokens(prompt_tgt.end() - n, prompt_tgt.end());

            // Get the n-gram candidates for the last n tokens
            auto it = ngram_cache.find(common_ngram(last_n_tokens.data(), n));
            if (it != ngram_cache.end() && !it->second.empty()) {
                // Find the candidate with the highest frequency
                auto best_candidate = std::max_element(
                    it->second.begin(), it->second.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; }
                );

                // Add the best candidate to the draft
                draft.push_back(best_candidate->first);
                break; // Exit the loop once a candidate is found
            }
        }
    }

    return draft;
}
static llama_tokens generate_dummy_draft_tokens(llama_context* ctx_tgt) {
    // Dummy string to tokenize
    std::string dummy_string = "Hillary Clinton";
    
    // Tokenize the dummy string
    llama_tokens draft = common_tokenize(ctx_tgt, dummy_string, true, true);
    return draft;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    //llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    params.devices      = params.speculative.devices;
    params.model        = params.speculative.model;
    params.n_ctx        = params.speculative.n_ctx;
    params.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers = params.speculative.n_gpu_layers;

    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);

    //model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        return 1;
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    int n_draft     = params.speculative.n_max;
    int n_draft_min = params.speculative.n_min;

    float p_min = params.speculative.p_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    struct common_speculative_params params_spec;
    params_spec.n_draft = n_draft;
    LOG_INF("\nKKKKKKKKKKKKKKKKKKKKKK params_spec.n_draft: %d\n", params_spec.n_draft);
    params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft;
    params_spec.p_min   = p_min;

    struct common_speculative * spec = common_speculative_init(ctx_dft);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    // Fill the n-gram cache with tokens from the prompt
    common_ngram_cache ngram_cache;
    const auto t_ngram_cache_start = ggml_time_us(); // Start timing
    common_ngram_cache_update(ngram_cache, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size(), false);
    const auto t_ngram_cache_end = ggml_time_us(); // End timing

    // Log the time taken to build the n-gram cache
    LOG_INF("Time taken to build n-gram cache from the prompt: %.3f seconds\n", (t_ngram_cache_end - t_ngram_cache_start) / 1e6f);

    // Log the contents of the n-gram cache
    LOG("\nContents of the n-gram cache:\n");
    std::map<std::string, std::vector<std::string>> sorted_entries;

    for (const auto& entry : ngram_cache) {
        // Detokenize the n-gram
        std::string ngram_text;
        for (const auto& token : entry.first.tokens) {
            if (token != -1) {
                ngram_text += common_token_to_piece(ctx_tgt, token) + " (" + std::to_string(token) + ") "; // Include token ID
            }
        }
        ngram_text = ngram_text.substr(0, ngram_text.size() - 1); // Remove trailing space

        // Prepare candidates
        std::vector<std::string> candidates_text;
        for (const auto& candidate : entry.second) {
            if (candidate.first != -1) {
                candidates_text.push_back(common_token_to_piece(ctx_tgt, candidate.first) + " (" + std::to_string(candidate.first) + ")"); // Detokenize candidate and include ID
            }
        }

        // Store in the map
        sorted_entries[ngram_text] = candidates_text; // Map n-gram text to its candidates
    }

    // Log sorted entries, limiting to the first 50
    int count = 0;
    for (const auto& [ngram, candidates] : sorted_entries) {
        LOG("%s → ", ngram.c_str());
        for (const auto& candidate : candidates) {
            LOG("%s ", candidate.c_str());
        }
        LOG("\n");

        count++;
        if (count >= 100) {
            break; // Limit to first 50 entries
        }
    }

    const auto t_dec_start = ggml_time_us(); // Decoding Phase Start
    std::vector<double> draft_times;
    std::vector<double> verify_times;
    std::vector<double> target_sample_times;
    std::vector<double> one_iter_times;
    while (true) {
        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        auto t_draft_start = ggml_time_us(); 
        // llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);
        // llama_tokens draft = generate_dummy_draft_tokens(ctx_tgt);
        llama_tokens draft = generate_draft_from_ngram(prompt_tgt, ngram_cache);

        auto t_draft_end = ggml_time_us(); 
        draft_times.push_back((t_draft_end - t_draft_start) / 1e3);
        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // LOG_INF to print [id_last, draft0, draft1, ..., draftN-1] to understand what draft sequence is
        // one each line print id, decode token from id
        LOG_INF("[check id_last]: %s, draft length: %zu\n", common_token_to_piece(ctx_tgt, id_last).c_str(), draft.size());
        for (size_t i = 0; i < draft.size(); i++) {
            LOG_INF("[check draft %zu]: %d: %s\n", i, draft[i], common_token_to_piece(ctx_dft, draft[i]).c_str());
        }
        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) n_draft_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());
            auto t_verify_start = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            auto t_verify_end = ggml_time_us();
            verify_times.push_back((t_verify_end - t_verify_start) / 1e3);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto t_target_sample_start = ggml_time_us();
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
        auto t_target_sample_end = ggml_time_us();
        target_sample_times.push_back((t_target_sample_end - t_target_sample_start) / 1e3);
        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
            // 0322: temporary disable to debug
            // if (params.use_color && i + 1 < ids.size()) {
            //     LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            // } else {
            //     LOG("%s", token_str.c_str());
            // }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_kv_self_seq_rm(ctx_tgt, 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
        auto t_one_iter_end = ggml_time_us();
        one_iter_times.push_back((t_one_iter_end - t_draft_start) / 1e3);
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");
    // After the while loop, calculate average times
    double avg_one_iter_time = std::accumulate(one_iter_times.begin(), one_iter_times.end(), 0.0) / one_iter_times.size();
    LOG_INF("Average one iter time: %.2f ms\n", avg_one_iter_time);

    // Calculate average drafting time and percentage
    double avg_drafting_time = std::accumulate(draft_times.begin(), draft_times.end(), 0.0) / draft_times.size();
    double drafting_percentage = (avg_drafting_time / avg_one_iter_time) * 100.0;
    LOG_INF("Average drafting time: %.2f ms (n_draft = %d, %.2f%% of one iter time)\n", avg_drafting_time, n_draft, drafting_percentage);

    // Calculate average verify time and percentage
    double avg_verify_time = std::accumulate(verify_times.begin(), verify_times.end(), 0.0) / verify_times.size();
    double verify_percentage = (avg_verify_time / avg_one_iter_time) * 100.0;
    LOG_INF("Average verify time: %.2f ms (%.2f%% of one iter time)\n", avg_verify_time, verify_percentage);

    // Calculate average target sample time and percentage
    double avg_target_sample_time = std::accumulate(target_sample_times.begin(), target_sample_times.end(), 0.0) / target_sample_times.size();
    double target_sample_percentage = (avg_target_sample_time / avg_one_iter_time) * 100.0;
    LOG_INF("Average target sample time: %.2f ms (%.2f%% of one iter time)\n", avg_target_sample_time, target_sample_percentage);

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
