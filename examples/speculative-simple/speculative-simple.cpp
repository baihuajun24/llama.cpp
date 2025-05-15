#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <tuple>
#include <fstream>

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
    // int n_draft_min = params.speculative.n_min;
    int n_draft_min = 1;

    float p_min = params.speculative.p_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // Add a verification list to track (match_n, accept_length, draft_size, draft_time, verify_time)
    std::vector<std::tuple<int, int, int64_t, int64_t, int64_t>> verify_list;

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
    params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft;
    params_spec.p_min   = p_min;

    struct common_speculative * spec = common_speculative_init(ctx_dft);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        const auto t_draft_start = ggml_time_us();
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);
        const auto t_draft_end = ggml_time_us();
        const int64_t draft_time_us = t_draft_end - t_draft_start;

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

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

            llama_decode(ctx_tgt, batch_tgt);
        }

        // Sample from the full target batch and return the accepted tokens based on the target sampler
        const auto t_verify_start = ggml_time_us();
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
        const auto t_verify_end = ggml_time_us();
        const int64_t verify_time_us = t_verify_end - t_verify_start;
        
        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // Record match_n (draft size), accept_length (ids size-1), and verification time
        int match_n = -1; // default value for spec method
        int accept_length = ids.size() - 1;
        verify_list.push_back(std::make_tuple(match_n, accept_length, draft.size(), draft_time_us, verify_time_us));

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

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_kv_self_seq_rm(ctx_tgt, 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();
    const float decoding_speed = n_predict / ((t_dec_end - t_dec_start) / 1e6f);

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, decoding_speed);

    // Calculate accept length average
    float sum = 0;
    for (size_t i = 0; i < verify_list.size(); i++) {
        sum += std::get<1>(verify_list[i]); // Access the accept_length part
    }
    float average = verify_list.empty() ? 0 : sum / verify_list.size();

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
    LOG_INF("0420 Check: accept length average = %.3f\n", average);

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    // Calculate average draft size
    float total_draft_size = 0;
    for (const auto& item : verify_list) {
        total_draft_size += std::get<2>(item);
    }
    float avg_draft_size = verify_list.empty() ? 0 : total_draft_size / verify_list.size();
    LOG_INF("avg_draft_size = %.3f\n", avg_draft_size);

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

    // For storing generated text
    std::stringstream generated_text;
    
    // Check for environment variable first, then fall back to params.out_file
    const char* env_output_path = std::getenv("LLAMA_OUTPUT_FILE");
    std::string output_path = env_output_path != nullptr ? env_output_path : params.out_file;
    bool write_to_file = !output_path.empty();

    // Write to file if requested
    if (write_to_file) {
        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", output_path.c_str());
        } else {
            // Write the statistics as header lines            
            output_file << "# Tokens: " << n_predict << ", Speed: " 
                      << decoding_speed << " t/s, # Forward: " 
                      << verify_list.size() << ", n_draft: "
                      << params_spec.n_draft << "\n";
            output_file << "# Accept length average: " << average << "\n";
            output_file << "# Verify list (match_n, accept_length, draft_size, draft_time_us, verify_time_us): " << verify_list_str << "\n";
            
            // Then write the generated text - only the newly generated tokens, not the original prompt
            // Get the original prompt length
            int prompt_length = inp.size() - n_predict;
            
            // Skip the prompt tokens and only write the generated text
            // Start from the tokens that were generated, not the original prompt
            for (size_t i = prompt_length; i < prompt_tgt.size(); i++) {
                output_file << common_token_to_piece(ctx_tgt, prompt_tgt[i]);
            }
            output_file << common_token_to_piece(ctx_tgt, id_last);
            
            output_file.close();
            
            LOG_INF("Generated text written to: %s\n", output_path.c_str());
        }
    }

    return 0;
}
