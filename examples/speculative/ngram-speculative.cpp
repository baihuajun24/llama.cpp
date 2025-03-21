#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ngram-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <numeric>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

// Function prototypes
/*
void generate_draft_sequences(int n_draft, std::vector<seq_draft>& drafts, llama_context* ctx_dft, llama_batch& batch_dft, 
                             llama_batch& batch_tgt, int& n_past_cur, int& n_drafted, float p_draft_split, int& n_past_tgt);
*/

void generate_draft_sequences_ngram(int n_draft, std::vector<seq_draft>& drafts, llama_context* ctx_tgt, llama_batch& batch_tgt, 
                              int& n_past_cur, int& n_drafted, float /* p_draft_split - unused */, int& n_past_tgt,
                              const common_ngram_cache& ngram_cache);

// Function to generate draft sequences from the n-gram cache
void generate_draft_sequences_ngram(int n_draft, std::vector<seq_draft>& drafts, llama_context* ctx_tgt, llama_batch& batch_tgt, 
                              int& n_past_cur, int& n_drafted, float /* p_draft_split - unused */, int& n_past_tgt,
                              const common_ngram_cache& ngram_cache) {

    // Sample n_draft tokens using the n-gram cache
    for (int i = 0; i < n_draft; ++i) {
        // Track sequences that need to be processed
        std::vector<int> active_sequences;
        for (size_t s = 0; s < drafts.size(); ++s) {
            drafts[s].skip = false; // Reset the skip flag for each draft
            if (drafts[s].drafting && !drafts[s].skip) {
                active_sequences.push_back(static_cast<int>(s));
            }
        }

        if (active_sequences.empty()) {
            break; // No active sequences to process
        }

        for (int s : active_sequences) {
            // For demonstration, use "Secret Service" tokens as dummy draft tokens
            // In a real implementation, you would:
            // 1. Extract the recent tokens from the history (last N tokens)
            // 2. Look up this n-gram in the cache
            // 3. Get the candidate tokens from the cache
            
            // Check if we have any tokens in the draft sequence
            std::vector<llama_token> context;
            if (!drafts[s].tokens.empty()) {
                // Get up to 3 recent tokens for context
                int start_idx = std::max(0, (int)drafts[s].tokens.size() - 3);
                context = std::vector<llama_token>(drafts[s].tokens.begin() + start_idx, drafts[s].tokens.end());
            } else if (n_past_tgt > 0) {
                // If no tokens in drafts yet, get last token from the input sequence
                // In a real implementation, you would get the actual last tokens from the context
                context = {n_past_tgt - 1}; // Just a placeholder
            }
            
            // Create an ngram-cache key from the context
            std::vector<llama_token> ngram_tokens = context;
            // Pad with -1 if needed to meet minimum size
            while (ngram_tokens.size() < LLAMA_NGRAM_MIN) {
                ngram_tokens.insert(ngram_tokens.begin(), -1);
            }
            // Truncate if larger than maximum size
            if (ngram_tokens.size() > LLAMA_NGRAM_MAX) {
                ngram_tokens.erase(ngram_tokens.begin(), ngram_tokens.begin() + (ngram_tokens.size() - LLAMA_NGRAM_MAX));
            }
            
            // Look for the ngram in the cache
            std::vector<llama_token_data> candidates;
            
            // Iterate through the cache to find matching n-grams
            for (const auto& entry : ngram_cache) {
                // Check if the end of the tokens matches the key
                bool matches = true;
                
                // Get tokens from the key
                const auto& key_tokens = entry.first.tokens;
                
                // Count non-padding tokens to determine n-gram size
                size_t key_n = 0;
                for (size_t i = 0; i < LLAMA_NGRAM_MAX; ++i) {
                    if (key_tokens[i] != -1) {
                        key_n++;
                    }
                }
                
                if (key_n <= ngram_tokens.size()) {
                    for (size_t i = 0; i < key_n; ++i) {
                        if (key_tokens[i] != ngram_tokens[ngram_tokens.size() - key_n + i]) {
                            matches = false;
                            break;
                        }
                    }
                    
                    if (matches) {
                        // Found the n-gram, use its candidates
                        const auto& ngram_candidates = entry.second;
                        
                        // Convert ngram candidates to token data format
                        for (const auto& candidate_pair : ngram_candidates) {
                            llama_token_data candidate;
                            candidate.id = candidate_pair.first;
                            // Calculate pseudo-probability based on the count
                            candidate.p = static_cast<float>(candidate_pair.second) / 100.0f; // Normalize
                            candidates.push_back(candidate);
                        }
                        
                        // Sort by probability (descending)
                        std::sort(candidates.begin(), candidates.end(), 
                            [](const llama_token_data& a, const llama_token_data& b) {
                                return a.p > b.p;
                            });
                            
                        break;  // Found a match, no need to continue searching
                    }
                }
            }
            
            // If no candidates found or empty cache, use dummy tokens
            if (candidates.empty()) {
                // Tokenize "Secret Service" as a fallback
                std::string dummy_text = "Secret Service";
                std::vector<llama_token> dummy_tokens = common_tokenize(ctx_tgt, dummy_text, false, false);
                
                // Use the first token as candidate with probability 1.0
                for (size_t j = 0; j < std::min(dummy_tokens.size(), (size_t)8); ++j) {
                    llama_token_data candidate;
                    candidate.id = dummy_tokens[j];
                    candidate.p = 1.0f - (j * 0.1f); // Decreasing probabilities
                    candidates.push_back(candidate);
                }
            }
            
            // Create a token_data_array to mimic the sampler output
            llama_token_data_array candidates_array;
            candidates_array.data = candidates.data();
            candidates_array.size = candidates.size();
            candidates_array.sorted = true;
            
            // // Log the candidates for debugging
            // for (size_t k = 0; k < std::min(candidates.size(), size_t(3)); ++k) {
            //     LOG_INF(" - ngram candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
            //             (int)k, s, i, candidates[k].id, candidates[k].p, 
            //             common_token_to_piece(ctx_tgt, candidates[k].id).c_str());
            // }

            std::vector<int> sa(1, s); // Store the current sequence index

            // Add drafted token for each sequence
            for (size_t is = 0; is < sa.size(); ++is) {
                // Get the candidate token ID
                const llama_token id = is < candidates.size() ? candidates[is].id : candidates[0].id;
                const int seq_idx = sa[is]; // Get the corresponding draft sequence index

                // Store the drafted token in the sequence
                drafts[seq_idx].tokens.push_back(id);
                
                // Store all the candidates as the distribution
                drafts[seq_idx].dists.push_back(candidates);

                // Add unique drafted tokens to the target batch
                drafts[seq_idx].i_batch_tgt.push_back(batch_tgt.n_tokens);
                common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { static_cast<llama_seq_id>(seq_idx) }, true);

                if (batch_tgt.n_tokens > n_draft) {
                    drafts[seq_idx].drafting = false; // Mark the draft as no longer active if it exceeds the limit
                }
            }
        }

        ++n_past_cur; // Increment the count of past tokens processed
        ++n_drafted; // Increment the count of drafted tokens

        if (batch_tgt.n_tokens > n_draft) {
            break; // Exit if the target batch exceeds the draft limit
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    // Initialize n-gram cache
    common_ngram_cache ngram_cache;

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_context * ctx_tgt = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;


    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_INF("vocab_type tgt: %d\n", vocab_type_tgt);


    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    // Log the first 10 tokens of the input
    LOG("\nFirst 10 tokens of the input:\n");
    for (int i = 0; i < std::min(10, (int) inp.size()); ++i) {
        LOG("Token %d: ID=%d, Text='%s'\n", i, inp[i], common_token_to_piece(ctx_tgt, inp[i]).c_str());
    }

    // Fill the n-gram cache with tokens from the prompt
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

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with target model only
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));

    const auto t_enc_end = ggml_time_us();

    // how many tokens to draft each time
    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();  // Keep this for sequence tracking

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);
    
    std::vector<double> all_draft_times;
    std::vector<double> all_target_decode_times;

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    // Global variables for timing sections
    std::vector<double> sampling_times;
    std::vector<double> kv_cache_times;
    std::vector<double> draft_setup_times;
    std::vector<double> draft_gen_times;
    std::vector<double> target_eval_times;

    // Detailed sampling statistics
    std::vector<int> sample_loops_counts;
    std::vector<int> match_attempts_counts;
    std::vector<int> accepted_tokens_counts;
    std::vector<double> token_sampling_times;
    std::vector<double> token_to_string_times;
    std::vector<double> token_matching_times;
    std::vector<double> token_output_times;

    while (true) {
        int s_keep = 0;
        llama_token token_id;
        std::string token_str;

        // Section 1: Sampling and matching
        const auto t_sampling_start = ggml_time_us();
        int n_sample_loops = 0;  // Count total sampling loops
        int n_match_attempts = 0; // Count total match attempts
        int n_accepted_tokens = 0; // Count accepted tokens
        
        // Timing vectors for submodules
        std::vector<double> sample_times;     // Time for token sampling
        std::vector<double> tokenize_times;   // Time for token to string conversion
        std::vector<double> match_times;      // Time for matching against drafts
        std::vector<double> output_times;     // Time for token output/logging

        // Pre-compute logits for all tokens in the batch
        const auto t_logits_start = ggml_time_us();
        llama_decode(ctx_tgt, batch_tgt);
        const auto t_logits_end = ggml_time_us();
        LOG_DBG("Logits computation time: %.3f ms\n", (t_logits_end - t_logits_start) / 1e3);

        // Continue until we have first not match in draft
        for (int i_dft = 0; ; ++i_dft) {
            ++n_sample_loops;
            
            // Time: Token sampling (now just sampling from pre-computed logits)
            const auto t_sample_start = ggml_time_us();
            token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
            common_sampler_accept(smpl, token_id, true);
            const auto t_sample_end = ggml_time_us();
            sample_times.push_back((t_sample_end - t_sample_start) / 1e3);

            // Time: Token to string conversion
            const auto t_tokenize_start = ggml_time_us();
            token_str = common_token_to_piece(ctx_tgt, token_id);
            const auto t_tokenize_end = ggml_time_us();
            tokenize_times.push_back((t_tokenize_end - t_tokenize_start) / 1e3);

            // Time: Match operation
            const auto t_match_start = ggml_time_us();
            bool accept = false;
            for (int s = 0; s < n_seq_dft; ++s) {
                ++n_match_attempts;
                if (!drafts[s].active) continue;
                
                if (i_dft < (int)drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                    s_keep = s;
                    accept = true;
                    ++n_accepted_tokens;
                } else {
                    drafts[s].active = false;
                }
            }
            const auto t_match_end = ggml_time_us();
            match_times.push_back((t_match_end - t_match_start) / 1e3);

            // Update counters and check end conditions
            if (llama_vocab_is_eog(vocab_tgt, token_id)) has_eos = true;
            ++n_predict;

            // Time: Output operation
            const auto t_output_start = ggml_time_us();
            if (params.use_color && accept) {
                char color_code[32];
                snprintf(color_code, sizeof(color_code), "\u001b[%dm", 36 - s_keep % 6);
                LOG("%s%s\u001b[37m", color_code, token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
            const auto t_output_end = ggml_time_us();
            output_times.push_back((t_output_end - t_output_start) / 1e3);

            if (!accept) break;

            // Update counters for accepted token
            ++n_accept;
            ++n_past_tgt;
            ++n_past_dft;
        }

        const auto t_sampling_end = ggml_time_us();
        sampling_times.push_back((t_sampling_end - t_sampling_start) / 1e3);

        // Record statistics for this iteration
        sample_loops_counts.push_back(n_sample_loops);
        match_attempts_counts.push_back(n_match_attempts);
        accepted_tokens_counts.push_back(n_accepted_tokens);
        token_sampling_times.insert(token_sampling_times.end(), sample_times.begin(), sample_times.end());
        token_to_string_times.insert(token_to_string_times.end(), tokenize_times.begin(), tokenize_times.end());
        token_matching_times.insert(token_matching_times.end(), match_times.begin(), match_times.end());
        token_output_times.insert(token_output_times.end(), output_times.begin(), output_times.end());

        LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

        // Section 2: KV Cache updates
        const auto t_kv_start = ggml_time_us();

        llama_kv_self_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
        llama_kv_self_seq_keep(ctx_tgt, s_keep);
        llama_kv_self_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
        llama_kv_self_seq_keep(ctx_tgt, 0);

        const auto t_kv_end = ggml_time_us();
        kv_cache_times.push_back((t_kv_end - t_kv_start) / 1e3);

        // Section 3: Draft setup
        const auto t_draft_setup_start = ggml_time_us();

        // Reset drafts
        for (size_t s = 0; s < static_cast<size_t>(n_seq_dft); ++s) {
            drafts[s].active = false;
            drafts[s].tokens.clear();
            drafts[s].i_batch_tgt.clear();
            drafts[s].dists.clear();
        }

        // Set up next token
        drafts[0].tokens.push_back(token_id);
        drafts[0].dists.push_back(std::vector<llama_token_data>());
        drafts[0].i_batch_tgt.push_back(0);

        ++n_past_dft;

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_past_cur = n_past_dft;

        for (size_t s = 0; s < static_cast<size_t>(n_seq_dft); ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        const auto t_draft_setup_end = ggml_time_us();
        draft_setup_times.push_back((t_draft_setup_end - t_draft_setup_start) / 1e3);

        // Section 4: Draft generation
        const auto t_draft_gen_start = ggml_time_us();
        
        generate_draft_sequences_ngram(n_draft, drafts, ctx_tgt, batch_tgt, 
                                      n_past_cur, n_drafted, p_draft_split, n_past_tgt, 
                                      ngram_cache);
        
        const auto t_draft_gen_end = ggml_time_us();
        draft_gen_times.push_back((t_draft_gen_end - t_draft_gen_start) / 1e3);

        // Section 5: Target model evaluation
        const auto t_target_eval_start = ggml_time_us();

        llama_kv_self_seq_keep(ctx_tgt, 0);
        for (int s = 1; s < n_seq_dft; ++s) {
            llama_kv_self_seq_cp(ctx_tgt, 0, s, -1, -1);
        }

        llama_decode(ctx_tgt, batch_tgt);
        ++n_past_tgt;

        // Remove first token as it's already processed
        for (size_t s = 0; s < static_cast<size_t>(n_seq_dft); ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }

        const auto t_target_eval_end = ggml_time_us();
        target_eval_times.push_back((t_target_eval_end - t_target_eval_start) / 1e3);
    }

    // Print average times for each section
    LOG_INF("\nAverage times per section:\n");
    LOG_INF("Sampling & matching: %.3f ms\n", std::accumulate(sampling_times.begin(), sampling_times.end(), 0.0) / sampling_times.size());
    LOG_INF("KV cache updates: %.3f ms\n", std::accumulate(kv_cache_times.begin(), kv_cache_times.end(), 0.0) / kv_cache_times.size());
    LOG_INF("Draft setup: %.3f ms\n", std::accumulate(draft_setup_times.begin(), draft_setup_times.end(), 0.0) / draft_setup_times.size());
    LOG_INF("Draft generation: %.3f ms\n", std::accumulate(draft_gen_times.begin(), draft_gen_times.end(), 0.0) / draft_gen_times.size());
    LOG_INF("Target evaluation: %.3f ms\n", std::accumulate(target_eval_times.begin(), target_eval_times.end(), 0.0) / target_eval_times.size());

    // Print detailed sampling statistics
    LOG_INF("\nDetailed sampling statistics:\n");
    LOG_INF("Average loops per iteration: %.2f\n", 
            std::accumulate(sample_loops_counts.begin(), sample_loops_counts.end(), 0.0) / sample_loops_counts.size());
    LOG_INF("Average match attempts per iteration: %.2f\n", 
            std::accumulate(match_attempts_counts.begin(), match_attempts_counts.end(), 0.0) / match_attempts_counts.size());
    LOG_INF("Average accepted tokens per iteration: %.2f\n", 
            std::accumulate(accepted_tokens_counts.begin(), accepted_tokens_counts.end(), 0.0) / accepted_tokens_counts.size());
    LOG_INF("Average token sampling time: %.3f ms\n", 
            std::accumulate(token_sampling_times.begin(), token_sampling_times.end(), 0.0) / token_sampling_times.size());
    LOG_INF("Average token to string time: %.3f ms\n", 
            std::accumulate(token_to_string_times.begin(), token_to_string_times.end(), 0.0) / token_to_string_times.size());
    LOG_INF("Average token matching time: %.3f ms\n", 
            std::accumulate(token_matching_times.begin(), token_matching_times.end(), 0.0) / token_matching_times.size());
    LOG_INF("Average token output time: %.3f ms\n", 
            std::accumulate(token_output_times.begin(), token_output_times.end(), 0.0) / token_output_times.size());

    // Calculate average total time per iteration
    double avg_total_ms = 
        std::accumulate(sampling_times.begin(), sampling_times.end(), 0.0) / sampling_times.size() +
        std::accumulate(kv_cache_times.begin(), kv_cache_times.end(), 0.0) / kv_cache_times.size() +
        std::accumulate(draft_setup_times.begin(), draft_setup_times.end(), 0.0) / draft_setup_times.size() +
        std::accumulate(draft_gen_times.begin(), draft_gen_times.end(), 0.0) / draft_gen_times.size() +
        std::accumulate(target_eval_times.begin(), target_eval_times.end(), 0.0) / target_eval_times.size();

    LOG_INF("\nTotal time per iteration: %.3f ms\n", avg_total_ms);
    LOG_INF("Estimated tokens/second: %.3f tok/s\n", 1000.0 / avg_total_ms);

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");
    LOG_INF("Average n-gram cache sampling time: %.3f ms\n", std::accumulate(all_draft_times.begin(), all_draft_times.end(), 0.0) / all_draft_times.size());
    LOG_INF("Average target model forward time: %.3f ms\n", std::accumulate(all_target_decode_times.begin(), all_target_decode_times.end(), 0.0) / all_target_decode_times.size());
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (size_t s = 0; s < static_cast<size_t>(n_seq_dft); ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}

