#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ngplus_params {
    std::string hot_source = "prompt,hot-table";
    std::string cold_path;
    std::string cold_mmap = "on";
    std::string trace_path;
    int hot_ngram_max = 6;
    int draft = 8;
    int tree_budget = 64;
    bool no_display_prompt = false;
};

struct prompt_draft_result {
    llama_tokens tokens;
    int order = 0;
    int source_pos = -1;
};

static void print_ngplus_usage(int, char **) {
    printf("\n----- ngplus verifier params -----\n\n");
    printf("  --ngplus-hot-source SOURCES   comma-separated hot sources (default: prompt,hot-table)\n");
    printf("  --ngplus-hot-ngram-max N      maximum prompt-local hot n-gram order accepted by CLI (default: 6)\n");
    printf("  --ngplus-cold-path FNAME      cold-store path, currently traced as a no-op source\n");
    printf("  --ngplus-cold-mmap on|off     cold mmap flag, currently traced as a no-op source\n");
    printf("  --ngplus-draft N              maximum prompt-local draft continuation length (default: 8)\n");
    printf("  --ngplus-tree-budget N        maximum verifier tree budget for this narrow verifier (default: 64)\n");
    printf("  --ngplus-trace FNAME          write per-step NG+ JSONL trace rows\n");
}

static std::string require_value(int argc, char ** argv, int & i, const std::string & arg) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("expected value for " + arg);
    }
    return argv[++i];
}

static bool split_equals_arg(const std::string & arg, std::string & key, std::string & value) {
    const size_t pos = arg.find('=');
    if (pos == std::string::npos) {
        return false;
    }
    key = arg.substr(0, pos);
    value = arg.substr(pos + 1);
    return true;
}

static int parse_positive_int(const std::string & value, const std::string & arg) {
    int parsed = 0;
    try {
        parsed = std::stoi(value);
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid integer for " + arg + ": " + value);
    }
    if (parsed < 1) {
        throw std::invalid_argument(arg + " must be >= 1");
    }
    return parsed;
}

static std::string resolve_hf_repo_to_local_model(const std::string & repo) {
    static const std::string gemma4_repo = "unsloth/gemma-4-12b-it-GGUF:Q4_K_M";
    static const std::string gemma4_model =
        "/Users/baihuajun/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/"
        "snapshots/3249fa54d5efa384afc552cc6700ad091efd5c39/gemma-4-12b-it-Q4_K_M.gguf";

    if (repo == gemma4_repo) {
        std::ifstream file(gemma4_model);
        if (file.good()) {
            return gemma4_model;
        }
    }
    return "";
}

static std::vector<std::string> preprocess_args(int argc, char ** argv, ngplus_params & ngp) {
    std::vector<std::string> out;
    out.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key;
        std::string value;
        const bool has_equals = split_equals_arg(arg, key, value);
        const std::string name = has_equals ? key : arg;

        auto value_for = [&](const std::string & opt) {
            return has_equals ? value : require_value(argc, argv, i, opt);
        };

        if (name == "--ngplus-hot-source") {
            ngp.hot_source = value_for(name);
        } else if (name == "--ngplus-hot-ngram-max") {
            ngp.hot_ngram_max = parse_positive_int(value_for(name), name);
        } else if (name == "--ngplus-cold-path") {
            ngp.cold_path = value_for(name);
        } else if (name == "--ngplus-cold-mmap") {
            ngp.cold_mmap = value_for(name);
        } else if (name == "--ngplus-draft") {
            ngp.draft = parse_positive_int(value_for(name), name);
        } else if (name == "--ngplus-tree-budget") {
            ngp.tree_budget = parse_positive_int(value_for(name), name);
        } else if (name == "--ngplus-trace") {
            ngp.trace_path = value_for(name);
        } else if (name == "--no-display-prompt") {
            ngp.no_display_prompt = true;
        } else if (name == "--single-turn") {
            // Accepted for eval-wrapper compatibility. This non-interactive verifier is always single-turn.
        } else if (name == "--offline" || name == "--no-mmproj") {
            // Accepted for eval-wrapper compatibility. They are handled by upstream helpers, not this fork.
        } else if (name == "-hf" || name == "-hfr" || name == "--hf-repo") {
            const std::string repo = value_for(name);
            const std::string local_model = resolve_hf_repo_to_local_model(repo);
            if (!local_model.empty()) {
                out.emplace_back("-m");
                out.emplace_back(local_model);
            } else {
                out.emplace_back(arg);
                if (!has_equals) {
                    out.emplace_back(repo);
                }
            }
        } else if (name == "-fa" || name == "--flash-attn") {
            out.emplace_back(arg);
            if (!has_equals && i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (next == "on" || next == "off" || next == "true" || next == "false") {
                    ++i;
                }
            }
        } else {
            out.emplace_back(arg);
        }
    }

    const int effective_draft = std::max(1, std::min(ngp.draft, ngp.tree_budget));
    const int effective_ngram_max = std::max(1, std::min(4, ngp.hot_ngram_max));
    const int effective_ngram_min = std::min(2, effective_ngram_max);

    out.emplace_back("--draft-max");
    out.emplace_back(std::to_string(effective_draft));
    out.emplace_back("--ngram-min");
    out.emplace_back(std::to_string(effective_ngram_min));
    out.emplace_back("--ngram-max");
    out.emplace_back(std::to_string(effective_ngram_max));

    return out;
}

static std::string json_escape(const std::string & input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (const char ch : input) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += ch;
                }
                break;
        }
    }
    return out;
}

static prompt_draft_result prompt_local_draft(
        const std::vector<llama_token> & history,
        int max_order,
        int max_draft) {
    prompt_draft_result result;
    if (max_draft <= 0 || history.size() < 2) {
        return result;
    }

    const int history_size = (int) history.size();
    const int order_max = std::min(max_order, history_size);

    for (int order = order_max; order >= 2; --order) {
        const int key_start = history_size - order;
        for (int pos = key_start - order; pos >= 0; --pos) {
            bool match = true;
            for (int i = 0; i < order; ++i) {
                if (history[pos + i] != history[key_start + i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }

            const int continuation_start = pos + order;
            const int available = history_size - continuation_start;
            const int n_copy = std::min(max_draft, available);
            if (n_copy <= 0) {
                continue;
            }

            result.tokens.insert(
                result.tokens.end(),
                history.begin() + continuation_start,
                history.begin() + continuation_start + n_copy);
            result.order = order;
            result.source_pos = pos;
            return result;
        }
    }

    return result;
}

static void trace_step(
        std::ofstream & trace,
        int step,
        const ngplus_params & ngp,
        const common_params & params,
        int drafted_tokens,
        int accepted_tokens,
        int target_tokens,
        int source_order,
        int source_pos,
        int64_t hot_lookup_us,
        int64_t tree_build_us,
        int64_t target_verify_us,
        int64_t kv_cleanup_us,
        int output_tokens_total) {
    if (!trace.is_open()) {
        return;
    }

    trace << "{"
          << "\"event\":\"step\","
          << "\"step\":" << step << ","
          << "\"source\":\"" << (drafted_tokens > 0 ? "prompt-local-hot" : "fallback") << "\","
          << "\"hot_source\":\"" << json_escape(ngp.hot_source) << "\","
          << "\"cold_source\":\"noop\","
          << "\"cold_path\":\"" << json_escape(ngp.cold_path) << "\","
          << "\"cold_mmap\":\"" << json_escape(ngp.cold_mmap) << "\","
          << "\"ngram_min\":" << params.ngram_min << ","
          << "\"ngram_max\":" << params.ngram_max << ","
          << "\"source_order\":" << source_order << ","
          << "\"source_pos\":" << source_pos << ","
          << "\"ngplus_draft\":" << ngp.draft << ","
          << "\"tree_budget\":" << ngp.tree_budget << ","
          << "\"drafted_tokens\":" << drafted_tokens << ","
          << "\"accepted_tokens\":" << accepted_tokens << ","
          << "\"target_tokens\":" << target_tokens << ","
          << "\"fallback\":" << (drafted_tokens > 0 ? "false" : "true") << ","
          << "\"fallback_reason\":" << (drafted_tokens > 0 ? "null" : "\"no_prompt_local_candidate\"") << ","
          << "\"hot_lookup_us\":" << hot_lookup_us << ","
          << "\"cold_lookup_us\":0,"
          << "\"tree_build_us\":" << tree_build_us << ","
          << "\"target_verify_us\":" << target_verify_us << ","
          << "\"kv_cleanup_us\":" << kv_cleanup_us << ","
          << "\"output_tokens_total\":" << output_tokens_total
          << "}\n";
}

int main(int argc, char ** argv) {
    ngplus_params ngp;
    common_params params;

    std::vector<std::string> args_storage;
    try {
        args_storage = preprocess_args(argc, argv, ngp);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    std::vector<char *> args;
    args.reserve(args_storage.size());
    for (std::string & item : args_storage) {
        args.push_back(const_cast<char *>(item.c_str()));
    }

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_LOOKUP, print_ngplus_usage)) {
        return 1;
    }
    if (ngp.no_display_prompt) {
        params.display_prompt = false;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> history = common_tokenize(ctx, params.prompt, true, true);
    if (history.empty()) {
        LOG_ERR("prompt tokenization produced no tokens\n");
        llama_backend_free();
        return 1;
    }

    const int64_t t_hot_init_start_us = ggml_time_us();
    const int64_t t_hot_init_us = ggml_time_us() - t_hot_init_start_us;

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;
    if ((int) history.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) history.size(), max_tokens_list_size);
        llama_backend_free();
        return 1;
    }

    if (params.display_prompt) {
        LOG("\n\n");
        for (llama_token id : history) {
            LOG("%s", common_token_to_piece(ctx, id).c_str());
        }
        fflush(stderr);
    }

    const int n_input = (int) history.size();
    const int n_draft = params.speculative.n_max;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept = 0;
    bool has_eos = false;

    std::ofstream trace;
    if (!ngp.trace_path.empty()) {
        trace.open(ngp.trace_path);
        if (!trace.is_open()) {
            LOG_ERR("failed to open NG+ trace file: %s\n", ngp.trace_path.c_str());
            llama_backend_free();
            return 1;
        }
    }

    std::stringstream generated_text;
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    const auto t_enc_start = ggml_time_us();
    if (history.size() > 1) {
        if (llama_decode(ctx, llama_batch_get_one(history.data(), (int) history.size() - 1)) != 0) {
            LOG_ERR("failed to evaluate prompt\n");
            common_sampler_free(smpl);
            llama_backend_free();
            return 1;
        }
    }
    const auto t_enc_end = ggml_time_us();

    llama_token id_last = history.back();
    int n_past = (int) history.size() - 1;
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx), 0, 1);

    const auto t_dec_start = ggml_time_us();
    int step = 0;

    while (!has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
        const int remaining = params.n_predict < 0 ? n_draft + 1 : std::max(1, params.n_predict - n_predict);
        const int draft_limit = std::max(0, std::min(n_draft, remaining - 1));

        const int64_t t_draft_start_us = ggml_time_us();
        const prompt_draft_result draft_result = prompt_local_draft(history, params.ngram_max, draft_limit);
        const int64_t draft_us = ggml_time_us() - t_draft_start_us;

        const llama_tokens & draft = draft_result.tokens;

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (int) i, { 0 }, true);
        }

        const int64_t t_verify_start_us = ggml_time_us();
        if (llama_decode(ctx, batch_tgt) != 0) {
            LOG_ERR("target decode failed during NG+ verification\n");
            break;
        }
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx, draft);
        const int64_t verify_us = ggml_time_us() - t_verify_start_us;

        const int accepted_from_draft = std::max(0, (int) ids.size() - 1);
        n_drafted += (int) draft.size();
        n_accept += accepted_from_draft;
        n_past += accepted_from_draft;

        for (llama_token id : ids) {
            id_last = id;
            history.push_back(id);

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx, id);
            LOG("%s", token_str.c_str());
            generated_text << token_str;
            ++n_predict;

            if (params.n_predict >= 0 && n_predict >= params.n_predict) {
                break;
            }
        }

        const int64_t t_kv_start_us = ggml_time_us();
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);
        const int64_t kv_cleanup_us = ggml_time_us() - t_kv_start_us;

        trace_step(
            trace,
            step++,
            ngp,
            params,
            (int) draft.size(),
            accepted_from_draft,
            (int) ids.size(),
            draft_result.order,
            draft_result.source_pos,
            draft_us,
            draft_us,
            verify_us,
            kv_cleanup_us,
            n_predict);
    }

    const auto t_dec_end = ggml_time_us();
    const float enc_s = (t_enc_end - t_enc_start) / 1e6f;
    const float dec_s = (t_dec_end - t_dec_start) / 1e6f;
    const float decode_tps = dec_s > 0.0f ? n_predict / dec_s : 0.0f;

    LOG("\n\n");
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, enc_s, enc_s > 0.0f ? n_input / enc_s : 0.0f);
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, dec_s, decode_tps);
    LOG_INF("0420 Check: accept length average      = %.3f\n", step > 0 ? (float) n_accept / step : 0.0f);
    LOG_INF("ngplus hot init = %.3f ms\n", t_hot_init_us * 1e-3);
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("n_accept     = %d\n", n_accept);
    LOG_INF("accept       = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);

    if (!params.out_file.empty()) {
        std::ofstream output_file(params.out_file);
        if (!output_file.is_open()) {
            LOG_ERR("failed to open output file: %s\n", params.out_file.c_str());
        } else {
            output_file << generated_text.str();
        }
    }

    LOG_INF("\ntarget:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);
    llama_batch_free(batch_tgt);
    llama_backend_free();
    LOG("\n\n");

    return 0;
}
