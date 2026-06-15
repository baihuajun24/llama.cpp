#include "arg.h"
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
#include "chat.h"
#endif
#include "common.h"
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
#include "ggml-backend.h"
#endif
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ngplus_params {
    std::string hot_source = "prompt,hot-table";
    std::string cold_path;
    std::string cold_mmap = "on";
    std::string trace_path;
    std::string out_file;
    int hot_ngram_max = 6;
    int draft = 8;
    int tree_budget = 64;
    int effective_ngram_min = 2;
    int effective_ngram_max = 4;
    int effective_draft = 8;
    bool cpu_fallback = false;
    bool single_turn = true;
    bool chat_template_applied = false;
    bool prompt_sampler_seeded = false;
    bool backend_sampling_requested = false;
    bool server_backend_sampling = false;
    bool draft_acceptance_enabled = false;
    bool no_display_prompt = false;
    int prompt_tokens = 0;
    int prompt_bytes = 0;
    std::string prompt_fingerprint;
    std::string chat_generation_prompt;
    std::string sampler_chain;
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

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
static bool is_inspection_arg(const std::string & name) {
    return name == "-h" || name == "--help" || name == "--usage" || name == "--list-devices";
}

static bool is_gpu_layers_arg(const std::string & name) {
    return name == "-ngl" || name == "--gpu-layers" || name == "--n-gpu-layers";
}

static bool is_device_arg(const std::string & name) {
    return name == "-dev" || name == "--device";
}

static bool is_flash_attn_arg(const std::string & name) {
    return name == "-fa" || name == "--flash-attn";
}

static bool gpu_layers_request_uses_offload(const std::string & value) {
    return value != "0" && value != "none" && value != "off";
}

static bool has_usable_offload_device() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                type != GGML_BACKEND_DEVICE_TYPE_META) {
            continue;
        }

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (total > 0) {
            return true;
        }
    }

    return false;
}

static bool should_force_cpu_fallback(int argc, char ** argv) {
    bool inspection_only = false;
    bool requested_gpu_offload = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key;
        std::string value;
        const bool has_equals = split_equals_arg(arg, key, value);
        const std::string name = has_equals ? key : arg;

        if (is_inspection_arg(name)) {
            inspection_only = true;
        }

        if (is_gpu_layers_arg(name)) {
            const std::string gpu_layers = has_equals ? value : (i + 1 < argc ? argv[i + 1] : "");
            requested_gpu_offload = gpu_layers_request_uses_offload(gpu_layers);
        }
    }

    return requested_gpu_offload && !inspection_only && !has_usable_offload_device();
}
#endif

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

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    const bool force_cpu = should_force_cpu_fallback(argc, argv);
    bool wrote_cpu_device = false;
    bool wrote_flash_attn = false;
    ngp.cpu_fallback = force_cpu;
#else
    const bool force_cpu = false;
#endif

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
        } else if (name == "-o" || name == "--output" || name == "--output-file") {
            ngp.out_file = value_for(name);
        } else if (name == "--no-display-prompt") {
            ngp.no_display_prompt = true;
        } else if (name == "--single-turn") {
            ngp.single_turn = true;
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
        } else if (is_flash_attn_arg(name)) {
            out.emplace_back(force_cpu ? "-fa" : arg);
            if (!has_equals && i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (next == "on" || next == "off" || next == "auto" || next == "true" || next == "false") {
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
                    out.emplace_back(force_cpu ? "off" : next);
                    wrote_flash_attn = true;
#endif
                    ++i;
                }
            }
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
            if (force_cpu && has_equals) {
                out.emplace_back("off");
                wrote_flash_attn = true;
            }
#endif
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
        } else if (is_gpu_layers_arg(name) && force_cpu) {
            if (!has_equals) {
                require_value(argc, argv, i, name);
            }
            out.emplace_back("-ngl");
            out.emplace_back("0");
        } else if (is_device_arg(name) && force_cpu) {
            if (!has_equals) {
                require_value(argc, argv, i, name);
            }
            out.emplace_back("-dev");
            out.emplace_back("none");
            wrote_cpu_device = true;
#endif
        } else {
            out.emplace_back(arg);
        }
    }

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    if (force_cpu && !wrote_cpu_device) {
        out.emplace_back("-dev");
        out.emplace_back("none");
    }
    if (force_cpu && !wrote_flash_attn) {
        out.emplace_back("-fa");
        out.emplace_back("off");
    }
#endif

    ngp.effective_draft = std::max(1, std::min(ngp.draft, ngp.tree_budget));
    ngp.effective_ngram_max = std::max(1, std::min(4, ngp.hot_ngram_max));
    ngp.effective_ngram_min = std::min(2, ngp.effective_ngram_max);

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

static std::string fnv1a64_hex(const std::string & input) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : input) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }

    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

static std::string json_float_or_null(float value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << value;
    return out.str();
}

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
static std::string top_candidates_json(
        llama_context * ctx,
        common_sampler * smpl,
        int max_items,
        llama_token selected) {
    llama_token_data_array * candidates = common_sampler_get_candidates(smpl, true);
    if (candidates == nullptr || candidates->size == 0 || max_items <= 0) {
        return "[]";
    }

    std::ostringstream out;
    out << "[";
    const int n_items = std::min(max_items, (int) candidates->size);
    for (int i = 0; i < n_items; ++i) {
        const llama_token_data & candidate = candidates->data[i];
        if (i > 0) {
            out << ",";
        }
        out << "{"
            << "\"id\":" << candidate.id << ","
            << "\"logit\":" << json_float_or_null(candidate.logit) << ","
            << "\"p\":" << json_float_or_null(candidate.p) << ","
            << "\"selected\":" << (candidate.id == selected ? "true" : "false") << ","
            << "\"piece\":\"" << json_escape(common_token_to_piece(ctx, candidate.id)) << "\""
            << "}";
    }
    out << "]";
    return out.str();
}
#endif

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
static bool apply_single_turn_chat_template(common_params & params, llama_model * model) {
    if (model == nullptr || params.prompt.empty()) {
        return false;
    }

    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, params.chat_template);
    auto caps = common_chat_templates_get_caps(chat_templates.get());

    common_chat_templates_inputs inputs;
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = params.prompt;
    inputs.messages.push_back(std::move(user_msg));
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.use_jinja = params.use_jinja;
    inputs.parallel_tool_calls = caps["supports_parallel_tool_calls"];
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = params.reasoning_format;
    inputs.enable_thinking = false;
    inputs.chat_template_kwargs = params.default_template_kwargs;
    inputs.chat_template_kwargs["enable_thinking"] = "false";
    inputs.force_pure_content = params.force_pure_content_parser;

    const common_chat_params chat_params = common_chat_templates_apply(chat_templates.get(), inputs);
    params.prompt = chat_params.prompt;
    params.sampling.generation_prompt = chat_params.generation_prompt;

    return true;
}
#endif

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
        int drafted_tokens,
        int accepted_tokens,
        int target_tokens,
        int source_order,
        int source_pos,
        int64_t hot_lookup_us,
        int64_t tree_build_us,
        int64_t target_verify_us,
        int64_t kv_cleanup_us,
        int output_tokens_total,
        llama_token sampled_token,
        const std::string & sampled_piece,
        const std::string & top_candidates) {
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
          << "\"prompt_format\":\"" << (ngp.chat_template_applied ? "chat-single-turn" : "raw") << "\","
          << "\"prompt_tokens\":" << ngp.prompt_tokens << ","
          << "\"prompt_bytes\":" << ngp.prompt_bytes << ","
          << "\"prompt_fingerprint\":\"" << ngp.prompt_fingerprint << "\","
          << "\"chat_generation_prompt\":\"" << json_escape(ngp.chat_generation_prompt) << "\","
          << "\"prompt_sampler_seeded\":" << (ngp.prompt_sampler_seeded ? "true" : "false") << ","
          << "\"backend_sampling_requested\":" << (ngp.backend_sampling_requested ? "true" : "false") << ","
          << "\"server_backend_sampling\":" << (ngp.server_backend_sampling ? "true" : "false") << ","
          << "\"sampler_chain\":\"" << json_escape(ngp.sampler_chain) << "\","
          << "\"verification_mode\":\"ar_exact_prefill_diagnostic_draft\","
          << "\"draft_acceptance_enabled\":" << (ngp.draft_acceptance_enabled ? "true" : "false") << ","
          << "\"device_fallback\":" << (ngp.cpu_fallback ? "\"cpu_no_usable_offload_device\"" : "null") << ","
          << "\"ngram_min\":" << ngp.effective_ngram_min << ","
          << "\"ngram_max\":" << ngp.effective_ngram_max << ","
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
          << "\"output_tokens_total\":" << output_tokens_total << ","
          << "\"sampled_token\":" << sampled_token << ","
          << "\"sampled_piece\":\"" << json_escape(sampled_piece) << "\","
          << "\"top_candidates\":" << top_candidates
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
    if (!ngp.out_file.empty()) {
        params.out_file = ngp.out_file;
    }
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    if (ngp.cpu_fallback) {
        params.devices.assign(1, nullptr);
        params.n_gpu_layers = 0;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        LOG_WRN("ngplus: no usable GPU/Metal offload device visible; using CPU fallback (-ngl 0, -dev none, -fa off)\n");
    }
#endif

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
#else
    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();
#endif
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to initialize Gemma4 target model/context for NG+ verification\n");
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    if (ngp.single_turn) {
        try {
            params.enable_reasoning = 0;
            params.default_template_kwargs["enable_thinking"] = "false";
            ngp.chat_template_applied = apply_single_turn_chat_template(params, model);
        } catch (const std::exception & e) {
            LOG_ERR("failed to apply single-turn chat template for NG+ verification: %s\n", e.what());
            llama_backend_free();
            return 1;
        }
        if (ngp.chat_template_applied) {
            LOG_INF("ngplus: applied single-turn chat template with reasoning disabled\n");
        }
    }
#endif

    std::vector<llama_token> history = common_tokenize(ctx, params.prompt, true, true);
    if (history.empty()) {
        LOG_ERR("prompt tokenization produced no tokens\n");
        llama_backend_free();
        return 1;
    }
    ngp.prompt_tokens = (int) history.size();
    ngp.prompt_bytes = (int) params.prompt.size();
    ngp.prompt_fingerprint = fnv1a64_hex(params.prompt);
    ngp.chat_generation_prompt = params.sampling.generation_prompt;

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
    const int n_draft = ngp.effective_draft;
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
    common_sampler_reset(smpl);
    ngp.backend_sampling_requested = params.sampling.backend_sampling;
    ngp.sampler_chain = common_sampler_print(smpl);

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    // The fixed server baseline only attaches a backend sampler when -bs /
    // --backend-sampling is requested. Respect that flag so the verifier does
    // not silently test a different sampling surface from the server command.
    if (params.sampling.backend_sampling) {
        ngp.server_backend_sampling = llama_set_sampler(ctx, 0, common_sampler_get(smpl));
        if (!ngp.server_backend_sampling) {
            LOG_WRN("ngplus: failed to attach requested backend sampler; falling back to host sampling\n");
        }
    }
#endif

    const int batch_capacity = std::max(
        (int) llama_n_batch(ctx),
        std::max((int) history.size(), ngp.effective_draft + 1));
    llama_batch batch_tgt = llama_batch_init(batch_capacity, 0, 1);

    const auto t_enc_start = ggml_time_us();
    common_batch_clear(batch_tgt);
    for (int i = 0; i < (int) history.size(); ++i) {
        common_batch_add(batch_tgt, history[i], i, { 0 }, i == (int) history.size() - 1);
    }
    if (llama_decode(ctx, batch_tgt) != 0) {
        LOG_ERR("failed to evaluate full prompt for server-aligned NG+ verification\n");
        common_sampler_free(smpl);
        llama_batch_free(batch_tgt);
        llama_backend_free();
        return 1;
    }
    const auto t_enc_end = ggml_time_us();

    common_sampler_reset(smpl);
    for (llama_token id : history) {
        common_sampler_accept(smpl, id, false);
    }
    ngp.prompt_sampler_seeded = true;

    int n_past = (int) history.size();

    const auto t_dec_start = ggml_time_us();
    int step = 0;

    while (!has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
        const int remaining = params.n_predict < 0 ? n_draft : std::max(1, params.n_predict - n_predict);
        const int draft_limit = std::max(0, std::min(n_draft, remaining));

        const int64_t t_draft_start_us = ggml_time_us();
        const prompt_draft_result draft_result = prompt_local_draft(history, ngp.effective_ngram_max, draft_limit);
        const int64_t draft_us = ggml_time_us() - t_draft_start_us;

        const llama_tokens & draft = draft_result.tokens;

        const int64_t t_verify_start_us = ggml_time_us();
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
        const std::string top_candidates = top_candidates_json(ctx, smpl, 5, id);
#else
        const std::string top_candidates = "[]";
#endif
        common_sampler_accept(smpl, id, true);

        const int accepted_from_draft = 0;
        n_drafted += (int) draft.size();
        n_accept += accepted_from_draft;

        history.push_back(id);

        if (llama_vocab_is_eog(vocab, id)) {
            has_eos = true;
        } else {
            const std::string token_str = common_token_to_piece(ctx, id);
            LOG("%s", token_str.c_str());
            generated_text << token_str;
            ++n_predict;
        }

        const bool need_next_logits =
            !has_eos && (params.n_predict < 0 || n_predict < params.n_predict);
        if (need_next_logits) {
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id, n_past, { 0 }, true);
            if (llama_decode(ctx, batch_tgt) != 0) {
                LOG_ERR("target decode failed during AR-exact NG+ verification\n");
                break;
            }
            ++n_past;
        }
        const int64_t verify_us = ggml_time_us() - t_verify_start_us;

        const int64_t t_kv_start_us = ggml_time_us();
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
#else
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);
#endif
        const int64_t kv_cleanup_us = ggml_time_us() - t_kv_start_us;

        trace_step(
            trace,
            step++,
            ngp,
            (int) draft.size(),
            accepted_from_draft,
            1,
            draft_result.order,
            draft_result.source_pos,
            draft_us,
            draft_us,
            verify_us,
            kv_cleanup_us,
            n_predict,
            id,
            llama_vocab_is_eog(vocab, id) ? std::string() : common_token_to_piece(ctx, id),
            top_candidates);
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
