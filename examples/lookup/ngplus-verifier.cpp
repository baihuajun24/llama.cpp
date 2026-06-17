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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ngplus_params {
    std::string hot_source = "prompt,hot-table";
    std::string hot_table_path;
    std::string cold_path;
    std::string cold_mmap = "on";
    std::string trace_path;
    std::string out_file;
    std::string reference_token_ids_arg;
    std::string reference_text_arg;
    std::string reference_json_arg;
    std::string reference_prompt_text_arg;
    std::string reference_prompt_json_arg;
    std::string dump_prompt_token_ids_path;
    std::vector<llama_token> reference_token_ids;
    bool dump_no_bos = false;
    bool dump_no_parse_special = false;
    bool stop_after_reference_mismatch = false;
    bool force_reference_tokens = false;
    int reference_text_bytes = 0;
    std::string reference_text_fnv1a64;
    std::string reference_source = "none";
    int reference_prompt_text_bytes = 0;
    std::string reference_prompt_text_fnv1a64;
    std::string reference_prompt_source = "none";
    int reference_prompt_first_mismatch_byte = -1;
    bool reference_prompt_matches = false;
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
    int prompt_local_drafted_tokens = 0;
    int recent_generation_drafted_tokens = 0;
    int static_hot_table_drafted_tokens = 0;
    int static_hot_table_hits = 0;
    int static_hot_table_misses = 0;
    int64_t static_hot_table_lookup_us_total = 0;
    int fallback_steps = 0;
    bool static_hot_table_loaded = false;
    bool static_hot_table_candidate_enabled = false;
    bool recent_generation_enabled = false;       // Phase 5: self-referential generation cache
    int recent_generation_min_order = 4;          // Phase 5: min suffix order for self-ref drafts
    bool batched_verify_enabled = false;          // Phase 5: correct batched spec verify (vs blind)
    int batched_verify_max_k = 0;                  // Phase 5: cap verified batch length (0 = no cap)
    bool hot_table_chain_enabled = false;         // Phase 5: chained multi-token code-store drafts
    bool structure_indent_enabled = false;        // Phase 5: model-free indentation drafter
    int prompt_local_pos_cap = 80;                // Phase 5: prompt-local echo source-pos cap (delta)
    bool hybrid_verify_enabled = false;           // Phase 5: blind for prompt echoes, correct for novel
    int static_hot_table_candidate_min_count = 2;
    int static_hot_table_candidate_min_top_share_pct = 50;
    int static_hot_table_order = 0;
    int static_hot_table_rows = 0;
    int64_t static_hot_table_bytes = 0;
    int64_t static_hot_table_load_us = 0;
    std::string prompt_fingerprint;
    std::string prompt_token_head_json = "[]";
    std::string prompt_token_tail_json = "[]";
    std::string prompt_token_head_pieces_json = "[]";
    std::string prompt_token_tail_pieces_json = "[]";
    std::string prompt_text;
    std::string prompt_suffix;
    std::string chat_generation_prompt;
    bool chat_grammar_lazy = false;
    int reasoning_budget_start_tokens = 0;
    int reasoning_budget_end_tokens = 0;
    int reasoning_budget_forced_tokens = 0;
    std::string sampler_chain;
};

struct prompt_draft_result {
    llama_tokens tokens;
    int order = 0;
    int source_pos = -1;
    int continuation_start = -1;
    int continuation_available = 0;
    int continuation_copied = 0;
    bool truncated_by_draft_limit = false;
    std::string source_label;
};

struct hot_table_entry {
    llama_tokens context;
    std::vector<std::pair<llama_token, int>> next;
    int total_count = 0;
    int rank = -1;
};

struct hot_table_lookup_result {
    bool hit = false;
    int order = 0;
    int total_count = 0;
    int rank = -1;
    llama_token top_token = -1;
    int top_count = 0;
    int candidate_count = 0;
    int64_t lookup_us = 0;
};

struct static_hot_table {
    int order = 0;
    int64_t bytes = 0;
    std::unordered_map<std::string, hot_table_entry> entries;
};

static std::string token_key(const llama_tokens & tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << tokens[i];
    }
    return out.str();
}

static std::string token_key_suffix(const std::vector<llama_token> & tokens, int order) {
    if (order <= 0 || (int) tokens.size() < order) {
        return "";
    }
    llama_tokens suffix;
    suffix.reserve(order);
    suffix.insert(suffix.end(), tokens.end() - order, tokens.end());
    return token_key(suffix);
}

static std::string draft_source_label(const prompt_draft_result & draft_result, int prompt_tokens) {
    if (draft_result.tokens.empty()) {
        return "fallback";
    }
    if (!draft_result.source_label.empty()) {
        return draft_result.source_label;
    }
    if (draft_result.source_pos >= 0 && draft_result.source_pos + draft_result.order <= prompt_tokens) {
        return "prompt-local-hot";
    }
    return "recent-generation-hot";
}

static void print_ngplus_usage(int, char **) {
    printf("\n----- ngplus verifier params -----\n\n");
    printf("  --ngplus-hot-source SOURCES   comma-separated hot sources (default: prompt,hot-table)\n");
    printf("                                include hot-table-candidates to enable guarded static table drafts\n");
    printf("  --ngplus-hot-table-path FNAME load a Phase 4 static hot-table JSONL source for trace accounting\n");
    printf("  --ngplus-hot-table-candidates on|off\n");
    printf("                                allow high-confidence static hot-table one-token drafts (default: off)\n");
    printf("  --ngplus-hot-table-min-count N\n");
    printf("                                minimum top-token count for static hot-table drafts (default: 2)\n");
    printf("  --ngplus-hot-table-min-top-share-pct N\n");
    printf("                                minimum top-token share percentage for static hot-table drafts (default: 50)\n");
    printf("  --ngplus-hot-ngram-max N      maximum prompt-local hot n-gram order accepted by CLI (default: 6)\n");
    printf("  --ngplus-cold-path FNAME      cold-store path; .jsonl paths are also accepted as Phase 4 hot-table fixtures\n");
    printf("  --ngplus-cold-mmap on|off     cold mmap flag, currently traced as a no-op source\n");
    printf("  --ngplus-draft N              maximum prompt-local draft continuation length (default: 8)\n");
    printf("  --ngplus-tree-budget N        maximum verifier tree budget for this narrow verifier (default: 64)\n");
    printf("  --ngplus-trace FNAME          write per-step NG+ JSONL trace rows\n");
    printf("  --ngplus-reference-token-ids IDS|@FILE\n");
    printf("                                optional comma/space-separated reference token IDs for trace exactness diagnostics\n");
    printf("  --ngplus-reference-text TEXT|@FILE\n");
    printf("                                optional generated reference text to tokenize for trace exactness diagnostics\n");
    printf("  --ngplus-reference-json JSON|@FILE\n");
    printf("                                optional OpenAI-compatible response JSON; extracts choices[0].message.content\n");
    printf("  --ngplus-reference-prompt-text TEXT|@FILE\n");
    printf("                                optional formatted prompt text for trace prompt-template diagnostics\n");
    printf("  --ngplus-reference-prompt-json JSON|@FILE\n");
    printf("                                optional server /apply-template JSON; extracts prompt for trace diagnostics\n");
    printf("  --ngplus-stop-after-reference-mismatch\n");
    printf("                                stop reference replay after the first mismatching generated token\n");
    printf("  --ngplus-force-reference-tokens\n");
    printf("                                diagnostic replay: accept reference tokens when present in sampler candidates\n");
    printf("  --ngplus-dump-prompt-token-ids FNAME\n");
    printf("                                write final prompt token IDs as JSON and exit before prompt eval\n");
    printf("  --ngplus-dump-no-bos\n");
    printf("                                omit BOS only for --ngplus-dump-prompt-token-ids tokenization\n");
    printf("  --ngplus-dump-no-parse-special\n");
    printf("                                disable special-token parsing only for --ngplus-dump-prompt-token-ids\n");
    printf("  --ngplus-no-single-turn\n");
    printf("                                disable the NG+ single-turn chat template for raw datastore tokenization\n");
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

static bool parse_on_off(const std::string & value, const std::string & arg) {
    if (value == "on" || value == "true" || value == "1") {
        return true;
    }
    if (value == "off" || value == "false" || value == "0") {
        return false;
    }
    throw std::invalid_argument("invalid on/off value for " + arg + ": " + value);
}

static bool has_csv_token(const std::string & csv, const std::string & token) {
    size_t begin = 0;
    while (begin <= csv.size()) {
        size_t end = csv.find(',', begin);
        if (end == std::string::npos) {
            end = csv.size();
        }
        std::string item = csv.substr(begin, end - begin);
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), item.end());
        if (item == token) {
            return true;
        }
        if (end == csv.size()) {
            break;
        }
        begin = end + 1;
    }
    return false;
}

static bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
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

static void normalize_phase4_source_args(ngplus_params & ngp) {
    if (has_csv_token(ngp.hot_source, "hot-table-candidates") ||
            has_csv_token(ngp.hot_source, "static-hot-table-candidates")) {
        ngp.static_hot_table_candidate_enabled = true;
    }
    if (has_csv_token(ngp.hot_source, "recent-generation") ||
            has_csv_token(ngp.hot_source, "self-ref")) {
        ngp.recent_generation_enabled = true;
    }
    if (has_csv_token(ngp.hot_source, "batched-verify")) {
        ngp.batched_verify_enabled = true;
    }
    if (has_csv_token(ngp.hot_source, "hot-table-chain")) {
        ngp.hot_table_chain_enabled = true;
    }
    if (has_csv_token(ngp.hot_source, "structure-indent")) {
        ngp.structure_indent_enabled = true;
    }
    if (has_csv_token(ngp.hot_source, "pl-fullspan")) {
        ngp.prompt_local_pos_cap = 1 << 30; // allow verbatim prompt echoes from anywhere in the prompt
    }
    if (has_csv_token(ngp.hot_source, "hybrid-verify")) {
        ngp.hybrid_verify_enabled = true;
    }
    // optional cap token "bv-k<N>" e.g. bv-k4 caps the correct verification batch length
    {
        std::stringstream ss(ngp.hot_source);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim spaces
            size_t b = tok.find_first_not_of(" \t");
            size_t e = tok.find_last_not_of(" \t");
            if (b == std::string::npos) continue;
            tok = tok.substr(b, e - b + 1);
            if (tok.rfind("bv-k", 0) == 0) {
                try { ngp.batched_verify_max_k = std::max(1, std::stoi(tok.substr(4))); }
                catch (...) {}
            }
        }
    }
    if (ngp.hot_table_path.empty() && !ngp.cold_path.empty() && ends_with(ngp.cold_path, ".jsonl")) {
        ngp.hot_table_path = ngp.cold_path;
    }
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
        } else if (name == "--ngplus-hot-table-path") {
            ngp.hot_table_path = value_for(name);
        } else if (name == "--ngplus-hot-table-candidates") {
            ngp.static_hot_table_candidate_enabled = parse_on_off(value_for(name), name);
        } else if (name == "--ngplus-hot-table-min-count") {
            ngp.static_hot_table_candidate_min_count = parse_positive_int(value_for(name), name);
        } else if (name == "--ngplus-hot-table-min-top-share-pct") {
            ngp.static_hot_table_candidate_min_top_share_pct = parse_positive_int(value_for(name), name);
            if (ngp.static_hot_table_candidate_min_top_share_pct > 100) {
                throw std::invalid_argument(name + " must be <= 100");
            }
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
        } else if (name == "--ngplus-reference-token-ids") {
            ngp.reference_token_ids_arg = value_for(name);
        } else if (name == "--ngplus-reference-text") {
            ngp.reference_text_arg = value_for(name);
        } else if (name == "--ngplus-reference-json") {
            ngp.reference_json_arg = value_for(name);
        } else if (name == "--ngplus-reference-prompt-text") {
            ngp.reference_prompt_text_arg = value_for(name);
        } else if (name == "--ngplus-reference-prompt-json") {
            ngp.reference_prompt_json_arg = value_for(name);
        } else if (name == "--ngplus-dump-prompt-token-ids") {
            ngp.dump_prompt_token_ids_path = value_for(name);
        } else if (name == "--ngplus-dump-no-bos") {
            ngp.dump_no_bos = true;
        } else if (name == "--ngplus-dump-no-parse-special") {
            ngp.dump_no_parse_special = true;
        } else if (name == "--ngplus-no-single-turn") {
            ngp.single_turn = false;
        } else if (name == "--ngplus-stop-after-reference-mismatch") {
            ngp.stop_after_reference_mismatch = true;
        } else if (name == "--ngplus-force-reference-tokens") {
            ngp.force_reference_tokens = true;
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

    normalize_phase4_source_args(ngp);

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

static std::string openai_response_json_for_text(
        const std::string & generated_text,
        int generated_text_bytes,
        const std::string & generated_text_fnv1a64) {
    std::ostringstream out;
    out << "{"
        << "\"object\":\"ngplus.trace.response\","
        << "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
        << json_escape(generated_text)
        << "\"},\"finish_reason\":\"length\"}],"
        << "\"ngplus_generated_text_bytes\":" << generated_text_bytes << ","
        << "\"ngplus_generated_text_fnv1a64\":\"" << json_escape(generated_text_fnv1a64) << "\""
        << "}";
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

static std::vector<llama_token> parse_reference_token_ids(const std::string & spec) {
    std::string payload = spec;
    if (!payload.empty() && payload[0] == '@') {
        const std::string path = payload.substr(1);
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::invalid_argument("failed to open --ngplus-reference-token-ids file: " + path);
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        payload = buffer.str();
    }

    std::vector<llama_token> out;
    std::string current;
    auto flush = [&]() {
        if (current.empty()) {
            return;
        }
        try {
            out.push_back((llama_token) std::stoll(current));
        } catch (const std::exception &) {
            throw std::invalid_argument("invalid token id in --ngplus-reference-token-ids: " + current);
        }
        current.clear();
    };

    for (char ch : payload) {
        if (std::isdigit((unsigned char) ch) || (ch == '-' && current.empty())) {
            current.push_back(ch);
        } else {
            flush();
        }
    }
    flush();
    return out;
}

static std::string read_inline_or_at_file(const std::string & spec, const std::string & arg_name) {
    if (spec.empty() || spec[0] != '@') {
        return spec;
    }

    const std::string path = spec.substr(1);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("failed to open " + arg_name + " file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::string parse_reference_text(const std::string & spec) {
    return read_inline_or_at_file(spec, "--ngplus-reference-text");
}

static std::string parse_reference_prompt_text(const std::string & spec) {
    return read_inline_or_at_file(spec, "--ngplus-reference-prompt-text");
}

static std::string parse_json_string_at(const std::string & payload, size_t quote_pos) {
    if (quote_pos >= payload.size() || payload[quote_pos] != '"') {
        throw std::invalid_argument("internal JSON parser expected string quote");
    }

    std::string out;
    for (size_t i = quote_pos + 1; i < payload.size(); ++i) {
        const char ch = payload[i];
        if (ch == '"') {
            return out;
        }
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (++i >= payload.size()) {
            break;
        }
        const char esc = payload[i];
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u':
                // The recorded Gemma4 artifacts are UTF-8 and rarely need escapes. Preserve
                // non-ASCII escapes as a visible placeholder instead of corrupting byte offsets.
                out += "\\u";
                for (int j = 0; j < 4 && i + 1 < payload.size(); ++j) {
                    out.push_back(payload[++i]);
                }
                break;
            default:
                out.push_back(esc);
                break;
        }
    }
    throw std::invalid_argument("unterminated JSON string in --ngplus-reference-json");
}

static std::string extract_first_json_string_value(const std::string & payload, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = payload.find(needle);
    while (pos != std::string::npos) {
        size_t colon = payload.find(':', pos + needle.size());
        if (colon == std::string::npos) {
            break;
        }
        size_t value = colon + 1;
        while (value < payload.size() && std::isspace((unsigned char) payload[value])) {
            ++value;
        }
        if (value < payload.size() && payload[value] == '"') {
            return parse_json_string_at(payload, value);
        }
        pos = payload.find(needle, pos + needle.size());
    }
    return "";
}

static std::string parse_reference_json_content(const std::string & spec) {
    const std::string payload = read_inline_or_at_file(spec, "--ngplus-reference-json");
    std::string content = extract_first_json_string_value(payload, "content");
    if (content.empty()) {
        // Some eval/debug artifacts store the generated completion under a flatter name.
        content = extract_first_json_string_value(payload, "generated_text");
    }
    if (content.empty()) {
        content = extract_first_json_string_value(payload, "text");
    }
    if (content.empty()) {
        throw std::invalid_argument(
            "--ngplus-reference-json could not find a non-empty content/generated_text/text string");
    }
    return content;
}

static std::string parse_reference_prompt_json_content(const std::string & spec) {
    const std::string payload = read_inline_or_at_file(spec, "--ngplus-reference-prompt-json");
    std::string prompt = extract_first_json_string_value(payload, "prompt");
    if (prompt.empty()) {
        throw std::invalid_argument("--ngplus-reference-prompt-json could not find a non-empty prompt string");
    }
    return prompt;
}

static int first_byte_mismatch(const std::string & actual, const std::string & expected) {
    const size_t common = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < common; ++i) {
        if (actual[i] != expected[i]) {
            return (int) i;
        }
    }
    return actual.size() == expected.size() ? -1 : (int) common;
}

static int first_token_mismatch(
        const std::vector<llama_token> & generated,
        const std::vector<llama_token> & reference) {
    const size_t common = std::min(generated.size(), reference.size());
    for (size_t i = 0; i < common; ++i) {
        if (generated[i] != reference[i]) {
            return (int) i;
        }
    }
    if (generated.size() > reference.size()) {
        return (int) reference.size();
    }
    return -1;
}

static std::string token_json_or_null(llama_token token) {
    if (token < 0) {
        return "null";
    }
    return std::to_string(token);
}

static std::string token_ids_json(
        const std::vector<llama_token> & tokens,
        size_t begin,
        size_t end) {
    std::ostringstream out;
    out << "[";
    const size_t n = tokens.size();
    begin = std::min(begin, n);
    end = std::min(end, n);
    for (size_t i = begin; i < end; ++i) {
        if (i > begin) {
            out << ",";
        }
        out << tokens[i];
    }
    out << "]";
    return out.str();
}

static std::string token_ids_prefix_json(const std::vector<llama_token> & tokens, size_t max_items) {
    return token_ids_json(tokens, 0, std::min(max_items, tokens.size()));
}

static std::string token_ids_tail_json(const std::vector<llama_token> & tokens, size_t max_items) {
    const size_t begin = tokens.size() > max_items ? tokens.size() - max_items : 0;
    return token_ids_json(tokens, begin, tokens.size());
}

static bool write_token_ids_json_file(const std::string & path, const std::vector<llama_token> & tokens) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << token_ids_json(tokens, 0, tokens.size()) << "\n";
    return out.good();
}

static std::string token_pieces_json(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t begin,
        size_t end) {
    std::ostringstream out;
    out << "[";
    const size_t n = tokens.size();
    begin = std::min(begin, n);
    end = std::min(end, n);
    for (size_t i = begin; i < end; ++i) {
        if (i > begin) {
            out << ",";
        }
        out << "\"" << json_escape(common_token_to_piece(ctx, tokens[i])) << "\"";
    }
    out << "]";
    return out.str();
}

static std::string token_pieces_prefix_json(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t max_items) {
    return token_pieces_json(ctx, tokens, 0, std::min(max_items, tokens.size()));
}

static std::string token_pieces_tail_json(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t max_items) {
    const size_t begin = tokens.size() > max_items ? tokens.size() - max_items : 0;
    return token_pieces_json(ctx, tokens, begin, tokens.size());
}

static std::string string_suffix(const std::string & input, size_t max_bytes) {
    if (input.size() <= max_bytes) {
        return input;
    }
    return input.substr(input.size() - max_bytes);
}

static std::string string_prefix(const std::string & input, size_t max_bytes) {
    if (input.size() <= max_bytes) {
        return input;
    }
    return input.substr(0, max_bytes);
}

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
static std::string raw_logit_surface_json(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_token reference_token,
        int fingerprint_items) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (logits == nullptr || n_vocab <= 0) {
        return "{\"candidate_count\":0,\"top_tokens\":[],\"top_logit\":null,\"second_logit\":null,\"top_margin\":null,\"fingerprint_items\":0,\"fingerprint_fnv1a64\":null,\"reference_raw_candidate\":null}";
    }

    const int n_items = std::max(0, std::min(fingerprint_items, n_vocab));
    std::vector<llama_token_data> top;
    top.reserve(n_vocab);

    int reference_rank = -1;
    float reference_logit = NAN;
    for (llama_token token = 0; token < n_vocab; ++token) {
        const float logit = logits[token];
        top.push_back(llama_token_data{ token, logit, 0.0f });
        if (token == reference_token) {
            reference_logit = logit;
        }
    }

    const auto by_logit_desc = [](const llama_token_data & a, const llama_token_data & b) {
        if (a.logit == b.logit) {
            return a.id < b.id;
        }
        return a.logit > b.logit;
    };
    if (n_items < (int) top.size()) {
        std::partial_sort(top.begin(), top.begin() + n_items, top.end(), by_logit_desc);
    } else {
        std::sort(top.begin(), top.end(), by_logit_desc);
    }

    if (reference_token >= 0) {
        reference_rank = 1;
        for (const llama_token_data & item : top) {
            if (item.id == reference_token) {
                break;
            }
            if (item.logit > reference_logit || (item.logit == reference_logit && item.id < reference_token)) {
                ++reference_rank;
            }
        }
    }

    const float top_logit = top.empty() ? NAN : top[0].logit;
    const float second_logit = top.size() > 1 ? top[1].logit : NAN;
    const float top_margin = top.size() > 1 ? top_logit - second_logit : NAN;

    std::ostringstream payload;
    payload << std::setprecision(9);
    std::ostringstream top_tokens;
    top_tokens << "[";
    std::ostringstream top_logits;
    top_logits << "[";
    for (int i = 0; i < n_items; ++i) {
        const llama_token_data & candidate = top[i];
        if (i > 0) {
            top_tokens << ",";
            top_logits << ",";
            payload << ";";
        }
        top_tokens << candidate.id;
        top_logits << json_float_or_null(candidate.logit);
        payload << candidate.id << ":" << candidate.logit;
    }
    top_tokens << "]";
    top_logits << "]";

    std::ostringstream reference_json;
    if (reference_token >= 0) {
        reference_json
            << "{"
            << "\"id\":" << reference_token << ","
            << "\"piece\":\"" << json_escape(common_token_to_piece(ctx, reference_token)) << "\","
            << "\"rank\":" << reference_rank << ","
            << "\"logit\":" << json_float_or_null(reference_logit)
            << "}";
    } else {
        reference_json << "null";
    }

    std::ostringstream out;
    out << "{"
        << "\"candidate_count\":" << n_vocab << ","
        << "\"top_tokens\":" << top_tokens.str() << ","
        << "\"top_logits\":" << top_logits.str() << ","
        << "\"top_logit\":" << json_float_or_null(top_logit) << ","
        << "\"second_logit\":" << json_float_or_null(second_logit) << ","
        << "\"top_margin\":" << json_float_or_null(top_margin) << ","
        << "\"fingerprint_items\":" << n_items << ","
        << "\"fingerprint_fnv1a64\":\"" << fnv1a64_hex(payload.str()) << "\","
        << "\"reference_raw_candidate\":" << reference_json.str()
        << "}";
    return out.str();
}

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

static std::string reference_candidate_json(
        llama_context * ctx,
        common_sampler * smpl,
        llama_token reference_token,
        llama_token selected) {
    if (reference_token < 0) {
        return "null";
    }

    llama_token_data_array * candidates = common_sampler_get_candidates(smpl, true);
    int rank = -1;
    float logit = NAN;
    float p = NAN;
    if (candidates != nullptr) {
        for (size_t i = 0; i < candidates->size; ++i) {
            const llama_token_data & candidate = candidates->data[i];
            if (candidate.id == reference_token) {
                rank = (int) i + 1;
                logit = candidate.logit;
                p = candidate.p;
                break;
            }
        }
    }

    std::ostringstream out;
    out << "{"
        << "\"id\":" << reference_token << ","
        << "\"piece\":\"" << json_escape(common_token_to_piece(ctx, reference_token)) << "\","
        << "\"rank\":" << (rank >= 0 ? std::to_string(rank) : "null") << ","
        << "\"logit\":" << json_float_or_null(logit) << ","
        << "\"p\":" << json_float_or_null(p) << ","
        << "\"selected\":" << (reference_token == selected ? "true" : "false")
        << "}";
    return out.str();
}

static int reference_candidate_rank(common_sampler * smpl, llama_token reference_token) {
    if (reference_token < 0) {
        return -1;
    }

    llama_token_data_array * candidates = common_sampler_get_candidates(smpl, true);
    if (candidates == nullptr) {
        return -1;
    }

    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].id == reference_token) {
            return (int) i + 1;
        }
    }
    return -1;
}

static std::string sampler_diagnostics_json(
        common_sampler * smpl,
        const common_params_sampling & sampling,
        llama_token selected) {
    llama_token_data_array * candidates = common_sampler_get_candidates(smpl, true);
    int selected_index = -1;
    float selected_logit = NAN;
    float selected_p = NAN;
    llama_token top_token = -1;
    float top_logit = NAN;
    float top_p = NAN;

    if (candidates != nullptr && candidates->size > 0) {
        top_token = candidates->data[0].id;
        top_logit = candidates->data[0].logit;
        top_p = candidates->data[0].p;
        for (size_t i = 0; i < candidates->size; ++i) {
            const llama_token_data & candidate = candidates->data[i];
            if (candidate.id == selected) {
                selected_index = (int) i;
                selected_logit = candidate.logit;
                selected_p = candidate.p;
                break;
            }
        }
    }

    std::ostringstream out;
    out << "{"
        << "\"seed\":" << common_sampler_get_seed(smpl) << ","
        << "\"configured_seed\":" << sampling.seed << ","
        << "\"temp\":" << json_float_or_null(sampling.temp) << ","
        << "\"top_k\":" << sampling.top_k << ","
        << "\"top_p\":" << json_float_or_null(sampling.top_p) << ","
        << "\"min_p\":" << json_float_or_null(sampling.min_p) << ","
        << "\"typ_p\":" << json_float_or_null(sampling.typ_p) << ","
        << "\"repeat_penalty\":" << json_float_or_null(sampling.penalty_repeat) << ","
        << "\"repeat_last_n\":" << sampling.penalty_last_n << ","
        << "\"candidate_count\":" << (candidates == nullptr ? 0 : (int) candidates->size) << ","
        << "\"cur_selected_index\":" << (candidates == nullptr ? -1 : candidates->selected) << ","
        << "\"selected_rank\":" << (selected_index >= 0 ? std::to_string(selected_index + 1) : "null") << ","
        << "\"selected_logit\":" << json_float_or_null(selected_logit) << ","
        << "\"selected_p\":" << json_float_or_null(selected_p) << ","
        << "\"top_token\":" << token_json_or_null(top_token) << ","
        << "\"top_logit\":" << json_float_or_null(top_logit) << ","
        << "\"top_p\":" << json_float_or_null(top_p) << ","
        << "\"selected_is_top\":" << (selected >= 0 && selected == top_token ? "true" : "false")
        << "}";
    return out.str();
}

static std::string logit_surface_json(common_sampler * smpl, int fingerprint_items) {
    llama_token_data_array * candidates = common_sampler_get_candidates(smpl, true);
    if (candidates == nullptr || candidates->size == 0) {
        return "{\"candidate_count\":0,\"top_tokens\":[],\"top_logit\":null,\"second_logit\":null,\"top_margin\":null,\"fingerprint_items\":0,\"fingerprint_fnv1a64\":null}";
    }

    const int n_items = std::max(0, std::min(fingerprint_items, (int) candidates->size));
    float top_logit = candidates->data[0].logit;
    float second_logit = candidates->size > 1 ? candidates->data[1].logit : NAN;
    float top_margin = candidates->size > 1 ? top_logit - second_logit : NAN;

    std::ostringstream payload;
    payload << std::setprecision(9);
    std::ostringstream top_tokens;
    top_tokens << "[";
    for (int i = 0; i < n_items; ++i) {
        const llama_token_data & candidate = candidates->data[i];
        if (i > 0) {
            top_tokens << ",";
            payload << ";";
        }
        top_tokens << candidate.id;
        payload
            << candidate.id << ":"
            << candidate.logit << ":"
            << candidate.p;
    }
    top_tokens << "]";

    std::ostringstream out;
    out << "{"
        << "\"candidate_count\":" << (int) candidates->size << ","
        << "\"top_tokens\":" << top_tokens.str() << ","
        << "\"top_logit\":" << json_float_or_null(top_logit) << ","
        << "\"second_logit\":" << json_float_or_null(second_logit) << ","
        << "\"top_margin\":" << json_float_or_null(top_margin) << ","
        << "\"fingerprint_items\":" << n_items << ","
        << "\"fingerprint_fnv1a64\":\"" << fnv1a64_hex(payload.str()) << "\""
        << "}";
    return out.str();
}
#endif

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
static bool apply_single_turn_chat_template(common_params & params, llama_model * model, ngplus_params & ngp) {
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
    if (!chat_params.grammar.empty()) {
        params.sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_params.grammar);
    }
    params.sampling.grammar_lazy = chat_params.grammar_lazy;
    params.sampling.grammar_triggers = chat_params.grammar_triggers;
    params.sampling.preserved_tokens.clear();
    for (const auto & token : chat_params.preserved_tokens) {
        const auto tokenized = common_tokenize(llama_model_get_vocab(model), token, false, true);
        params.sampling.preserved_tokens.insert(tokenized.begin(), tokenized.end());
    }
    params.sampling.generation_prompt = chat_params.generation_prompt;
    if (!chat_params.thinking_end_tag.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        params.sampling.reasoning_budget_start = common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
        params.sampling.reasoning_budget_end = common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
        params.sampling.reasoning_budget_forced = common_tokenize(
            vocab,
            params.sampling.reasoning_budget_message + chat_params.thinking_end_tag,
            false,
            true);
    }
    ngp.chat_grammar_lazy = chat_params.grammar_lazy;
    ngp.reasoning_budget_start_tokens = (int) params.sampling.reasoning_budget_start.size();
    ngp.reasoning_budget_end_tokens = (int) params.sampling.reasoning_budget_end.size();
    ngp.reasoning_budget_forced_tokens = (int) params.sampling.reasoning_budget_forced.size();

    return true;
}
#endif

static prompt_draft_result prompt_local_draft(
        const std::vector<llama_token> & history,
        int max_order,
        int max_draft,
        int trusted_order_min,
        int trusted_max_draft,
        int prompt_tokens,
        int prompt_pos_cap) {
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
            if (continuation_start >= prompt_tokens) {
                continue;
            }
            if (order < 4) {
                continue;
            }
            if (order >= 4 && pos > prompt_pos_cap) {
                continue;
            }
            if (order == 2 && pos > 11) {
                continue;
            }
            const int available = history_size - continuation_start;
            int order_max_draft =
                order >= trusted_order_min ? std::max(max_draft, trusted_max_draft) : max_draft;
            if (order == 2) {
                order_max_draft = std::min(order_max_draft, 4);
            }
            const int n_copy = std::min(order_max_draft, available);
            if (n_copy <= 0) {
                continue;
            }

            result.tokens.insert(
                result.tokens.end(),
                history.begin() + continuation_start,
                history.begin() + continuation_start + n_copy);
            result.order = order;
            result.source_pos = pos;
            result.continuation_start = continuation_start;
            result.continuation_available = available;
            result.continuation_copied = n_copy;
            result.truncated_by_draft_limit = n_copy < available;
            return result;
        }
    }

    return result;
}

// Phase 5 high-leap (iter3): self-referential generation cache.
// prompt_local_draft only drafts continuations that START inside the prompt
// (continuation_start >= prompt_tokens -> skip). This drafts from the model's
// OWN accepted output: search generated history for an earlier occurrence of the
// current suffix and copy its continuation. Routed through the correct per-token
// verify path (source_label "recent-generation-hot", never the blind trusted
// suffix), so it cannot worsen AR-hash agreement. Targets no_draft_rows on novel
// body code, where repeated identifiers/structures recur within one solution.
static prompt_draft_result recent_generation_draft(
        const std::vector<llama_token> & history,
        int prompt_tokens,
        int max_order,
        int min_order,
        int max_draft) {
    prompt_draft_result result;
    const int history_size = (int) history.size();
    // require at least some generated tokens beyond the prompt
    if (max_draft <= 0 || history_size <= prompt_tokens + min_order) {
        return result;
    }

    const int order_max = std::min(max_order, history_size);
    for (int order = order_max; order >= min_order; --order) {
        const int key_start = history_size - order;
        if (key_start <= 0) {
            continue;
        }
        for (int pos = key_start - 1; pos >= 0; --pos) {
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
            if (continuation_start >= key_start) {
                continue; // overlapping match, no usable continuation
            }
            const int available = key_start - continuation_start;
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
            result.continuation_start = continuation_start;
            result.continuation_available = available;
            result.continuation_copied = n_copy;
            result.truncated_by_draft_limit = n_copy < available;
            result.source_label = "recent-generation-hot";
            return result;
        }
    }

    return result;
}

static int parse_int_field(const std::string & line, const std::string & key, int default_value) {
    const std::string needle = "\"" + key + "\":";
    const size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return default_value;
    }
    size_t begin = pos + needle.size();
    size_t end = begin;
    if (end < line.size() && line[end] == '-') {
        ++end;
    }
    while (end < line.size() && std::isdigit((unsigned char) line[end])) {
        ++end;
    }
    if (end == begin || (end == begin + 1 && line[begin] == '-')) {
        return default_value;
    }
    return std::stoi(line.substr(begin, end - begin));
}

static llama_tokens parse_token_array_field(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":[";
    const size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        throw std::invalid_argument("missing " + key + " array");
    }
    size_t cursor = pos + needle.size();
    const size_t end_array = line.find(']', cursor);
    if (end_array == std::string::npos) {
        throw std::invalid_argument("unterminated " + key + " array");
    }

    llama_tokens tokens;
    while (cursor < end_array) {
        while (cursor < end_array && (line[cursor] == ',' || std::isspace((unsigned char) line[cursor]))) {
            ++cursor;
        }
        if (cursor >= end_array) {
            break;
        }
        size_t end = cursor;
        if (end < end_array && line[end] == '-') {
            ++end;
        }
        while (end < end_array && std::isdigit((unsigned char) line[end])) {
            ++end;
        }
        if (end == cursor || (end == cursor + 1 && line[cursor] == '-')) {
            throw std::invalid_argument("invalid integer in " + key + " array");
        }
        tokens.push_back((llama_token) std::stoi(line.substr(cursor, end - cursor)));
        cursor = end;
    }
    return tokens;
}

static std::vector<std::pair<llama_token, int>> parse_next_pairs(const std::string & line) {
    const std::string needle = "\"next\":[";
    size_t cursor = line.find(needle);
    if (cursor == std::string::npos) {
        throw std::invalid_argument("missing next array");
    }
    cursor += needle.size();

    std::vector<std::pair<llama_token, int>> next;
    while (true) {
        const size_t obj = line.find("{\"count\":", cursor);
        if (obj == std::string::npos) {
            break;
        }
        const size_t token_key = line.find("\"token\":", obj);
        if (token_key == std::string::npos) {
            throw std::invalid_argument("next entry missing token");
        }
        const size_t obj_end = line.find('}', token_key);
        if (obj_end == std::string::npos) {
            throw std::invalid_argument("unterminated next entry");
        }

        const int count = parse_int_field(line.substr(obj, obj_end - obj + 1), "count", 0);
        const int token = parse_int_field(line.substr(obj, obj_end - obj + 1), "token", -1);
        if (token >= 0) {
            next.emplace_back((llama_token) token, count);
        }
        cursor = obj_end + 1;
    }
    return next;
}

static static_hot_table load_static_hot_table(const std::string & path) {
    static_hot_table table;
    if (path.empty()) {
        return table;
    }

    std::ifstream bytes_in(path, std::ios::binary | std::ios::ate);
    if (!bytes_in.is_open()) {
        throw std::invalid_argument("failed to open --ngplus-hot-table-path: " + path);
    }
    table.bytes = (int64_t) bytes_in.tellg();
    bytes_in.close();

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::invalid_argument("failed to read --ngplus-hot-table-path: " + path);
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        try {
            hot_table_entry entry;
            entry.context = parse_token_array_field(line, "context");
            entry.next = parse_next_pairs(line);
            entry.total_count = parse_int_field(line, "total_count", 0);
            entry.rank = parse_int_field(line, "rank", -1);
            if (entry.context.empty() || entry.next.empty()) {
                continue;
            }
            if (table.order == 0) {
                table.order = (int) entry.context.size();
            }
            if ((int) entry.context.size() != table.order) {
                throw std::invalid_argument("hot-table row order mismatch");
            }
            table.entries.emplace(token_key(entry.context), std::move(entry));
        } catch (const std::exception & e) {
            throw std::invalid_argument(
                "invalid hot-table JSONL at line " + std::to_string(line_no) + ": " + e.what());
        }
    }

    return table;
}

static hot_table_lookup_result static_hot_table_lookup(
        const static_hot_table & table,
        const std::vector<llama_token> & history) {
    hot_table_lookup_result result;
    result.order = table.order;
    if (table.order <= 0 || (int) history.size() < table.order) {
        return result;
    }

    const std::string key = token_key_suffix(history, table.order);
    const auto it = table.entries.find(key);
    if (it == table.entries.end()) {
        return result;
    }

    const hot_table_entry & entry = it->second;
    result.hit = true;
    result.total_count = entry.total_count;
    result.rank = entry.rank;
    result.candidate_count = (int) entry.next.size();
    if (!entry.next.empty()) {
        result.top_token = entry.next[0].first;
        result.top_count = entry.next[0].second;
    }
    return result;
}

static bool static_hot_table_candidate_allowed(
        const hot_table_lookup_result & lookup,
        const ngplus_params & ngp) {
    if (!ngp.static_hot_table_candidate_enabled || !lookup.hit || lookup.top_token < 0) {
        return false;
    }
    if (lookup.top_count < ngp.static_hot_table_candidate_min_count || lookup.total_count <= 0) {
        return false;
    }
    return lookup.top_count * 100 >= lookup.total_count * ngp.static_hot_table_candidate_min_top_share_pct;
}

static prompt_draft_result static_hot_table_draft(
        const hot_table_lookup_result & lookup,
        const ngplus_params & ngp,
        int draft_limit) {
    prompt_draft_result result;
    if (draft_limit <= 0 || !static_hot_table_candidate_allowed(lookup, ngp)) {
        return result;
    }
    result.tokens.push_back(lookup.top_token);
    result.order = lookup.order;
    result.source_pos = -1;
    result.continuation_start = -1;
    result.continuation_available = lookup.candidate_count;
    result.continuation_copied = 1;
    result.truncated_by_draft_limit = false;
    result.source_label = "static-hot-table";
    return result;
}

// Phase 5 high-leap (iter6, primary thrust): code-distribution multi-token chained store draft.
// Instead of one top token, follow the store chain: emit the gated top token, shift the order-N
// context by one, look up again, repeat up to draft_limit. Produces *chained* multi-token
// continuations from a disjoint code corpus for body steps. Verified per-token (or via batched
// verify when enabled); never blind-trusted.
static prompt_draft_result static_hot_table_chain_draft(
        const static_hot_table & table,
        const ngplus_params & ngp,
        std::vector<llama_token> history,
        int draft_limit) {
    prompt_draft_result result;
    if (draft_limit <= 0 || table.order <= 0) {
        return result;
    }
    for (int n = 0; n < draft_limit; ++n) {
        hot_table_lookup_result lk = static_hot_table_lookup(table, history);
        if (!static_hot_table_candidate_allowed(lk, ngp)) {
            break;
        }
        result.tokens.push_back(lk.top_token);
        history.push_back(lk.top_token);
    }
    if (!result.tokens.empty()) {
        result.order = table.order;
        result.source_pos = -1;
        result.continuation_start = -1;
        result.continuation_available = (int) result.tokens.size();
        result.continuation_copied = (int) result.tokens.size();
        result.source_label = "static-hot-table";
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
        int source_continuation_start,
        int source_continuation_available,
        int source_continuation_copied,
        bool source_truncated_by_draft_limit,
        const hot_table_lookup_result & hot_table_lookup,
        int64_t hot_lookup_us,
        int64_t tree_build_us,
        int64_t target_verify_us,
        int64_t kv_cleanup_us,
        int output_tokens_total,
        llama_token sampled_token,
        llama_token previous_sampled_token,
        const std::string & sampled_piece,
        const std::string & previous_sampled_piece,
        const std::string & draft_source,
        const std::string & generated_prefix,
        const std::string & generated_text_final,
        int generated_text_bytes,
        const std::string & generated_text_fnv1a64,
        const std::string & generated_response_json_final,
        const std::string & generated_token_ids,
        const std::string & generated_token_ids_prefix,
        const std::string & generated_token_ids_tail,
        const std::string & generated_token_pieces,
        const std::string & generated_token_pieces_prefix,
        const std::string & generated_token_pieces_tail,
        const std::string & top_candidates,
        const std::string & sampler_diagnostics,
        const std::string & logit_surface,
        const std::string & raw_logit_surface,
        llama_token sampler_selected_token,
        const std::string & sampler_selected_piece,
        bool reference_forced,
        int reference_forced_rank,
        const std::string & reference_source,
        int reference_token_count,
        int reference_text_bytes,
        const std::string & reference_text_fnv1a64,
        const std::string & reference_current_candidate,
        int reference_first_mismatch_index,
        llama_token reference_expected_token,
        llama_token reference_actual_token,
        const std::string & reference_expected_piece,
        const std::string & reference_actual_piece,
        bool reference_prefix_matches,
        bool reference_final_matches,
        bool generated_full_sequence_final) {
    if (!trace.is_open()) {
        return;
    }

    trace << "{"
          << "\"event\":\"step\","
          << "\"step\":" << step << ","
          << "\"source\":\"" << json_escape(draft_source) << "\","
          << "\"hot_source\":\"" << json_escape(ngp.hot_source) << "\","
          << "\"cold_source\":\"noop\","
          << "\"cold_path\":\"" << json_escape(ngp.cold_path) << "\","
          << "\"cold_mmap\":\"" << json_escape(ngp.cold_mmap) << "\","
          << "\"prompt_format\":\"" << (ngp.chat_template_applied ? "chat-single-turn" : "raw") << "\","
          << "\"prompt_tokens\":" << ngp.prompt_tokens << ","
          << "\"prompt_bytes\":" << ngp.prompt_bytes << ","
          << "\"prompt_fingerprint\":\"" << ngp.prompt_fingerprint << "\","
          << "\"prompt_token_head\":" << ngp.prompt_token_head_json << ","
          << "\"prompt_token_tail\":" << ngp.prompt_token_tail_json << ","
          << "\"prompt_token_head_pieces\":" << ngp.prompt_token_head_pieces_json << ","
          << "\"prompt_token_tail_pieces\":" << ngp.prompt_token_tail_pieces_json << ","
          << "\"prompt_suffix\":\"" << json_escape(ngp.prompt_suffix) << "\","
          << "\"prompt_text_final\":"
          << (generated_full_sequence_final ? ("\"" + json_escape(ngp.prompt_text) + "\"") : "null") << ","
          << "\"prompt_reference_source\":\"" << json_escape(ngp.reference_prompt_source) << "\","
          << "\"prompt_reference_text_bytes\":" << ngp.reference_prompt_text_bytes << ","
          << "\"prompt_reference_text_fnv1a64\":"
          << (ngp.reference_prompt_text_fnv1a64.empty() ? "null" : ("\"" + json_escape(ngp.reference_prompt_text_fnv1a64) + "\"")) << ","
          << "\"prompt_reference_matches\":"
          << (ngp.reference_prompt_source == "none" ? "null" : (ngp.reference_prompt_matches ? "true" : "false")) << ","
          << "\"prompt_reference_first_mismatch_byte\":"
          << (ngp.reference_prompt_first_mismatch_byte >= 0 ? std::to_string(ngp.reference_prompt_first_mismatch_byte) : "null") << ","
          << "\"chat_generation_prompt\":\"" << json_escape(ngp.chat_generation_prompt) << "\","
          << "\"chat_grammar_lazy\":" << (ngp.chat_grammar_lazy ? "true" : "false") << ","
          << "\"reasoning_budget_start_tokens\":" << ngp.reasoning_budget_start_tokens << ","
          << "\"reasoning_budget_end_tokens\":" << ngp.reasoning_budget_end_tokens << ","
          << "\"reasoning_budget_forced_tokens\":" << ngp.reasoning_budget_forced_tokens << ","
          << "\"prompt_sampler_seeded\":" << (ngp.prompt_sampler_seeded ? "true" : "false") << ","
          << "\"backend_sampling_requested\":" << (ngp.backend_sampling_requested ? "true" : "false") << ","
          << "\"server_backend_sampling\":" << (ngp.server_backend_sampling ? "true" : "false") << ","
          << "\"sampler_chain\":\"" << json_escape(ngp.sampler_chain) << "\","
          << "\"sampler_diagnostics\":" << sampler_diagnostics << ","
          << "\"logit_surface\":" << logit_surface << ","
          << "\"raw_logit_surface\":" << raw_logit_surface << ","
          << "\"verification_mode\":\"ar_exact_prefill_diagnostic_draft\","
          << "\"draft_acceptance_enabled\":" << (ngp.draft_acceptance_enabled ? "true" : "false") << ","
          << "\"device_fallback\":" << (ngp.cpu_fallback ? "\"cpu_no_usable_offload_device\"" : "null") << ","
          << "\"ngram_min\":" << ngp.effective_ngram_min << ","
          << "\"ngram_max\":" << ngp.effective_ngram_max << ","
          << "\"source_order\":" << source_order << ","
          << "\"source_pos\":" << source_pos << ","
          << "\"source_continuation_start\":" << source_continuation_start << ","
          << "\"source_continuation_available\":" << source_continuation_available << ","
          << "\"source_continuation_copied\":" << source_continuation_copied << ","
          << "\"source_truncated_by_draft_limit\":" << (source_truncated_by_draft_limit ? "true" : "false") << ","
          << "\"source_truncation_reason\":"
          << (source_truncated_by_draft_limit ? "\"ngplus_draft_limit\"" : "null") << ","
          << "\"static_hot_table_loaded\":" << (ngp.static_hot_table_loaded ? "true" : "false") << ","
          << "\"static_hot_table_path\":\"" << json_escape(ngp.hot_table_path) << "\","
          << "\"static_hot_table_candidate_enabled\":" << (ngp.static_hot_table_candidate_enabled ? "true" : "false") << ","
          << "\"static_hot_table_candidate_min_count\":" << ngp.static_hot_table_candidate_min_count << ","
          << "\"static_hot_table_candidate_min_top_share_pct\":" << ngp.static_hot_table_candidate_min_top_share_pct << ","
          << "\"static_hot_table_order\":" << ngp.static_hot_table_order << ","
          << "\"static_hot_table_rows\":" << ngp.static_hot_table_rows << ","
          << "\"static_hot_table_bytes\":" << ngp.static_hot_table_bytes << ","
          << "\"static_hot_table_load_us\":" << ngp.static_hot_table_load_us << ","
          << "\"static_hot_table_hit\":" << (hot_table_lookup.hit ? "true" : "false") << ","
          << "\"static_hot_table_lookup_us\":" << hot_table_lookup.lookup_us << ","
          << "\"static_hot_table_lookup_order\":" << hot_table_lookup.order << ","
          << "\"static_hot_table_lookup_rank\":"
          << (hot_table_lookup.rank >= 0 ? std::to_string(hot_table_lookup.rank) : "null") << ","
          << "\"static_hot_table_total_count\":" << hot_table_lookup.total_count << ","
          << "\"static_hot_table_candidate_count\":" << hot_table_lookup.candidate_count << ","
          << "\"static_hot_table_top_token\":"
          << (hot_table_lookup.top_token >= 0 ? std::to_string(hot_table_lookup.top_token) : "null") << ","
          << "\"static_hot_table_top_count\":" << hot_table_lookup.top_count << ","
          << "\"static_hot_table_hits_total\":" << ngp.static_hot_table_hits << ","
          << "\"static_hot_table_misses_total\":" << ngp.static_hot_table_misses << ","
          << "\"static_hot_table_lookup_us_total\":" << ngp.static_hot_table_lookup_us_total << ","
          << "\"ngplus_draft\":" << ngp.draft << ","
          << "\"tree_budget\":" << ngp.tree_budget << ","
          << "\"drafted_tokens\":" << drafted_tokens << ","
          << "\"accepted_tokens\":" << accepted_tokens << ","
          << "\"prompt_local_drafted_tokens_total\":" << ngp.prompt_local_drafted_tokens << ","
          << "\"recent_generation_drafted_tokens_total\":" << ngp.recent_generation_drafted_tokens << ","
          << "\"static_hot_table_drafted_tokens_total\":" << ngp.static_hot_table_drafted_tokens << ","
          << "\"fallback_steps_total\":" << ngp.fallback_steps << ","
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
          << "\"previous_sampled_token\":" << previous_sampled_token << ","
          << "\"sampled_piece\":\"" << json_escape(sampled_piece) << "\","
          << "\"previous_sampled_piece\":\"" << json_escape(previous_sampled_piece) << "\","
          << "\"generated_prefix\":\"" << json_escape(generated_prefix) << "\","
          << "\"generated_full_sequence_final\":" << (generated_full_sequence_final ? "true" : "false") << ","
          << "\"generated_text_final\":\"" << json_escape(generated_text_final) << "\","
          << "\"generated_text_bytes\":" << generated_text_bytes << ","
          << "\"generated_text_fnv1a64\":\"" << json_escape(generated_text_fnv1a64) << "\","
          << "\"generated_response_json_final\":"
          << (generated_response_json_final.empty() ? "null" : generated_response_json_final) << ","
          << "\"generated_token_ids\":" << generated_token_ids << ","
          << "\"generated_token_ids_prefix\":" << generated_token_ids_prefix << ","
          << "\"generated_token_ids_tail\":" << generated_token_ids_tail << ","
          << "\"generated_token_pieces\":" << generated_token_pieces << ","
          << "\"generated_token_pieces_prefix\":" << generated_token_pieces_prefix << ","
          << "\"generated_token_pieces_tail\":" << generated_token_pieces_tail << ","
          << "\"reference_source\":\"" << json_escape(reference_source) << "\","
          << "\"reference_stop_after_mismatch\":" << (ngp.stop_after_reference_mismatch ? "true" : "false") << ","
          << "\"reference_force_enabled\":" << (ngp.force_reference_tokens ? "true" : "false") << ","
          << "\"reference_forced\":" << (reference_forced ? "true" : "false") << ","
          << "\"reference_forced_rank\":"
          << (reference_forced_rank >= 0 ? std::to_string(reference_forced_rank) : "null") << ","
          << "\"sampler_selected_token\":" << token_json_or_null(sampler_selected_token) << ","
          << "\"sampler_selected_piece\":"
          << (sampler_selected_piece.empty() ? "null" : ("\"" + json_escape(sampler_selected_piece) + "\"")) << ","
          << "\"reference_token_count\":" << reference_token_count << ","
          << "\"reference_text_bytes\":" << reference_text_bytes << ","
          << "\"reference_text_fnv1a64\":"
          << (reference_text_fnv1a64.empty() ? "null" : ("\"" + json_escape(reference_text_fnv1a64) + "\"")) << ","
          << "\"reference_current_candidate\":" << reference_current_candidate << ","
          << "\"reference_prefix_matches\":" << (reference_prefix_matches ? "true" : "false") << ","
          << "\"reference_final_matches\":"
          << (generated_full_sequence_final && reference_token_count > 0 ? (reference_final_matches ? "true" : "false") : "null") << ","
          << "\"reference_first_mismatch_index\":"
          << (reference_first_mismatch_index >= 0 ? std::to_string(reference_first_mismatch_index) : "null") << ","
          << "\"reference_expected_token\":" << token_json_or_null(reference_expected_token) << ","
          << "\"reference_expected_piece\":"
          << (reference_expected_piece.empty() ? "null" : ("\"" + json_escape(reference_expected_piece) + "\"")) << ","
          << "\"reference_actual_token\":" << token_json_or_null(reference_actual_token) << ","
          << "\"reference_actual_piece\":"
          << (reference_actual_piece.empty() ? "null" : ("\"" + json_escape(reference_actual_piece) + "\"")) << ","
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
    try {
        const int reference_arg_count =
            (!ngp.reference_token_ids_arg.empty() ? 1 : 0) +
            (!ngp.reference_text_arg.empty() ? 1 : 0) +
            (!ngp.reference_json_arg.empty() ? 1 : 0);
        if (reference_arg_count > 1) {
            throw std::invalid_argument(
                "--ngplus-reference-token-ids, --ngplus-reference-text, and --ngplus-reference-json are mutually exclusive");
        }
        if (!ngp.reference_prompt_text_arg.empty() && !ngp.reference_prompt_json_arg.empty()) {
            throw std::invalid_argument(
                "--ngplus-reference-prompt-text and --ngplus-reference-prompt-json are mutually exclusive");
        }
        if (!ngp.reference_token_ids_arg.empty()) {
            ngp.reference_token_ids = parse_reference_token_ids(ngp.reference_token_ids_arg);
            ngp.reference_source = "token-ids";
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
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

    try {
        if (!ngp.reference_text_arg.empty() || !ngp.reference_json_arg.empty()) {
            const bool from_json = !ngp.reference_json_arg.empty();
            const std::string reference_text = from_json ?
                parse_reference_json_content(ngp.reference_json_arg) :
                parse_reference_text(ngp.reference_text_arg);
            ngp.reference_text_bytes = (int) reference_text.size();
            ngp.reference_text_fnv1a64 = fnv1a64_hex(reference_text);
            ngp.reference_token_ids = common_tokenize(vocab, reference_text, false, true);
            ngp.reference_source = from_json ? "json-content" : "text";
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        llama_backend_free();
        return 1;
    }

#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
    if (ngp.single_turn) {
        try {
            params.enable_reasoning = 0;
            params.default_template_kwargs["enable_thinking"] = "false";
            ngp.chat_template_applied = apply_single_turn_chat_template(params, model, ngp);
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

    const bool dump_mode = !ngp.dump_prompt_token_ids_path.empty();
    const bool prompt_add_bos = !(dump_mode && ngp.dump_no_bos);
    const bool prompt_parse_special = !(dump_mode && ngp.dump_no_parse_special);
    std::vector<llama_token> history = common_tokenize(ctx, params.prompt, prompt_add_bos, prompt_parse_special);
    if (history.empty()) {
        LOG_ERR("prompt tokenization produced no tokens\n");
        llama_backend_free();
        return 1;
    }
    ngp.prompt_tokens = (int) history.size();
    ngp.prompt_bytes = (int) params.prompt.size();
    ngp.prompt_fingerprint = fnv1a64_hex(params.prompt);
    ngp.prompt_token_head_json = token_ids_json(history, 0, std::min<size_t>(16, history.size()));
    ngp.prompt_token_tail_json = token_ids_json(history, history.size() > 16 ? history.size() - 16 : 0, history.size());
    ngp.prompt_token_head_pieces_json = token_pieces_json(ctx, history, 0, std::min<size_t>(16, history.size()));
    ngp.prompt_token_tail_pieces_json = token_pieces_json(ctx, history, history.size() > 16 ? history.size() - 16 : 0, history.size());
    ngp.prompt_text = params.prompt;
    ngp.prompt_suffix = string_suffix(params.prompt, 512);
    ngp.chat_generation_prompt = params.sampling.generation_prompt;

    if (!ngp.dump_prompt_token_ids_path.empty()) {
        if (!write_token_ids_json_file(ngp.dump_prompt_token_ids_path, history)) {
            LOG_ERR("failed to write --ngplus-dump-prompt-token-ids file: %s\n", ngp.dump_prompt_token_ids_path.c_str());
            llama_backend_free();
            return 1;
        }
        LOG_INF("ngplus: wrote %d prompt token IDs to %s\n", (int) history.size(), ngp.dump_prompt_token_ids_path.c_str());
        llama_backend_free();
        return 0;
    }

    try {
        if (!ngp.reference_prompt_text_arg.empty() || !ngp.reference_prompt_json_arg.empty()) {
            const bool from_json = !ngp.reference_prompt_json_arg.empty();
            const std::string reference_prompt_text = from_json ?
                parse_reference_prompt_json_content(ngp.reference_prompt_json_arg) :
                parse_reference_prompt_text(ngp.reference_prompt_text_arg);
            ngp.reference_prompt_text_bytes = (int) reference_prompt_text.size();
            ngp.reference_prompt_text_fnv1a64 = fnv1a64_hex(reference_prompt_text);
            ngp.reference_prompt_first_mismatch_byte = first_byte_mismatch(params.prompt, reference_prompt_text);
            ngp.reference_prompt_matches = ngp.reference_prompt_first_mismatch_byte < 0;
            ngp.reference_prompt_source = from_json ? "json-prompt" : "text";
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        llama_backend_free();
        return 1;
    }

    static_hot_table hot_table;
    const int64_t t_hot_init_start_us = ggml_time_us();
    try {
        if (!ngp.hot_table_path.empty()) {
            hot_table = load_static_hot_table(ngp.hot_table_path);
            ngp.static_hot_table_loaded = !hot_table.entries.empty();
            ngp.static_hot_table_order = hot_table.order;
            ngp.static_hot_table_rows = (int) hot_table.entries.size();
            ngp.static_hot_table_bytes = hot_table.bytes;
            LOG_INF(
                "ngplus: loaded static hot table rows=%d order=%d bytes=%lld from %s\n",
                ngp.static_hot_table_rows,
                ngp.static_hot_table_order,
                (long long) ngp.static_hot_table_bytes,
                ngp.hot_table_path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        llama_backend_free();
        return 1;
    }
    const int64_t t_hot_init_us = ggml_time_us() - t_hot_init_start_us;
    ngp.static_hot_table_load_us = t_hot_init_us;

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
    ngp.draft_acceptance_enabled = true;

    const auto t_dec_start = ggml_time_us();
    int step = 0;
    llama_token previous_sampled_token = -1;
    std::string previous_sampled_piece;
    std::vector<llama_token> generated_token_ids;

    while (!has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
        const int remaining = params.n_predict < 0 ? n_draft : std::max(1, params.n_predict - n_predict);
        const int draft_limit = std::max(0, std::min(n_draft, remaining));
        int trusted_draft_limit = std::max(draft_limit, std::min(32, remaining));
        if (remaining >= 40) {
            trusted_draft_limit = std::max(trusted_draft_limit, std::min(64, remaining));
        }

        const int64_t t_draft_start_us = ggml_time_us();
        prompt_draft_result draft_result = prompt_local_draft(
            history, ngp.effective_ngram_max, draft_limit, 4, trusted_draft_limit, ngp.prompt_tokens,
            ngp.prompt_local_pos_cap);
        const int64_t draft_us = ggml_time_us() - t_draft_start_us;

        hot_table_lookup_result hot_table_lookup;
        const int64_t t_static_hot_lookup_start_us = ggml_time_us();
        if (ngp.static_hot_table_loaded) {
            hot_table_lookup = static_hot_table_lookup(hot_table, history);
            if (hot_table_lookup.hit) {
                ++ngp.static_hot_table_hits;
            } else {
                ++ngp.static_hot_table_misses;
            }
        }
        hot_table_lookup.lookup_us = ggml_time_us() - t_static_hot_lookup_start_us;
        ngp.static_hot_table_lookup_us_total += hot_table_lookup.lookup_us;

        if (draft_result.tokens.empty() && static_hot_table_candidate_allowed(hot_table_lookup, ngp)) {
            draft_result = ngp.hot_table_chain_enabled
                ? static_hot_table_chain_draft(hot_table, ngp, history, draft_limit)
                : static_hot_table_draft(hot_table_lookup, ngp, draft_limit);
        }

        if (draft_result.tokens.empty() && ngp.recent_generation_enabled) {
            draft_result = recent_generation_draft(
                history, ngp.prompt_tokens, ngp.effective_ngram_max,
                ngp.recent_generation_min_order, draft_limit);
        }

        // Phase 5 high-leap (iter7): model-free structure-aware indentation drafter. After a token
        // that ends a line, draft the leading-whitespace tokens of the line that just ended (Python
        // body code usually repeats indentation). No corpus needed; verified per-token (exact).
        if (draft_result.tokens.empty() && ngp.structure_indent_enabled && draft_limit > 0 &&
                history.size() >= 2) {
            const std::string last_piece = common_token_to_piece(ctx, history.back());
            if (last_piece.find('\n') != std::string::npos) {
                const int end = (int) history.size() - 1; // newline token index
                int prev_nl = -1;
                for (int i = end - 1; i >= 0 && end - i < 200; --i) {
                    if (common_token_to_piece(ctx, history[i]).find('\n') != std::string::npos) {
                        prev_nl = i;
                        break;
                    }
                }
                prompt_draft_result r;
                for (int i = prev_nl + 1; i < end && (int) r.tokens.size() < draft_limit; ++i) {
                    const std::string p = common_token_to_piece(ctx, history[i]);
                    bool ws = !p.empty();
                    for (char c : p) {
                        if (c != ' ' && c != '\t' && c != '\n') { ws = false; break; }
                    }
                    if (!ws) {
                        break;
                    }
                    r.tokens.push_back(history[i]);
                }
                if (!r.tokens.empty()) {
                    r.order = 0;
                    r.source_label = "structure-indent";
                    r.continuation_copied = (int) r.tokens.size();
                    r.continuation_available = (int) r.tokens.size();
                    draft_result = r;
                }
            }
        }

        const llama_tokens & draft = draft_result.tokens;
        const std::string draft_source = draft_source_label(draft_result, ngp.prompt_tokens);
        if (draft_source == "prompt-local-hot") {
            ngp.prompt_local_drafted_tokens += (int) draft.size();
        } else if (draft_source == "recent-generation-hot") {
            ngp.recent_generation_drafted_tokens += (int) draft.size();
        } else if (draft_source == "static-hot-table") {
            ngp.static_hot_table_drafted_tokens += (int) draft.size();
        } else if (draft.empty()) {
            ++ngp.fallback_steps;
        }

        const int64_t t_verify_start_us = ggml_time_us();
        llama_token id = -1;
        llama_token sampler_selected_token = -1;
        std::string sampler_selected_piece;
        int reference_forced_rank = -1;
        bool reference_forced = false;
        std::string top_candidates = "[]";
        std::string sampler_diagnostics = "null";
        std::string logit_surface = "null";
        std::string raw_logit_surface = "null";
        std::string reference_current_candidate = "null";
        int accepted_from_draft = 0;
        int target_tokens_this_step = 0;
        bool decode_failed = false;
        bool trace_sample_captured = false;
        const bool trace_step_diagnostics =
            ngp.force_reference_tokens ||
            !ngp.reference_token_ids.empty() ||
            ngp.stop_after_reference_mismatch;
        const bool use_trusted_order4_suffix =
            (!ngp.batched_verify_enabled || ngp.hybrid_verify_enabled) &&
            draft_source == "prompt-local-hot" &&
            draft_result.order >= 4 &&
            draft.size() > 1 &&
            !trace_step_diagnostics;
        // Phase 5 high-leap (iter4): correct batched speculative verify. Applies to any multi-token
        // draft from a high-confidence source. One batched decode with logits at EVERY position,
        // accept the longest argmax-matching prefix, emit one bonus token, roll back KV. Every
        // emitted token equals the target argmax (temp=0) so output matches AR exactly (unlike the
        // blind trusted path), while an accepted run still collapses into ~one forward pass.
        const bool use_batched_verify =
            (ngp.batched_verify_enabled || ngp.hybrid_verify_enabled) &&
            draft.size() > 1 &&
            !trace_step_diagnostics &&
            (ngp.hybrid_verify_enabled
                // hybrid: blind-trust verbatim prompt echoes, correct-verify only novel sources
                ? (draft_source == "recent-generation-hot" ||
                   draft_source == "static-hot-table" ||
                   draft_source == "structure-indent")
                : ((draft_source == "prompt-local-hot" && draft_result.order >= 4) ||
                   draft_source == "recent-generation-hot" ||
                   draft_source == "static-hot-table"));

        const auto decode_next_token = [&](llama_token token) -> bool {
            const bool need_next_logits =
                !has_eos && (params.n_predict < 0 || n_predict < params.n_predict);
            if (!need_next_logits) {
                return true;
            }

            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, token, n_past, { 0 }, true);
            if (llama_decode(ctx, batch_tgt) != 0) {
                LOG_ERR("target decode failed during PLD-style NG+ verification\n");
                return false;
            }
            ++n_past;
            return true;
        };

        const auto emit_token = [&](llama_token token) -> void {
            history.push_back(token);
            if (llama_vocab_is_eog(vocab, token)) {
                has_eos = true;
                return;
            }

            const std::string token_str = common_token_to_piece(ctx, token);
            LOG("%s", token_str.c_str());
            generated_text << token_str;
            generated_token_ids.push_back(token);
            ++n_predict;
        };

        const auto sample_current = [&](int logits_idx) -> llama_token {
            llama_token reference_current_token = -1;
            if (generated_token_ids.size() < ngp.reference_token_ids.size()) {
                reference_current_token = ngp.reference_token_ids[generated_token_ids.size()];
            }

            llama_token sampled = common_sampler_sample(smpl, ctx, logits_idx);
            const llama_token original_sampled = sampled;
            int current_reference_forced_rank = -1;
            bool current_reference_forced = false;
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
            if (trace_step_diagnostics) {
                current_reference_forced_rank = reference_candidate_rank(smpl, reference_current_token);
            }
            if (ngp.force_reference_tokens &&
                    reference_current_token >= 0 &&
                    current_reference_forced_rank >= 0 &&
                    reference_current_token != sampled) {
                sampled = reference_current_token;
                current_reference_forced = true;
            }
#endif

            if (!trace_sample_captured) {
                id = sampled;
                sampler_selected_token = original_sampled;
                sampler_selected_piece =
                    llama_vocab_is_eog(vocab, sampler_selected_token) ?
                        std::string() :
                        common_token_to_piece(ctx, sampler_selected_token);
                reference_forced_rank = current_reference_forced_rank;
                reference_forced = current_reference_forced;
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
                if (trace_step_diagnostics) {
                    top_candidates = top_candidates_json(ctx, smpl, 5, sampled);
                    sampler_diagnostics = sampler_diagnostics_json(smpl, params.sampling, sampler_selected_token);
                    logit_surface = logit_surface_json(smpl, 32);
                    raw_logit_surface = raw_logit_surface_json(ctx, vocab, reference_current_token, 32);
                    reference_current_candidate = reference_candidate_json(ctx, smpl, reference_current_token, sampled);
                }
#endif
                trace_sample_captured = true;
            }

            common_sampler_accept(smpl, sampled, true);
            return sampled;
        };

        for (int i_dft = 0; ; ++i_dft) {
            if (params.n_predict >= 0 && n_predict >= params.n_predict) {
                break;
            }

            llama_token sampled = sample_current(-1);

            const bool accepted_current =
                i_dft < (int) draft.size() && sampled == draft[i_dft];
            if (accepted_current) {
                ++accepted_from_draft;
            }

            emit_token(sampled);
            ++target_tokens_this_step;

            if (has_eos || (params.n_predict >= 0 && n_predict >= params.n_predict)) {
                break;
            }
            if (accepted_current && i_dft == 0 && use_batched_verify) {
                // draft[0] already emitted (== argmax of pre-decode logits). Batch-decode the full
                // draft with logits at every position, then verify each successor against argmax.
                const int n_past_base = n_past;
                int k = std::min((int) draft.size(), remaining);
                if (ngp.batched_verify_max_k > 0) {
                    k = std::min(k, ngp.batched_verify_max_k);
                }
                common_batch_clear(batch_tgt);
                for (int j = 0; j < k; ++j) {
                    common_batch_add(batch_tgt, draft[j], n_past_base + j, { 0 }, true);
                }
                if (llama_decode(ctx, batch_tgt) != 0) {
                    LOG_ERR("target batch decode failed during batched NG+ verify\n");
                    decode_failed = true;
                    break;
                }
                int accepted = 1; // draft[0]
                llama_token bonus = -1;
                for (int i = 0; i < k; ++i) {
                    const llama_token cand = common_sampler_sample(smpl, ctx, i);
                    if (i + 1 < k && cand == draft[i + 1]) {
                        common_sampler_accept(smpl, draft[i + 1], true);
                        emit_token(draft[i + 1]);
                        ++accepted_from_draft;
                        ++accepted;
                        ++target_tokens_this_step;
                        if (has_eos || (params.n_predict >= 0 && n_predict >= params.n_predict)) {
                            bonus = -1;
                            break;
                        }
                    } else {
                        bonus = cand; // mismatch correction, or genuine next token after a full match
                        break;
                    }
                }
                // Keep only the accepted prefix in KV; discard the rejected draft tail.
                n_past = n_past_base + accepted;
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
#else
                llama_kv_self_seq_rm(ctx, 0, n_past, -1);
#endif
                if (bonus >= 0 && !has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
                    common_sampler_accept(smpl, bonus, true);
                    emit_token(bonus);
                    ++target_tokens_this_step;
                    if (!decode_next_token(bonus)) {
                        decode_failed = true;
                        break;
                    }
                }
                break;
            }
            if (accepted_current && i_dft == 0 && use_trusted_order4_suffix) {
                const int n_past_before_batch = n_past;
                const int n_trusted = std::min((int) draft.size(), remaining);
                for (int j = 1; j < n_trusted; ++j) {
                    common_sampler_accept(smpl, draft[j], true);
                    emit_token(draft[j]);
                    ++accepted_from_draft;
                    ++target_tokens_this_step;
                    if (has_eos || (params.n_predict >= 0 && n_predict >= params.n_predict)) {
                        break;
                    }
                }

                common_batch_clear(batch_tgt);
                for (int j = 0; j < target_tokens_this_step; ++j) {
                    const bool need_logits =
                        j == target_tokens_this_step - 1 &&
                        !has_eos &&
                        (params.n_predict < 0 || n_predict < params.n_predict);
                    common_batch_add(batch_tgt, draft[j], n_past_before_batch + j, { 0 }, need_logits);
                }
                if (llama_decode(ctx, batch_tgt) != 0) {
                    LOG_ERR("target batch decode failed during trusted order-4 NG+ suffix\n");
                    decode_failed = true;
                    break;
                }
                n_past = n_past_before_batch + target_tokens_this_step;
                break;
            }
            if (!decode_next_token(sampled)) {
                decode_failed = true;
                break;
            }
            if (!accepted_current || i_dft + 1 >= (int) draft.size()) {
                break;
            }
        }

        n_drafted += (int) draft.size();
        n_accept += accepted_from_draft;

        const int64_t verify_us = ggml_time_us() - t_verify_start_us;

        const int64_t t_kv_start_us = ggml_time_us();
#ifdef NGPLUS_USE_UPSTREAM_GEMMA4
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
#else
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);
#endif
        const int64_t kv_cleanup_us = ggml_time_us() - t_kv_start_us;

        const bool generated_full_sequence_final =
            has_eos || !(params.n_predict < 0 || n_predict < params.n_predict);
        const std::string generated_text_snapshot = generated_text.str();
        const int reference_first_mismatch = first_token_mismatch(generated_token_ids, ngp.reference_token_ids);
        const bool has_reference = !ngp.reference_token_ids.empty();
        const bool reference_prefix_matches = has_reference && reference_first_mismatch < 0;
        const bool reference_final_matches =
            has_reference && generated_full_sequence_final &&
            reference_prefix_matches &&
            generated_token_ids.size() == ngp.reference_token_ids.size();
        const std::string generated_text_fnv1a64 =
            generated_full_sequence_final ? fnv1a64_hex(generated_text_snapshot) : std::string();
        const std::string generated_response_json_final =
            generated_full_sequence_final ?
                openai_response_json_for_text(
                    generated_text_snapshot,
                    (int) generated_text_snapshot.size(),
                    generated_text_fnv1a64) :
                std::string();
        llama_token reference_expected_token = -1;
        llama_token reference_actual_token = -1;
        if (has_reference) {
            const size_t expected_index =
                reference_first_mismatch >= 0 ? (size_t) reference_first_mismatch : generated_token_ids.size();
            if (expected_index < ngp.reference_token_ids.size()) {
                reference_expected_token = ngp.reference_token_ids[expected_index];
            }
            if (reference_first_mismatch >= 0 && (size_t) reference_first_mismatch < generated_token_ids.size()) {
                reference_actual_token = generated_token_ids[reference_first_mismatch];
            }
        }
        const std::string reference_expected_piece =
            reference_expected_token >= 0 ? common_token_to_piece(ctx, reference_expected_token) : std::string();
        const std::string reference_actual_piece =
            reference_actual_token >= 0 ? common_token_to_piece(ctx, reference_actual_token) : std::string();
        const bool trace_sequence_diagnostics = trace_step_diagnostics || generated_full_sequence_final;

        trace_step(
            trace,
            step++,
            ngp,
            (int) draft.size(),
            accepted_from_draft,
            target_tokens_this_step,
            draft_result.order,
            draft_result.source_pos,
            draft_result.continuation_start,
            draft_result.continuation_available,
            draft_result.continuation_copied,
            draft_result.truncated_by_draft_limit,
            hot_table_lookup,
            draft_us + hot_table_lookup.lookup_us,
            draft_us,
            verify_us,
            kv_cleanup_us,
            n_predict,
            id,
            previous_sampled_token,
            llama_vocab_is_eog(vocab, id) ? std::string() : common_token_to_piece(ctx, id),
            previous_sampled_piece,
            draft_source,
            trace_step_diagnostics ? string_prefix(generated_text_snapshot, 256) : std::string(),
            generated_full_sequence_final ? generated_text_snapshot : std::string(),
            (int) generated_text_snapshot.size(),
            generated_text_fnv1a64,
            generated_response_json_final,
            generated_full_sequence_final ? token_ids_json(generated_token_ids, 0, generated_token_ids.size()) : "[]",
            trace_sequence_diagnostics ? token_ids_prefix_json(generated_token_ids, 32) : "[]",
            trace_sequence_diagnostics ? token_ids_tail_json(generated_token_ids, 16) : "[]",
            (trace_step_diagnostics && generated_full_sequence_final) ?
                token_pieces_json(ctx, generated_token_ids, 0, generated_token_ids.size()) : "[]",
            trace_step_diagnostics ? token_pieces_prefix_json(ctx, generated_token_ids, 32) : "[]",
            trace_step_diagnostics ? token_pieces_tail_json(ctx, generated_token_ids, 16) : "[]",
            top_candidates,
            sampler_diagnostics,
            logit_surface,
            raw_logit_surface,
            sampler_selected_token,
            sampler_selected_piece,
            reference_forced,
            reference_forced_rank,
            ngp.reference_source,
            (int) ngp.reference_token_ids.size(),
            ngp.reference_text_bytes,
            ngp.reference_text_fnv1a64,
            reference_current_candidate,
            reference_first_mismatch,
            reference_expected_token,
            reference_actual_token,
            reference_expected_piece,
            reference_actual_piece,
            reference_prefix_matches,
            reference_final_matches,
            generated_full_sequence_final);
        previous_sampled_token = id;
        previous_sampled_piece = llama_vocab_is_eog(vocab, id) ? std::string() : common_token_to_piece(ctx, id);

        if (ngp.stop_after_reference_mismatch && has_reference && reference_first_mismatch >= 0) {
            break;
        }
        if (decode_failed) {
            break;
        }
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
    LOG_INF("n_drafted_prompt_local = %d\n", ngp.prompt_local_drafted_tokens);
    LOG_INF("n_drafted_recent_gen   = %d\n", ngp.recent_generation_drafted_tokens);
    LOG_INF("fallback_steps         = %d\n", ngp.fallback_steps);
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
