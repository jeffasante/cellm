#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t cellm_engine_t;
typedef uint64_t cellm_session_t;
typedef uint64_t cellm_tokenizer_t;

typedef enum cellm_backend_kind_t {
    CELLM_BACKEND_CPU = 0,
    CELLM_BACKEND_METAL = 1,
} cellm_backend_kind_t;

// Error handling (thread-local).
// Returns number of bytes written (excluding null terminator).
size_t cellm_last_error_message(char* out_buf, size_t buf_len);

// Engine lifecycle.
cellm_engine_t cellm_engine_create(
    const char* model_path,
    uint32_t tokens_per_block,
    uint32_t total_blocks,
    uint32_t top_k
);

// Engine lifecycle (v2 sampling controls).
// - temperature: 0 => greedy
// - repeat_penalty: 1 => disabled
// - seed: 0 => uses an internal default
cellm_engine_t cellm_engine_create_v2(
    const char* model_path,
    uint32_t tokens_per_block,
    uint32_t total_blocks,
    uint32_t top_k,
    float temperature,
    float repeat_penalty,
    uint32_t repeat_window,
    uint64_t seed
);

// Engine lifecycle (v3) with backend selector.
cellm_engine_t cellm_engine_create_v3(
    const char* model_path,
    uint32_t tokens_per_block,
    uint32_t total_blocks,
    uint32_t top_k,
    float temperature,
    float repeat_penalty,
    uint32_t repeat_window,
    uint64_t seed,
    uint32_t backend // cellm_backend_kind_t
);

void cellm_engine_destroy(cellm_engine_t engine);

// Returns active backend name ("cpu" or "metal"), bytes written excluding null terminator.
size_t cellm_engine_backend_name(
    cellm_engine_t engine,
    char* out_buf,
    size_t buf_len
);

// Tokenizers.
cellm_tokenizer_t cellm_tokenizer_create(const char* tokenizer_path);
void cellm_tokenizer_destroy(cellm_tokenizer_t tok);

// If out_tokens is null/max_tokens==0, returns required token count.
size_t cellm_tokenizer_encode(
    cellm_tokenizer_t tok,
    const char* text_utf8,
    uint32_t* out_tokens,
    size_t max_tokens
);

// If out_buf is null/buf_len==0, returns required byte count (excluding null terminator).
size_t cellm_tokenizer_decode(
    cellm_tokenizer_t tok,
    const uint32_t* tokens,
    size_t token_count,
    char* out_buf,
    size_t buf_len
);

// Sessions.
cellm_session_t cellm_session_create(cellm_engine_t engine);
int32_t cellm_session_cancel(cellm_engine_t engine, cellm_session_t session);

// Prompt/turn input (token ids only, for now).
int32_t cellm_submit_tokens(
    cellm_engine_t engine,
    cellm_session_t session,
    const uint32_t* tokens,
    size_t token_count,
    uint32_t* out_next_token
);

// One decode step for the next scheduled session.
// Returns:
//   1 -> produced a token (out_session/out_token set)
//   0 -> nothing to do
//  -1 -> error (see cellm_last_error_message)
int32_t cellm_step_decode(
    cellm_engine_t engine,
    cellm_session_t* out_session,
    uint32_t* out_token
);

// Paged KV cache stats.
int32_t cellm_engine_kv_stats(
    cellm_engine_t engine,
    uint32_t* out_used_blocks,
    uint32_t* out_free_blocks
);

// VLM image description path.
// - `image_bytes` should be encoded image data (jpeg/png).
// - `prompt_utf8` is the user prompt/question.
// - If `out_buf` is non-null and `buf_len > 0`, generated text is written and null-terminated.
int32_t cellm_vlm_describe_image(
    cellm_engine_t engine,
    cellm_session_t session,
    const uint8_t* image_bytes,
    size_t image_len,
    const char* prompt_utf8,
    char* out_buf,
    size_t buf_len
);

// Last VLM timing breakdown in milliseconds for current thread.
// Returns 0 on success, -1 on error (see cellm_last_error_message).
int32_t cellm_vlm_last_timings_ms(
    double* out_patch_ms,
    double* out_encoder_ms,
    double* out_decode_ms,
    double* out_total_ms
);

// Per-layer encoder timing for last VLM run in current thread.
uint32_t cellm_vlm_last_encoder_layer_count(void);
int32_t cellm_vlm_last_encoder_layer_time_ms(uint32_t layer_index, double* out_ms);

#ifdef __cplusplus
} // extern "C"
#endif
