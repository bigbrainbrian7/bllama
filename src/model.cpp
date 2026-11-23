#include "model.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cmath>
#include <numbers>
#include <algorithm>

RunState::RunState(const Config& c)
    : x(static_cast<size_t>(c.dim)),
        xb(static_cast<size_t>(c.dim)),
        xb2(static_cast<size_t>(c.dim)),
        hb(static_cast<size_t>(c.hidden_dim)),
        hb2(static_cast<size_t>(c.hidden_dim)),
        q(static_cast<size_t>(c.n_heads) * (c.dim / c.n_heads)),
        k(static_cast<size_t>(c.n_kv_heads) * (c.dim / c.n_heads)),
        v(static_cast<size_t>(c.n_kv_heads) * (c.dim / c.n_heads)),
        att(static_cast<size_t>(c.n_heads) * c.seq_len),
        logits(static_cast<size_t>(c.vocab_size)),
        kv_cache(static_cast<size_t>(c.n_layers),
                 LayerCache{
                     std::vector<float>(static_cast<size_t>(c.seq_len) * c.n_kv_heads * (c.dim / c.n_heads)),
                     std::vector<float>(static_cast<size_t>(c.seq_len) * c.n_kv_heads * (c.dim / c.n_heads))})
{}

Block::Block(
    int layer_i,
    std::span<const float> embedding_table, //(vocab_size, dim)
    std::span<const float> rms_attention_weights, //(dim,)
    std::span<const float> rms_ffn_weights, //(dim, )
    std::span<const float> wq, //(dim, n_heads*head_dim)
    std::span<const float> wk, //(dim, n_kv_heads*head_dim)
    std::span<const float> wv, //(dim, n_kv_heads*head_dim)
    std::span<const float> wo, //(n_heads*head_dim, dim)
    std::span<const float> w1, //(dim, hidden_dim)
    std::span<const float> w2, //(hidden_dim, dim)
    std::span<const float> w3 //(dim, hidden_dim) glu path
)
    : _layer_i(layer_i),
      _embedding_table(embedding_table),
      _rms_attention_weights(rms_attention_weights),
      _rms_ffn_weights(rms_ffn_weights),
      _wq(wq),
      _wk(wk),
      _wv(wv),
      _wo(wo),
      _w1(w1),
      _w2(w2),
      _w3(w3)
{}

Block::~Block() = default;

// technically not just weights, but just preallocates and maps where necessary
static void memory_map_weights(Model& model, bool shared_weights) {
    const Config& config = model.config;
    // size_t keeps the offset arithmetic 64-bit for 13B+ parameter counts.
    const size_t dim = config.dim;
    const size_t hidden_dim = config.hidden_dim;
    const size_t n_layers = config.n_layers;
    const size_t vocab_size = config.vocab_size;
    const size_t seq_len = config.seq_len;
    const size_t head_dim = dim / config.n_heads;
    const size_t q_dim = config.n_heads * head_dim;
    const size_t kv_dim = config.n_kv_heads * head_dim;

    const size_t rms_size = dim;
    const size_t wq_size = dim * q_dim;
    const size_t wk_size = dim * kv_dim;
    const size_t wv_size = dim * kv_dim;
    const size_t wo_size = q_dim * dim;
    const size_t w1_size = dim * hidden_dim;
    const size_t w2_size = hidden_dim * dim;
    const size_t w3_size = dim * hidden_dim;

    const float *ptr = model.data + sizeof(Config) / sizeof(float);
    const float *embedding_table = ptr; ptr += vocab_size * dim;
    const float *rms_att_base    = ptr; ptr += n_layers * rms_size;
    const float *wq_base         = ptr; ptr += n_layers * wq_size;
    const float *wk_base         = ptr; ptr += n_layers * wk_size;
    const float *wv_base         = ptr; ptr += n_layers * wv_size;
    const float *wo_base         = ptr; ptr += n_layers * wo_size;
    const float *rms_ffn_base    = ptr; ptr += n_layers * rms_size;
    const float *w1_base         = ptr; ptr += n_layers * w1_size;
    const float *w2_base         = ptr; ptr += n_layers * w2_size;
    const float *w3_base         = ptr; ptr += n_layers * w3_size;
    const float *rms_final       = ptr; ptr += dim;
    // Legacy freq_cis_real and freq_cis_imag RoPE tables, still present in the file format.
    ptr += seq_len * head_dim;
    const float *wcls = shared_weights ? embedding_table : ptr;

    model.token_embedding_table = {embedding_table, vocab_size * dim};
    model.rms_final_weight = {rms_final, dim};
    model.wcls = {wcls, vocab_size * dim};

    model.blocks.clear();
    model.blocks.reserve(n_layers);
    for (size_t i = 0; i < n_layers; i++) {
        model.blocks.emplace_back(
            static_cast<int>(i),
            model.token_embedding_table,
            std::span<const float>(rms_att_base + i * rms_size, rms_size),
            std::span<const float>(rms_ffn_base + i * rms_size, rms_size),
            std::span<const float>(wq_base + i * wq_size, wq_size),
            std::span<const float>(wk_base + i * wk_size, wk_size),
            std::span<const float>(wv_base + i * wv_size, wv_size),
            std::span<const float>(wo_base + i * wo_size, wo_size),
            std::span<const float>(w1_base + i * w1_size, w1_size),
            std::span<const float>(w2_base + i * w2_size, w2_size),
            std::span<const float>(w3_base + i * w3_size, w3_size)
        );
    }
}

//will initially be string literal. later will select
void build_model(const char *checkpoint, Model& model) {
    FILE *file = std::fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        // does it need to exit and cleanup properly?
        std::exit(EXIT_FAILURE);
    }

    //reads directly from model metadata binary data into config object
    // since we testing out with llama.c model, we will have to stay consistent here
    if(fread(&model.config, sizeof(Config), 1, file)!=1) {
        fprintf(stderr, "Could not load into config");
        std::exit(EXIT_FAILURE);
    }

    // llama2.c encodes an unshared classifier (wcls != embedding table) as a negative vocab_size.
    bool shared_weights = model.config.vocab_size > 0;
    model.config.vocab_size = abs(model.config.vocab_size);

    std::fseek(file, 0, SEEK_END);
    model.size = std::ftell(file);
    std::fclose(file);

    model.fd = open(checkpoint, O_RDONLY);
    if (model.fd == -1) {
        fprintf(stderr, "open failed for %s\n", checkpoint);
        std::exit(EXIT_FAILURE);
    }

    model.data = static_cast<float *>(mmap(nullptr, model.size, PROT_READ, MAP_PRIVATE, model.fd, 0));
    if (model.data == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", checkpoint);
        std::exit(EXIT_FAILURE);
    }

    memory_map_weights(model, shared_weights);
}

//neural network blocks
void rmsnorm(std::vector<float>& o, const std::vector<float>& x, std::span<const float> weights) {
    float rms = 0.0f;
    for (const auto& val: x) {
        rms += val * val;
    }
    rms /= static_cast<float>(x.size());
    rms += 1e-5f;
    rms = 1.0f / std::sqrt(rms);

    for (size_t i=0; i<o.size(); i++) {
        o[i] = x[i] * weights[i] * rms;
    }
}

// Takes a span so attention can normalize just the first pos+1 scores of a head's row.
void softmax(std::span<float> x) {
    const float max = std::ranges::max(x);
    float denom = 0.0f;

    for (auto& val: x) {
        val = std::exp(val - max);
        denom += val;
    }

    const float inv_denom = 1.0f / denom;
    for (auto& val: x) {
        val *= inv_denom;
    }
}

/**
 * GEMV
 * @param w (d, n)
 * @param x (n, )
 * @param o (d, )
 * @param n collapsed dim
 */
void matmul(std::vector<float>& o, const std::vector<float>& x, std::span<const float> w, int n, int d) {
    for (int i=0; i<d; i++) {
        float sum = 0.0f;
        for (int j=0; j<n; j++) {
            sum += w[static_cast<size_t>(i)*n+j] * x[j];
        }

        o[i] = sum;
    }
}

void rope(std::vector<float>& q, std::vector<float>& k, int pos, int head_dim) {
    const int q_dim = static_cast<int>(q.size());
    const int kv_dim = static_cast<int>(k.size());
    for (int i = 0; i < q_dim; i += 2) {
        const int head_pos = i % head_dim;
        const float freq = 1.0f / std::pow(10000.0f, head_pos / static_cast<float>(head_dim));
        const float theta = pos * freq;
        const float fcr = std::cos(theta);
        const float fci = std::sin(theta);
        // K is only kv_dim wide under GQA, so past that point only Q gets rotated.
        const int rotn = i < kv_dim ? 2 : 1;
        for (int t = 0; t < rotn; t++) {
            std::vector<float>& vec = t == 0 ? q : k;
            const float v0 = vec[i];
            const float v1 = vec[i+1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

void swiglu(std::vector<float>& gate, const std::vector<float>& up) {
    for (size_t i = 0; i < gate.size(); i++) {
        const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        gate[i] = silu * up[i];
    }
}

std::vector<float>& forward(const Model& model, RunState& state, int token, int pos) {
    const Config& config = model.config;
    const int dim = config.dim;
    const int head_dim = dim / config.n_heads;
    const int kv_dim = config.n_kv_heads * head_dim;
    // GQA: each K/V head is shared by kv_mul consecutive query heads.
    const int kv_mul = config.n_heads / config.n_kv_heads;
    const float inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto content_row = model.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim);
    std::ranges::copy(content_row, state.x.begin());

    for (const auto& block: model.blocks) {
        LayerCache& cache = state.kv_cache[static_cast<size_t>(block._layer_i)];
        rmsnorm(state.xb, state.x, block._rms_attention_weights);

        matmul(state.q, state.xb, block._wq, dim, dim);
        matmul(state.k, state.xb, block._wk, dim, kv_dim);
        matmul(state.v, state.xb, block._wv, dim, kv_dim);

        rope(state.q, state.k, pos, head_dim);

        const auto cache_offset = static_cast<size_t>(pos) * kv_dim;
        std::ranges::copy(state.k, cache.key.begin() + cache_offset);
        std::ranges::copy(state.v, cache.value.begin() + cache_offset);

        std::ranges::fill(state.xb, 0.0f);
        for (int h=0; h<config.n_heads; h++) {
            const int q_i = h*head_dim;
            const int a_i = h*config.seq_len;
            const int kv_head_offset = (h / kv_mul) * head_dim;

            for (int t=0; t<=pos; t++) {
                const size_t kv_i = static_cast<size_t>(t) * kv_dim + kv_head_offset;
                float score = 0.0f;

                for (int i=0; i<head_dim; i++) {
                    score += state.q[q_i+i] * cache.key[kv_i+i];
                }

                state.att[a_i+t] = score * inv_sqrt_head_dim;
            }

            softmax(std::span(state.att).subspan(static_cast<size_t>(a_i), static_cast<size_t>(pos) + 1));

            for (int t=0; t<=pos; t++) {
                const size_t kv_i = static_cast<size_t>(t) * kv_dim + kv_head_offset;
                const float a = state.att[a_i+t];

                for (int i=0; i<head_dim; i++) {
                    state.xb[q_i + i] += a * cache.value[kv_i + i];
                }
            }
        }

        matmul(state.xb2, state.xb, block._wo, dim, dim);
        for (int i=0; i<dim; i++) {
            state.x[i] += state.xb2[i];
        }

        rmsnorm(state.xb, state.x, block._rms_ffn_weights);
        matmul(state.hb, state.xb, block._w1, dim, config.hidden_dim);
        matmul(state.hb2, state.xb, block._w3, dim, config.hidden_dim);

        swiglu(state.hb, state.hb2);

        matmul(state.xb, state.hb, block._w2, config.hidden_dim, dim);
        for (int i=0; i<dim; i++) {
            state.x[i] += state.xb[i];
        }
    }

    rmsnorm(state.x, state.x, model.rms_final_weight);
    matmul(state.logits, state.x, model.wcls, dim, config.vocab_size);
    return state.logits;
}
