#pragma once

#include <string>
#include <vector>
#include <span>

// field order matches the llama2.c checkpoint header
struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads; // < n_heads under GQA
    int vocab_size;
    int seq_len;
};

struct LayerCache {
    std::vector<float> key;   // (seq_len, kv_dim)
    std::vector<float> value; // (seq_len, kv_dim)
};

struct RunState {
    std::vector<float>      x;        // (dim)
    std::vector<float>      xb;       // (dim)
    std::vector<float>      xb2;      // (dim)
    std::vector<float>      hb;       // (hidden_dim) GLU gate
    std::vector<float>      hb2;      // (hidden_dim) GLU up
    std::vector<float>      q;        // (n_heads * head_dim)
    std::vector<float>      k;        // (n_kv_heads * head_dim)
    std::vector<float>      v;        // (n_kv_heads * head_dim)
    std::vector<float>      att;      // (n_heads, seq_len)
    std::vector<float>      logits;   // (vocab_size)
    std::vector<LayerCache> kv_cache; // (n_layers)

    explicit RunState(const Config & c);
};

struct Block {
    Block(
        int layer_i,
        // why do they need to be consts? when its constructed, are rvalues pased in?
        // shouldnt it be recieving direct mappings or sum
        std::span<const float> embedding_table,       // (vocab_size, dim)
        std::span<const float> rms_attention_weights, // (dim)
        std::span<const float> rms_ffn_weights,       // (dim)
        std::span<const float> wq,                    // (dim, n_heads * head_dim)
        std::span<const float> wk,                    // (dim, n_kv_heads * head_dim)
        std::span<const float> wv,                    // (dim, n_kv_heads * head_dim)
        std::span<const float> wo,                    // (n_heads * head_dim, dim)
        std::span<const float> w1,                    // (dim, hidden_dim)
        std::span<const float> w2,                    // (hidden_dim, dim)
        std::span<const float> w3                     // (dim, hidden_dim)
    );
    ~Block();

    int                    _layer_i;
    std::span<const float> _embedding_table;
    std::span<const float> _rms_attention_weights;
    std::span<const float> _rms_ffn_weights;
    std::span<const float> _wq;
    std::span<const float> _wk;
    std::span<const float> _wv;
    std::span<const float> _wo;
    std::span<const float> _w1;
    std::span<const float> _w2;
    std::span<const float> _w3;
};

struct Model {
    Config                 config;
    std::vector<Block>     blocks;
    std::span<const float> token_embedding_table; // (vocab_size, dim)
    std::span<const float> rms_final_weight;      // (dim)
    std::span<const float> wcls;                  // (vocab_size, dim), may alias token_embedding_table
    int                    fd;
    float *                data;                  // mmap of the checkpoint
    size_t                 size;

    explicit Model(const char * checkpoint);
};

void rmsnorm(std::vector<float> & o, const std::vector<float> & x, std::span<const float> weights);
void softmax(std::span<float> x);
void matmul (std::vector<float> & o, const std::vector<float> & x, std::span<const float> w, int n, int d);
void rope   (std::vector<float> & q, std::vector<float> & k, int pos, int head_dim);
void swiglu (std::vector<float> & gate, const std::vector<float> & up);

std::vector<float> & forward(const Model & model, RunState & state, int token, int pos);
