#pragma once

#include <string>
#include <vector>
#include <span>

// stay consistent with llama.c impl
// struct Config {
//     int dim; //input and output dimensions
//     int hidden_dim; //ffn dimensions
//     int head_dim; //dimension of attention head (dim / n_heads, but js being explicit)
//     int n_layers; //num of layers
//     int n_heads; //number of query heads
//     int n_kv_heads; //number of kv heads (GQA)
//     int vocab_size; //vocab size
//     int seq_len; //max sequence length
// };

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct LayerCache {
    std::vector<float> key; //(seq_len, kv_dim)
    std::vector<float> value; //(seq_len, kv_dim)
};

struct RunState{
    std::vector<float> x; //latest activation (dim, )
    std::vector<float> xb; //residual activation (dim, )
    std::vector<float> xb2; //attention output projection (dim, )
    std::vector<float> hb; //GLU slot 1 (hidden_dim, ), passed through activation function
    std::vector<float> hb2;//GLU slot 2 (hidden_dim, )
    std::vector<float> q; //query vector (head_dim * n_heads)
    std::vector<float> k; //key vector (head_dim * n_kv_heads)
    std::vector<float> v; //value vector (head_dim * n_kv_heads)
    std::vector<float> att; //attention scores (n_heads, seq_len)
    std::vector<float> logits; //(vocab_size, )
    std::vector<LayerCache> kv_cache; //(n_layers, )

    explicit RunState(const Config& c);
};

struct Block{
    Block(
        int layer_i,
        // why do they need to be consts? when its constructed, are rvalues pased in? 
        // shouldnt it be recieving direct mappings or sum
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
    );
    ~Block();

    int _layer_i;
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

struct Model{
    Config config;
    std::vector<Block> blocks;
    std::span<const float> token_embedding_table; //(vocab_size, dim)
    std::span<const float> rms_final_weight; //(dim, )
    std::span<const float> wcls; //(vocab_size, dim) classifier, may alias token_embedding_table
    int fd; //file descriptor for mmaping
    float *data; //memory mapped data pointer
    size_t size; //file size of model checkpoint
};

void build_model(const char *checkpoint, Model& model);

void rmsnorm(std::vector<float>& o, const std::vector<float>& x, std::span<const float> weights);
void softmax(std::span<float> x);
void matmul(std::vector<float>& o, const std::vector<float>& x, std::span<const float> w, int n, int d);
void rope(std::vector<float>& q, std::vector<float>& k, int pos, int head_dim);
void swiglu(std::vector<float>& gate, const std::vector<float>& up);

std::vector<float>& forward(const Model& model, RunState& state, int token, int pos);