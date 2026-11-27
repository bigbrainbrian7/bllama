#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct Sampler {
    int             vocab_size;
    float           temperature;
    std::mt19937_64 rng;

    Sampler(int vocab_size, float temperature, uint64_t seed);

    int sample_argmax(std::vector<float> & probs) const;

    // temperature 0 is greedy, otherwise logits are overwritten with probabilities
    int sample(std::vector<float> & logits);
};
