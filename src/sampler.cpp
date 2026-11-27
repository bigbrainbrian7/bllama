#include "sampler.h"
#include "model.h"

Sampler::Sampler(int vocab_size, float temperature, uint64_t seed)
    : vocab_size(vocab_size), temperature(temperature), rng(seed) {}

int Sampler::sample_argmax(std::vector<float> & probs) const {
    int max_i = 0;
    float max = probs[0];
    for (std::size_t i = 0; i < probs.size(); i++) {
        if (probs[i] > max) {
            max   = probs[i];
            max_i = i;
        }
    }

    return max_i;
}

int Sampler::sample(std::vector<float> & logits) {
    if (temperature == 0.0f) {
        return sample_argmax(logits);
    }

    for (auto & l : logits) {
        l /= temperature;
    }
    softmax(logits);

    const float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float cdf = 0.0f;
    for (std::size_t i = 0; i < logits.size(); i++) {
        cdf += logits[i];
        if (r < cdf) {
            return i;
        }
    }
    return logits.size() - 1; // float rounding can leave cdf just under r
}
