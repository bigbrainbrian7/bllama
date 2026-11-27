#include "sampler.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char * test, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "%s: %s\n", test, what);
        failures++;
    }
}

void test_constructor() {
    const Sampler sampler(32000, 0.8f, 5);
    expect(sampler.vocab_size == 32000, "constructor", "vocab_size not stored");
    expect(sampler.temperature == 0.8f, "constructor", "temperature not stored");
}

void test_argmax() {
    const Sampler sampler(4, 0.0f, 0);
    std::vector<float> v{0.1f, 3.0f, -2.0f, 1.0f};
    expect(sampler.sample_argmax(v) == 1, "argmax", "picked the wrong index");

    std::vector<float> first{5.0f, 1.0f, 2.0f};
    expect(sampler.sample_argmax(first) == 0, "argmax first", "missed a max at index 0");

    std::vector<float> last{-3.0f, -2.0f, -1.0f};
    expect(sampler.sample_argmax(last) == 2, "argmax last", "missed a max at the last index");

    std::vector<float> ties{1.0f, 7.0f, 7.0f};
    expect(sampler.sample_argmax(ties) == 1, "argmax ties", "should keep the first of equal maxima");
}

void test_temperature_zero_is_greedy() {
    Sampler sampler(4, 0.0f, 123);
    const std::vector<float> logits{0.5f, -1.0f, 2.5f, 2.0f};
    for (int i = 0; i < 100; i++) {
        std::vector<float> v = logits;
        if (sampler.sample(v) != 2 || v != logits) {
            expect(false, "temperature 0", "should return the argmax and leave logits untouched");
            return;
        }
    }
}

std::vector<int> draw(uint64_t seed, float temperature, const std::vector<float> & logits, int n) {
    Sampler sampler(static_cast<int>(logits.size()), temperature, seed);
    std::vector<int> out;
    for (int i = 0; i < n; i++) {
        std::vector<float> v = logits;
        out.push_back(sampler.sample(v));
    }
    return out;
}

void test_seeding() {
    const std::vector<float> logits(8, 0.0f);
    expect(draw(42, 1.0f, logits, 200) == draw(42, 1.0f, logits, 200), "seed", "same seed gave different draws");
    expect(draw(42, 1.0f, logits, 200) != draw(43, 1.0f, logits, 200), "seed", "different seeds gave identical draws");
}

void test_logits_become_probabilities() {
    Sampler sampler(3, 0.7f, 1);
    std::vector<float> v{1.0f, 2.0f, 3.0f};
    const int token = sampler.sample(v);
    expect(token >= 0 && token < 3, "probabilities", "token out of range");

    const float e0 = std::exp(1.0f / 0.7f), e1 = std::exp(2.0f / 0.7f), e2 = std::exp(3.0f / 0.7f);
    const float z = e0 + e1 + e2;
    expect(std::fabs(v[0] - e0 / z) < 1e-6f && std::fabs(v[1] - e1 / z) < 1e-6f && std::fabs(v[2] - e2 / z) < 1e-6f,
           "probabilities", "logits not replaced by softmax(logits / temperature)");
}

// each count should land within 5 standard deviations of n * softmax(logits / T)
void check_distribution(const char * test, float temperature, const std::vector<float> & logits) {
    constexpr int n = 200000;
    std::vector<int> counts(logits.size(), 0);
    for (const int t : draw(7, temperature, logits, n)) {
        counts[static_cast<size_t>(t)]++;
    }

    std::vector<double> p(logits.size());
    double z = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        p[i] = std::exp(static_cast<double>(logits[i]) / temperature);
        z += p[i];
    }
    for (size_t i = 0; i < logits.size(); i++) {
        p[i] /= z;
        const double expected = n * p[i];
        const double tol = 5.0 * std::sqrt(n * p[i] * (1.0 - p[i])) + 1.0;
        if (std::fabs(counts[i] - expected) > tol) {
            std::fprintf(stderr, "%s: token %zu drawn %d times, expected %.0f +- %.0f\n", test, i, counts[i],
                         expected, tol);
            failures++;
        }
    }
}

void test_distribution() {
    const std::vector<float> logits{1.0f, 2.0f, 0.0f, -1.0f, 1.5f};
    check_distribution("distribution T=1", 1.0f, logits);
    check_distribution("distribution T=0.5", 0.5f, logits);
    check_distribution("distribution T=2", 2.0f, logits);
}

void test_zero_probability_never_drawn() {
    constexpr float ninf = -std::numeric_limits<float>::infinity();
    const std::vector<float> logits{ninf, 0.0f, ninf, 0.0f, ninf};
    for (const int t : draw(9, 1.0f, logits, 10000)) {
        if (t != 1 && t != 3) {
            expect(false, "zero probability", "drew a token with probability 0");
            return;
        }
    }
}

void test_vocab_sized_input() {
    std::vector<float> logits(32000, 0.0f);
    logits[12345] = 30.0f; // ~e^30 / 32000 times likelier than all others combined
    for (const int t : draw(11, 1.0f, logits, 100)) {
        if (t != 12345) {
            expect(false, "vocab sized", "missed an overwhelmingly likely token");
            return;
        }
    }
}

}  // namespace

int main() {
    test_constructor();
    test_argmax();
    test_temperature_zero_is_greedy();
    test_seeding();
    test_logits_become_probabilities();
    test_distribution();
    test_zero_probability_never_drawn();
    test_vocab_sized_input();

    std::puts(failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
