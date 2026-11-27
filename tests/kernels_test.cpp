#include "model.h"
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr float tolerance = 1e-5f;
int failures = 0;

void expect_near(const char * test, std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: size %zu, expected %zu\n", test, actual.size(), expected.size());
        failures++;
        return;
    }
    for (size_t i = 0; i < actual.size(); i++) {
        // negated <= so NaN is reported, not passed
        if (!(std::fabs(actual[i] - expected[i]) <= tolerance)) {
            std::fprintf(stderr, "%s: [%zu] = %.7f, expected %.7f\n", test, i, actual[i], expected[i]);
            failures++;
        }
    }
}

void test_rmsnorm() {
    const std::vector<float> x{1, 2, 3, 4};
    std::vector<float> o(4);

    const std::vector<float> unit{1, 1, 1, 1};
    rmsnorm(o, x, unit);
    expect_near("rmsnorm unit weights", o, std::vector<float>{0.3651481f, 0.7302963f, 1.0954444f, 1.4605925f});

    const std::vector<float> weights{0.5f, 1, 2, 0};
    rmsnorm(o, x, weights);
    expect_near("rmsnorm scaled weights", o, std::vector<float>{0.1825741f, 0.7302963f, 2.1908888f, 0.0f});

    // forward() runs the final norm in place on state.x
    std::vector<float> in_place = x;
    rmsnorm(in_place, in_place, weights);
    expect_near("rmsnorm in place", in_place, std::vector<float>{0.1825741f, 0.7302963f, 2.1908888f, 0.0f});
}

void test_softmax() {
    std::vector<float> x{1, 2, 3};
    softmax(x);
    expect_near("softmax", x, std::vector<float>{0.0900306f, 0.2447285f, 0.6652410f});

    // without max subtraction exp(1000) is inf and the result is NaN
    std::vector<float> large{1000, 1001};
    softmax(large);
    expect_near("softmax large inputs", large, std::vector<float>{0.2689414f, 0.7310586f});

    // attention normalizes only the first pos+1 scores, the rest must stay untouched
    std::vector<float> row{1, 2, 3, 99};
    softmax(std::span(row).first(3));
    expect_near("softmax prefix", row, std::vector<float>{0.0900306f, 0.2447285f, 0.6652410f, 99.0f});
}

void test_matmul() {
    // non-square so swapped n/d or a transposed read fails
    const std::vector<float> w{1, 2, 3,
                               4, 5, 6};
    const std::vector<float> x{1, 0, -1};
    std::vector<float> o{42, 42};
    matmul(o, x, w, 3, 2);
    expect_near("matmul", o, std::vector<float>{-2, -2});

    const std::vector<float> y{1, 1, 1};
    matmul(o, y, w, 3, 2);
    expect_near("matmul overwrites output", o, std::vector<float>{6, 15});
}

void test_rope() {
    constexpr int head_dim = 4;
    const float c1 = std::cos(1.0f), s1 = std::sin(1.0f);
    // second pair of each head uses freq 10000^(-2/4) = 0.01
    const float c2 = std::cos(0.01f), s2 = std::sin(0.01f);

    std::vector<float> q{1, 0, 1, 0, 1, 0, 1, 0};
    std::vector<float> k{1, 0, 1, 0};
    rope(q, k, 0, head_dim);
    expect_near("rope pos 0 is identity q", q, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0});
    expect_near("rope pos 0 is identity k", k, std::vector<float>{1, 0, 1, 0});

    // GQA: two query heads share one kv head, so k is half as wide as q
    rope(q, k, 1, head_dim);
    expect_near("rope q", q, std::vector<float>{c1, s1, c2, s2, c1, s1, c2, s2});
    expect_near("rope k", k, std::vector<float>{c1, s1, c2, s2});

    // rotation keeps each pair's length: (3, 4) has norm 5
    std::vector<float> q2{3, 4, 0, 0};
    std::vector<float> k2{0, 0, 0, 0};
    rope(q2, k2, 7, head_dim);
    const float norm = std::hypot(q2[0], q2[1]);
    expect_near("rope preserves norm", std::span(&norm, 1), std::vector<float>{5});
}

void test_swiglu() {
    std::vector<float> gate{0, 1, -1};
    const std::vector<float> up{2, 3, 4};
    swiglu(gate, up);
    expect_near("swiglu", gate, std::vector<float>{0.0f, 2.1931757f, -1.0757657f});
}

}  // namespace

int main() {
    test_rmsnorm();
    test_softmax();
    test_matmul();
    test_rope();
    test_swiglu();

    std::puts(failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
