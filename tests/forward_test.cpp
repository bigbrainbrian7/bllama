#include "model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace {

struct Step {
    int input_token;
    int expected_token;
    float expected_max_logit;
    float expected_logit_0;
    float expected_logit_1;
};

// from an independent float64 NumPy llama2.c implementation, so a bug shared with this code cannot hide
// decodes "Once upon a time, there was a little girl named Lily. She loved"
constexpr std::array<Step, 16> steps{{
    {1, 9038, 12.290031f, -6.790780f, 0.828117f},
    {9038, 2501, 25.372501f, -2.921234f, 1.508056f},
    {2501, 263, 20.238472f, -10.223969f, 0.960062f},
    {263, 931, 20.871307f, -4.997939f, 0.659605f},
    {931, 29892, 18.359753f, -10.420858f, 0.461925f},
    {29892, 727, 16.633376f, -9.885887f, -1.374925f},
    {727, 471, 13.205778f, -8.455117f, -4.046688f},
    {471, 263, 13.860286f, -14.491254f, -4.536631f},
    {263, 2217, 9.799217f, -17.457565f, -1.592137f},
    {2217, 7826, 10.761473f, -17.325998f, -5.445385f},
    {7826, 4257, 20.295226f, -11.715060f, -1.263996f},
    {4257, 365, 13.459838f, -7.660067f, -2.903360f},
    {365, 2354, 22.684059f, -2.694567f, 0.203714f},
    {2354, 29889, 22.145936f, -11.005780f, 2.108231f},
    {29889, 2296, 20.304336f, -10.387771f, 4.460352f},
    {2296, 18012, 21.613805f, -6.522605f, 1.247606f},
}};

// float32 drifts ~1e-4 from the float64 reference over 16 steps
constexpr float tolerance = 1e-3f;

bool check_logit(int pos, const char * name, float actual, float expected) {
    if (std::fabs(actual - expected) <= tolerance) {
        return true;
    }
    std::fprintf(stderr, "pos %d: %s = %f, expected %f\n", pos, name, actual, expected);
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * checkpoint = argc > 1 ? argv[1] : "./models/stories15M.bin";
    Model model(checkpoint);
    RunState state(model.config);

    bool ok = true;
    int token = steps.front().input_token;
    for (int pos = 0; pos < static_cast<int>(steps.size()); pos++) {
        const Step & step = steps[static_cast<size_t>(pos)];
        if (token != step.input_token) {
            std::fprintf(stderr, "pos %d: fed token %d, expected %d\n", pos, token, step.input_token);
            return 1;
        }

        const std::vector<float> & logits = forward(model, state, token, pos);
        const auto max_it = std::ranges::max_element(logits);
        const int next_token = static_cast<int>(std::distance(logits.begin(), max_it));

        if (next_token != step.expected_token) {
            std::fprintf(stderr, "pos %d: argmax %d, expected %d\n", pos, next_token, step.expected_token);
            ok = false;
        }
        ok &= check_logit(pos, "max logit", *max_it, step.expected_max_logit);
        ok &= check_logit(pos, "logit[0]", logits[0], step.expected_logit_0);
        ok &= check_logit(pos, "logit[1]", logits[1], step.expected_logit_1);

        // feed the expected token so one mismatch does not cascade
        token = step.expected_token;
    }

    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
