#include "model.h"
#include <iostream>

int main() {
    Model model;
    build_model("./models/stories15M.bin", model);
    RunState state(model.config);

    constexpr int bos_token = 1;
    const std::vector<float>& logits = forward(model, state, bos_token, 0);

    for (size_t i = 0; i < logits.size(); i++) {
        std::cout << i << ": " << logits[i] << '\n';
    }
}
