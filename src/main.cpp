#include "model.h"
#include "tokenizer.h"
#include <iostream>

int sample_argmax(std::vector<float>& probs){
    int max_i = 0;
    float max = probs[0];
    for (std::size_t i=0; i<probs.size(); i++) {
        if (probs[i] > max) {
            max = probs[i];
            max_i = i;
        }
    }

    return max_i;
}

// int sample_topp(std::vector<float> probs, int n, float topp, )


int main() {
    Model model("./models/stories15M.bin");
    RunState state(model.config);
    Tokenizer tokenizer("./models/tokenizer.bin", model.config.vocab_size);

    constexpr int bos_token = 1;
    int prev_token = bos_token;
    for(int i=0; i<100; i++) {
        std::vector<float>& logits = forward(model, state, prev_token, i);
        std::cout << tokenizer.decode(prev_token, sample_argmax(logits)) << std::flush;
        prev_token = sample_argmax(logits);
    }
}
