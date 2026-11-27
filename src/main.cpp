#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>


void generate(const Model & model, RunState & state, const Tokenizer & tokenizer, Sampler & sampler, const std::string & prompt, int steps) {
    if (steps <= 0 || steps > model.config.seq_len) {
        steps = model.config.seq_len;
    }

    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    if (static_cast<int>(prompt_tokens.size()) > steps) {
        prompt_tokens.resize(steps); // no room left in the context for the rest of the prompt
    }
    const int n_prompt = static_cast<int>(prompt_tokens.size());

    using clock = std::chrono::steady_clock;

    for (int pos = 1; pos < n_prompt; pos++) {
        std::cout << tokenizer.decode(prompt_tokens[pos - 1], prompt_tokens[pos]) << std::flush;
    }

    // untimed warmup: one pass touches every weight, so mmap page faults and cold caches stay out of the numbers
    // prefill overwrites the pos 0 kv entry it leaves behind
    forward(model, state, prompt_tokens[0], 0);

    const auto prefill_start = clock::now();
    std::vector<float> * logits = nullptr;
    for (int pos = 0; pos < n_prompt; pos++) {
        logits = &forward(model, state, prompt_tokens[pos], pos);
    }
    const auto prefill_end = clock::now();

    int token       = prompt_tokens.back();
    int pos         = n_prompt;
    int n_generated = 0;
    // a \r status line would overwrite the story if both streams share the terminal
    const bool live = isatty(STDERR_FILENO) && !isatty(STDOUT_FILENO);
    while (true) {
        const int next = sampler.sample(*logits);

        if (next == Tokenizer::bos_id || next == Tokenizer::eos_id) {
            break;
        }

        std::cout << tokenizer.decode(token, next) << std::flush;
        token = next;
        n_generated++;

        if (live) {
            const std::chrono::duration<double> s = clock::now() - prefill_end;
            std::fprintf(stderr, "\rgeneration: %d tokens, %.1f tok/s", n_generated, n_generated / s.count());
        }

        if (pos >= steps) {
            break;
        }
        logits = &forward(model, state, token, pos);
        pos++;
    }
    const auto gen_end = clock::now();
    std::cout << '\n';
    if (live) {
        std::fprintf(stderr, "\r\033[K");
    }

    const std::chrono::duration<double> prefill_s = prefill_end - prefill_start;
    const std::chrono::duration<double> gen_s     = gen_end - prefill_end;
    auto rate = [](int n, double s) { return s > 0 ? n / s : 0.0; };
    std::cerr << "prefill:    " << n_prompt << " tokens in " << prefill_s.count() << "s, "
              << rate(n_prompt, prefill_s.count()) << " tok/s\n"
              << "generation: " << n_generated << " tokens in " << gen_s.count() << "s, "
              << rate(n_generated, gen_s.count()) << " tok/s\n";
}


int main() {
    constexpr float temperature = 1.0f;

    Model     model("./models/stories15M.bin");
    RunState  state(model.config);
    Tokenizer tokenizer("./models/tokenizer.bin", model.config.vocab_size);
    Sampler   sampler(model.config.vocab_size, temperature, std::random_device{}());

    std::string prompt(std::istreambuf_iterator<char>(std::cin), {});
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r')) {
        prompt.pop_back();
    }

    generate(model, state, tokenizer, sampler, prompt, 0);
}
