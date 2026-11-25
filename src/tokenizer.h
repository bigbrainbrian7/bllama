#pragma once

#include <vector>
#include <string>
#include <unordered_map>

struct Tokenizer {
    static constexpr int bos_id = 1;
    static constexpr int eos_id = 2;
    static constexpr int byte_token_offset = 3;

    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::unordered_map<std::string, int> token_to_id;

    Tokenizer(const char *tokenizer_path, int vocab_size);

    std::vector<int> encode(const std::string& prompt, bool bos, bool eos) const;

    std::string decode(int prev_token, int token) const;
};
