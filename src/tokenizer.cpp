#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

Tokenizer::Tokenizer(const char *tokenizer_path, int vocab_size) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(tokenizer_path, "rb"), &std::fclose);
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", tokenizer_path);
        std::exit(EXIT_FAILURE);
    }

    // llama2.c tokenizer.bin header; the per-token lengths make it redundant here.
    int max_token_length;
    if (std::fread(&max_token_length, sizeof(int), 1, file.get()) != 1) {
        fprintf(stderr, "Could not read max token length from %s\n", tokenizer_path);
        std::exit(EXIT_FAILURE);
    }

    vocab.reserve(static_cast<size_t>(vocab_size));
    scores.reserve(static_cast<size_t>(vocab_size));
    token_to_id.reserve(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        if (std::fread(&score, sizeof(float), 1, file.get()) != 1 ||
            std::fread(&len, sizeof(int), 1, file.get()) != 1 || len < 0) {
            fprintf(stderr, "Could not read token %d header from %s\n", i, tokenizer_path);
            std::exit(EXIT_FAILURE);
        }
        scores.push_back(score);

        std::string& token = vocab.emplace_back(static_cast<size_t>(len), '\0');
        if (std::fread(token.data(), 1, token.size(), file.get()) != token.size()) {
            fprintf(stderr, "Could not read token %d from %s\n", i, tokenizer_path);
            std::exit(EXIT_FAILURE);

        }

        // Keep the lowest id if the vocab contains duplicate strings.
        token_to_id.try_emplace(token, i);
    }
}

std::vector<int> Tokenizer::encode(const std::string& prompt, bool bos, bool eos) const {
    std::vector<int> tokens;

    // SentencePiece's add_dummy_prefix: the model was trained with a leading space on every input.
    const std::string text = prompt.empty() ? prompt : " " + prompt;
    tokens.reserve(text.size() + 2);

    //transform all ids into token
    for (size_t i = 0; i < text.size();) {
        size_t cp_len = 1;
        // Group UTF-8 continuation bytes (10xxxxxx) with their lead byte so whole codepoints are looked up.
        while (cp_len < 4 && i + cp_len < text.size() &&
               (static_cast<unsigned char>(text[i + cp_len]) & 0xC0) == 0x80) {
            cp_len++;
        }

        const auto it = token_to_id.find(text.substr(i, cp_len));
        if (it != token_to_id.end()) {
            tokens.push_back(it->second);
        } else {
            // Byte fallback: ids 0-2 are <unk>, <s>, </s>, so <0x00>..<0xFF> start at 3
            // these are the fundamental mapping of bytes to ids for llamas tokenizer
            for (size_t j = 0; j < cp_len; j++) {
                tokens.push_back(static_cast<unsigned char>(text[i + j]) + byte_token_offset);
            }
        }
        i += cp_len;
    }

    // naive sentencepiece merge implementation, recalculates scores on every pass
    while (true) {
        float best_score = std::numeric_limits<float>::lowest();
        int best_id = -1;
        size_t best_idx = 0;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            const auto it = token_to_id.find(vocab[tokens[i]] + vocab[tokens[i + 1]]);
            if (it != token_to_id.end() && scores[it->second] > best_score) {
                best_score = scores[it->second];
                best_id = it->second;
                best_idx = i;
            }
        }

        if (best_id == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    }

    // Added after merging so the special tokens never form a merge pair with prompt text.
    if (bos) {
        tokens.insert(tokens.begin(), bos_id);
    }
    if (eos) {
        tokens.push_back(eos_id);
    }

    return tokens;
}

std::string Tokenizer::decode (int prev_token, int token) const {
    const std::string& piece = vocab[token];

    if (token >= byte_token_offset and token < byte_token_offset + 256) {
        return std::string(1, static_cast<char>(token-byte_token_offset));
    }

    if (prev_token == bos_id && piece[0]== ' ') {
        return piece.substr(1);
    }

    return piece;
}
