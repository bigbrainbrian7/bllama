#include "tokenizer.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int vocab_size = 32000;
int failures = 0;

void print_tokens(const char * label, const std::vector<int> & tokens) {
    std::fprintf(stderr, "  %s:", label);
    for (const int id : tokens) {
        std::fprintf(stderr, " %d", id);
    }
    std::fputc('\n', stderr);
}

void expect_tokens(const char * test, const std::vector<int> & actual, const std::vector<int> & expected) {
    if (actual == expected) {
        return;
    }
    std::fprintf(stderr, "%s: token mismatch\n", test);
    print_tokens("actual  ", actual);
    print_tokens("expected", expected);
    failures++;
}

void test_vocab(const Tokenizer & tokenizer) {
    if (tokenizer.vocab.size() != vocab_size || tokenizer.scores.size() != vocab_size) {
        std::fprintf(stderr, "vocab: %zu tokens, %zu scores, expected %d\n", tokenizer.vocab.size(),
                     tokenizer.scores.size(), vocab_size);
        failures++;
        return;
    }

    // encode() byte fallback assumes <0xNN> is at id NN + 3
    for (int byte = 0; byte < 256; byte++) {
        std::array<char, 8> expected{};
        std::snprintf(expected.data(), expected.size(), "<0x%02X>", byte);
        if (tokenizer.vocab[static_cast<size_t>(byte + 3)] != expected.data()) {
            std::fprintf(stderr, "vocab: id %d is \"%s\", expected \"%s\"\n", byte + 3,
                         tokenizer.vocab[static_cast<size_t>(byte + 3)].c_str(), expected.data());
            failures++;
        }
    }

    const auto it = tokenizer.token_to_id.find(" Hello");
    if (it == tokenizer.token_to_id.end() || it->second != 15043) {
        std::fprintf(stderr, "vocab: \" Hello\" does not map to 15043\n");
        failures++;
    }
}

struct Case {
    const char * name;
    std::string prompt;
    std::vector<int> expected;
};

// from the sentencepiece Python library on tokenizer.model, so a bug shared with the tokenizer.bin export cannot hide
void test_encode_matches_sentencepiece(const Tokenizer & tokenizer) {
    const std::vector<Case> cases{
        {"ascii", "Hello world", {15043, 3186}},
        {"sentence",
         "Once upon a time, there was a little girl named Lily.",
         {9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889}},
        {"single char", "x", {921}},
        {"multi-piece word", "tokenization unbelievable", {5993, 2133, 443, 6596, 10384, 519}},
        {"repeated spaces", "  two  spaces", {259, 1023, 29871, 8162}},
        {"2-byte codepoint", "caf\xC3\xA9", {274, 28059}},
        {"3-byte codepoints", "\xE4\xB8\xAD\xE6\x96\x87", {29871, 30275, 30333}},
        // split literal so the hex escape does not eat the 'b'
        {"control codepoint", "a\xC2\x80" "b", {263, 30751, 29890}},
        {"byte fallback 4-byte", "\xF0\x9F\x98\x80", {29871, 243, 162, 155, 131}},
        {"byte fallback 1-byte", "\n", {29871, 13}},
    };

    for (const Case & c : cases) {
        expect_tokens(c.name, tokenizer.encode(c.prompt, false, false), c.expected);
    }
}

void test_special_tokens(const Tokenizer & tokenizer) {
    expect_tokens("bos", tokenizer.encode("Hello world", true, false), {Tokenizer::bos_id, 15043, 3186});
    expect_tokens("eos", tokenizer.encode("Hello world", false, true), {15043, 3186, Tokenizer::eos_id});
    expect_tokens("bos and eos", tokenizer.encode("Hello world", true, true),
                  {Tokenizer::bos_id, 15043, 3186, Tokenizer::eos_id});

    // empty prompt gets no dummy-prefix space, so generation can start from BOS alone
    expect_tokens("empty", tokenizer.encode("", false, false), {});
    expect_tokens("empty bos", tokenizer.encode("", true, false), {Tokenizer::bos_id});
    expect_tokens("empty bos and eos", tokenizer.encode("", true, true), {Tokenizer::bos_id, Tokenizer::eos_id});
}

}  // namespace

int main(int argc, char ** argv) {
    // make test passes the checkpoint path, tokenizer.bin is in the same directory
    const std::filesystem::path checkpoint = argc > 1 ? argv[1] : "./models/stories15M.bin";
    const std::string tokenizer_path = (checkpoint.parent_path() / "tokenizer.bin").string();
    const Tokenizer tokenizer(tokenizer_path.c_str(), vocab_size);

    test_vocab(tokenizer);
    test_encode_matches_sentencepiece(tokenizer);
    test_special_tokens(tokenizer);

    std::puts(failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
