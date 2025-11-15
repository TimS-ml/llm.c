/*
================================================================================
File: tokenizer.h
Purpose: GPT-2 BPE tokenizer (decode-only) for text generation
================================================================================

Overview:
---------
This file implements a minimal GPT-2 tokenizer that supports ONLY decoding
(converting token IDs back to text). This is sufficient for unconditional
generation where the model produces tokens and we need to display the result.

Why Decode-Only?
---------------
- Encoding (text -> tokens) requires complex regex for byte-pair merging
- Decoding (tokens -> text) is simple: just table lookup
- For generation, we only need decoding (model outputs tokens, we display text)
- If prompting is needed later, encoding would need to be implemented

Tokenization Basics:
-------------------
GPT-2 uses Byte-Pair Encoding (BPE):
- Vocabulary: ~50K tokens (subword units)
- Each token maps to a byte sequence
- <|endoftext|> is a special token marking document boundaries

File Format:
-----------
The tokenizer file (.bin) contains:
- Header: [magic, version, vocab_size, eot_token]
- Token table: [length, bytes] for each token

Usage:
------
    Tokenizer tok;
    tokenizer_init(&tok, "gpt2_tokenizer.bin");
    const char* text = tokenizer_decode(&tok, token_id);
    safe_printf(text);  // Print token (handles unprintable characters)
    tokenizer_free(&tok);
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ============================================================================
// TOKENIZER DATA STRUCTURE
// ============================================================================

/**
 * Tokenizer - GPT-2 tokenizer state
 *
 * @field vocab_size: Number of tokens in vocabulary (~50257 for GPT-2)
 * @field token_table: Array of strings, one per token ID
 * @field init_ok: 1 if successfully initialized, 0 if init failed
 * @field eot_token: ID of the <|endoftext|> token (usually 50256)
 */
typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;

/**
 * safe_printf - Prints a token string, filtering out unprintable characters
 *
 * @param piece: Token string to print
 *
 * Many tokens represent raw bytes including control codes, backspace, etc.
 * This function only prints tokens that are:
 * - Printable (isprint returns true)
 * - Whitespace (isspace returns true)
 *
 * Single-byte tokens are checked more carefully since they might be
 * control codes. Multi-byte tokens are printed as-is.
 *
 * This prevents terminal corruption when displaying generated text.
 */
void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

/**
 * tokenizer_init - Loads tokenizer from a binary file
 *
 * @param tokenizer: Tokenizer struct to initialize
 * @param filename: Path to tokenizer file (.bin)
 *
 * File format:
 * - Header (256 uint32_t):
 *   [0]: Magic number (20240328)
 *   [1]: Version (1 or 2)
 *   [2]: Vocabulary size
 *   [3]: EOT token ID (only in version 2)
 * - For each token:
 *   - 1 byte: String length
 *   - N bytes: Token bytes
 *
 * Version differences:
 * - Version 1: EOT token assumed to be 50256 (GPT-2 default)
 * - Version 2: EOT token ID stored in header[3]
 *
 * If file cannot be opened or is invalid, sets init_ok=0 and prints warnings.
 */
void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

/**
 * tokenizer_decode - Converts a token ID to its string representation
 *
 * @param tokenizer: Initialized tokenizer
 * @param token_id: Token ID to decode
 * @return: Pointer to token string, or NULL if tokenizer not initialized or invalid ID
 *
 * Simple table lookup: token_table[token_id].
 * Returns NULL for out-of-range token IDs.
 */
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

/**
 * tokenizer_free - Frees all memory allocated by the tokenizer
 *
 * @param tokenizer: Tokenizer to free
 *
 * Frees:
 * - Each token string
 * - The token table array
 */
void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}
