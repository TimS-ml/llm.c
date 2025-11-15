/*
================================================================================
File: utils.h
Purpose: Common utility functions and error-checking wrappers for file I/O and
         memory management operations
================================================================================

Overview:
---------
This file provides a comprehensive set of error-handling wrappers around standard
C library functions for file operations, memory allocation, and data validation.
The primary design pattern is to replace standard library calls with "Check"
versions that validate return codes and provide detailed error messages.

Design Pattern:
--------------
Each wrapper follows a consistent pattern:
1. An inline function (e.g., fopen_check) that performs the actual operation
2. Checks the return code for errors
3. Prints detailed diagnostic information including file name, line number, and context
4. Exits the program if an error occurred
5. A macro (e.g., fopenCheck) that automatically injects __FILE__ and __LINE__

Benefits:
---------
- Eliminates boilerplate error checking code
- Provides consistent, detailed error messages
- Makes debugging easier by showing exact location of failures
- Includes helpful hints for common issues (e.g., dataset location)

Usage Example:
-------------
    Instead of:
        FILE* fp = fopen("data.bin", "rb");
        if (fp == NULL) { handle_error(); }

    Simply write:
        FILE* fp = fopenCheck("data.bin", "rb");

    The macro automatically expands to include file/line information.

Functions Provided:
------------------
- File I/O: fopenCheck, freadCheck, fwriteCheck, fcloseCheck, fseekCheck
- Socket I/O: scloseCheck, closesocketCheck (Windows)
- Memory: mallocCheck
- Validation: tokenCheck
- Utilities: create_dir_if_not_exists, find_max_step, ends_with_bin
*/
#ifndef UTILS_H
#define UTILS_H

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#include <dirent.h>
#include <arpa/inet.h>
#endif

// ============================================================================
// FILE I/O ERROR-CHECKING WRAPPERS
// ============================================================================
// The following functions wrap standard C file I/O operations with comprehensive
// error checking. Each function validates the operation's return code and provides
// detailed diagnostic information if an error occurs.
//
// Implementation Note: These use the 'extern inline' specifier to allow the
// compiler to inline them for performance while still being available for linking.
// ============================================================================

/**
 * fopen_check - Opens a file with comprehensive error checking
 *
 * @param path: Path to the file to open
 * @param mode: File open mode (e.g., "r", "rb", "w", "wb", "a")
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 * @return: File pointer on success, exits program on failure
 *
 * This function wraps fopen() with error checking. If the file cannot be opened,
 * it prints detailed error information including:
 * - The file path that failed to open
 * - The source code location (file:line) where the error occurred
 * - Helpful hints for common issues (e.g., dataset location changes)
 *
 * Note: Users should call this via the fopenCheck macro, not directly.
 */
extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

/**
 * fopenCheck - Macro wrapper for fopen_check
 *
 * Usage: FILE* fp = fopenCheck("data.bin", "rb");
 *
 * Automatically captures the current file name and line number using __FILE__
 * and __LINE__ preprocessor macros, then calls fopen_check with this context.
 */
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

/**
 * fread_check - Reads data from a file with comprehensive error checking
 *
 * @param ptr: Pointer to buffer where data will be stored
 * @param size: Size of each element to read (in bytes)
 * @param nmemb: Number of elements to read
 * @param stream: File pointer to read from
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * This function wraps fread() with error checking. It ensures that exactly
 * nmemb elements are read from the file. If fewer elements are read, it
 * determines the cause:
 * - End of file (EOF) was reached unexpectedly
 * - A file read error occurred
 * - A partial read occurred (read some but not all requested elements)
 *
 * The function exits the program if the read operation fails or returns
 * fewer than the requested number of elements.
 *
 * Note: Users should call this via the freadCheck macro, not directly.
 */
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

/**
 * freadCheck - Macro wrapper for fread_check
 *
 * Usage: freadCheck(buffer, sizeof(int), 100, fp);
 *
 * Automatically injects source file and line number for error reporting.
 */
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

/**
 * fclose_check - Closes a file with error checking
 *
 * @param fp: File pointer to close
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Wraps fclose() and ensures the file is successfully closed. If fclose()
 * returns a non-zero value (indicating an error, such as failure to flush
 * buffered data), the program exits with an error message.
 *
 * Note: Users should call this via the fcloseCheck macro, not directly.
 */
extern inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

/**
 * fcloseCheck - Macro wrapper for fclose_check
 *
 * Usage: fcloseCheck(fp);
 */
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

/**
 * sclose_check - Closes a socket with error checking (Unix/Linux)
 *
 * @param sockfd: Socket file descriptor to close
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Wraps close() for socket file descriptors with error checking. Used for
 * network programming to ensure sockets are properly closed.
 *
 * Note: Users should call this via the scloseCheck macro, not directly.
 */
extern inline void sclose_check(int sockfd, const char *file, int line) {
    if (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

/**
 * scloseCheck - Macro wrapper for sclose_check
 *
 * Usage: scloseCheck(sockfd);
 */
#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
/**
 * closesocket_check - Closes a socket with error checking (Windows)
 *
 * @param sockfd: Socket file descriptor to close
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Windows-specific version of socket closing with error checking. Uses the
 * closesocket() function from Winsock API instead of close().
 *
 * Note: Users should call this via the closesocketCheck macro, not directly.
 */
extern inline void closesocket_check(int sockfd, const char *file, int line) {
    if (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

/**
 * closesocketCheck - Macro wrapper for closesocket_check (Windows)
 *
 * Usage: closesocketCheck(sockfd);
 */
#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

/**
 * fseek_check - Seeks to a position in a file with error checking
 *
 * @param fp: File pointer
 * @param off: Offset in bytes
 * @param whence: Starting position (SEEK_SET, SEEK_CUR, or SEEK_END)
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Wraps fseek() with error checking. Ensures the file position indicator
 * is successfully moved to the specified location. Common uses:
 * - SEEK_SET: Seek from beginning of file
 * - SEEK_CUR: Seek from current position
 * - SEEK_END: Seek from end of file
 *
 * Note: Users should call this via the fseekCheck macro, not directly.
 */
extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
    if (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

/**
 * fseekCheck - Macro wrapper for fseek_check
 *
 * Usage: fseekCheck(fp, 0, SEEK_SET);  // Seek to beginning of file
 */
#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

/**
 * fwrite_check - Writes data to a file with comprehensive error checking
 *
 * @param ptr: Pointer to data to write
 * @param size: Size of each element (in bytes)
 * @param nmemb: Number of elements to write
 * @param stream: File pointer to write to
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Wraps fwrite() with error checking. Ensures that exactly nmemb elements
 * are written to the file. Similar to fread_check, it detects:
 * - Unexpected end of file
 * - File write errors
 * - Partial writes (wrote some but not all requested elements)
 *
 * Common causes of write failures include:
 * - Disk full
 * - Permission issues
 * - I/O errors
 *
 * Note: Users should call this via the fwriteCheck macro, not directly.
 */
extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

/**
 * fwriteCheck - Macro wrapper for fwrite_check
 *
 * Usage: fwriteCheck(data, sizeof(float), 1000, fp);
 */
#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ============================================================================
// MEMORY ALLOCATION ERROR-CHECKING WRAPPER
// ============================================================================
// Provides a malloc wrapper that checks for allocation failures and provides
// detailed error information when out-of-memory conditions occur.
// ============================================================================

/**
 * malloc_check - Allocates memory with error checking
 *
 * @param size: Number of bytes to allocate
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 * @return: Pointer to allocated memory on success, exits program on failure
 *
 * Wraps malloc() with error checking. If memory allocation fails (returns NULL),
 * the program prints detailed error information including:
 * - The source location where allocation was attempted
 * - The number of bytes requested
 * - Then exits the program
 *
 * This prevents the program from continuing with a NULL pointer, which would
 * likely cause a segmentation fault later.
 *
 * Note: Users should call this via the mallocCheck macro, not directly.
 */
extern inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/**
 * mallocCheck - Macro wrapper for malloc_check
 *
 * Usage: int* data = (int*)mallocCheck(1000 * sizeof(int));
 */
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ============================================================================
// TOKEN VALIDATION
// ============================================================================
// Validates that token IDs are within the valid vocabulary range. This is
// important for catching data corruption or indexing errors early.
// ============================================================================

/**
 * token_check - Validates that all tokens are within the vocabulary range
 *
 * @param tokens: Array of token IDs to validate
 * @param token_count: Number of tokens in the array
 * @param vocab_size: Size of the vocabulary (valid tokens are 0 to vocab_size-1)
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Iterates through an array of token IDs and ensures each one is within the
 * valid range [0, vocab_size). If an out-of-range token is found, prints:
 * - The invalid token value
 * - Its position in the array
 * - The vocabulary size
 * - The source location where the check was performed
 *
 * This is useful for catching:
 * - Data corruption in token files
 * - Tokenization errors
 * - Vocabulary mismatch between model and data
 *
 * Note: Users should call this via the tokenCheck macro, not directly.
 */
extern inline void token_check(const int* tokens, int token_count, int vocab_size, const char *file, int line) {
    for(int i = 0; i < token_count; i++) {
        if(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, line);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * tokenCheck - Macro wrapper for token_check
 *
 * Usage: tokenCheck(token_array, num_tokens, vocab_size);
 */
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)

// ============================================================================
// DIRECTORY AND FILE UTILITIES
// ============================================================================
// Helper functions for directory management, checkpoint discovery, and
// file path validation.
// ============================================================================

/**
 * create_dir_if_not_exists - Creates a directory if it doesn't already exist
 *
 * @param dir: Path to the directory to create
 *
 * Uses stat() to check if the directory exists. If it doesn't, creates it
 * with mode 0700 (owner has read/write/execute permissions, nobody else).
 *
 * If dir is NULL, this function does nothing (allows safe handling of
 * optional directory parameters).
 *
 * Common usage: Creating output directories for logs and checkpoints before
 * starting training.
 *
 * Note: Exits the program if directory creation fails.
 */
extern inline void create_dir_if_not_exists(const char *dir) {
    if (dir == NULL) { return; }
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

/**
 * find_max_step - Finds the highest completed training step from checkpoint files
 *
 * @param output_log_dir: Path to the log/checkpoint directory
 * @return: Highest step number found, or -1 if directory is NULL/doesn't exist
 *
 * This function scans the output log directory for files named "DONE_<step>"
 * which indicate completed training steps. It returns the highest step number
 * found, allowing training to resume from the last completed checkpoint.
 *
 * The function:
 * 1. Opens the directory
 * 2. Reads all entries
 * 3. Looks for files starting with "DONE_"
 * 4. Extracts the step number from the filename
 * 5. Tracks and returns the maximum step number
 *
 * Common usage: When resuming training, this determines which checkpoint to
 * load and what step number to continue from.
 *
 * Returns -1 if:
 * - output_log_dir is NULL
 * - Directory cannot be opened
 * - No DONE files are found
 */
extern inline int find_max_step(const char* output_log_dir) {
    if (output_log_dir == NULL) { return -1; }
    DIR* dir;
    struct dirent* entry;
    int max_step = -1;
    dir = opendir(output_log_dir);
    if (dir == NULL) { return -1; }
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "DONE_", 5) == 0) {
            int step = atoi(entry->d_name + 5);
            if (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    return max_step;
}

/**
 * ends_with_bin - Checks if a string ends with ".bin" extension
 *
 * @param str: String to check (typically a filename)
 * @return: 1 if string ends with ".bin", 0 otherwise
 *
 * This function performs a simple suffix check to determine if a filename
 * has a ".bin" extension. Useful for:
 * - Validating binary file inputs
 * - Filtering files by extension
 * - Ensuring correct file types are being loaded
 *
 * Implementation details:
 * - Returns 0 for NULL strings
 * - Returns 0 if string is shorter than ".bin" (4 characters)
 * - Compares the last 4 characters against ".bin"
 *
 * Example usage:
 *     if (ends_with_bin("model.bin")) {
 *         // Process binary file
 *     }
 *
 * Note: This could be generalized to check for arbitrary suffixes in the future.
 */
extern inline int ends_with_bin(const char* str) {
    if (str == NULL) { return 0; }
    size_t len = strlen(str);
    const char* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    if (len < suffix_len) { return 0; }
    int suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    return suffix_matches;
}

#endif