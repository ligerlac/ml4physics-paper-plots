#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include "cicada_clgn_model.h"

#define N_BITS 8
#define N_BITS_ENCODING 6

// Function to read input from a file
bool *read_input_from_file(const char *filename, size_t *size, size_t len, size_t *pad_len) {
    printf("in read_input_from_file\n");
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s for reading.\n", filename);
        return NULL;
    }

    size_t batch_size_div_bits = (len + N_BITS - 1) / N_BITS;
    *pad_len = batch_size_div_bits * N_BITS - len;
    size_t batch_size = batch_size_div_bits * N_BITS;

    printf("pad_len = %u\n", pad_len);
    printf("batch_size = %u\n", batch_size);

    size_t capacity = batch_size * N_BITS_ENCODING * 18 * 14;
    bool *input = (bool *)malloc(capacity * sizeof(bool));
    if (!input) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }
    
    size_t line = 0;
    size_t index = 0;
    int bit;
    size_t bits_in_line = 0;

    while (fscanf(file, "%d", &bit) == 1) {
        if (bit != 0 && bit != 1) {
            fprintf(stderr, "Invalid bit: %d at index %zu\n", bit, index);
            free(input);
            fclose(file);
            return NULL;
        }

        input[index++] = (bool)bit;
        bits_in_line++;

        if (bits_in_line == 18 * 14 * N_BITS_ENCODING) {
            bits_in_line = 0;
            line++;

            if (line >= len) break;
        }
    }

    if (line < len) {
        printf("Warning: requested %zu lines, but only %zu available in file.\n", len, line);
    }

    *size = index;
    fclose(file);
    return input;

}

// Function to read exactly n_events lines from reference file
int *read_expected_output_from_file(const char *filename, size_t n_events) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    int *output = (int *)malloc(n_events * sizeof(int));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    size_t index = 0;
    while (index < n_events && fscanf(file, "%d", &output[index]) == 1) {
        index++;
    }

    if (index < n_events) {
        fprintf(stderr, "Warning: Expected %zu values, but only read %zu from file.\n", n_events, index);
    }

    fclose(file);

    return output;
}

void apply_logic_net(bool const *inp, int *out, size_t len) {
    char *inp_temp = malloc(1512*sizeof(char));
    char *out_temp_o = malloc(9*sizeof(char));

    for(size_t i = 0; i < len; ++i) {

        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < 1512; ++d) {
            char res = (char) 0;
            for(size_t b = 0; b < 8; ++b) {
                res <<= 1;
                res += !!(inp[i * 1512 * 8 + (8 - b - 1) * 1512 + d]);
            }
            inp_temp[d] = res;
        }

        // Applying the logic net
        top_logic_net(inp_temp, out_temp_o);

        // Unpack the result bits
        for(size_t b = 0; b < 8; ++b) {
            const char bit_mask = (char) 1 << b;
            int res = 0;
            for(size_t d = 0; d < 9; ++d) {
                res <<= 1;
                res += !!(out_temp_o[d] & bit_mask);
            }
            out[i * 8 + b] = res;
        }
    }
    free(inp_temp);
    free(out_temp_o);
}


// Test bench function
void test_logic_net() {
    printf("Start of test bench function \n");

    size_t input_size = 0, expected_output_size = 0, pad_len = 0;
    size_t n_events = 10;

    printf("input_size = %ld\n", input_size);
    printf("expected_output_size = %ld\n", expected_output_size);
    printf("allocated_size = %ld\n", n_events);

    bool *test_input = read_input_from_file("x_val.txt", &input_size, n_events, &pad_len);
    printf("input_size = %ld\n", input_size);
    printf("pad_len = %zu\n", pad_len);

    size_t len = (n_events + pad_len) / N_BITS;
    size_t output_size = (n_events + pad_len);
    int *test_output = (int *)malloc(output_size * sizeof(int));
    apply_logic_net(test_input, test_output, len);

    printf("test_output = \n");
    for (size_t j = 0; j < n_events; ++j) { 
        printf("%d ", test_output[j]);
        if ((j + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    int *expected_output = read_expected_output_from_file("y_val_lgn_ref.txt", n_events);
    printf("expected_output = \n");
    for (size_t k = 0; k < n_events; ++k) { 
        printf("%d ", expected_output[k]);
        if ((k + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    int match = 1;
    for (size_t l = 0; l < n_events; ++l) {
        if (test_output[l] != expected_output[l]) {
            fprintf(stderr, "Mismatch at index %zu: expected %d, got %d\n", l, expected_output[l], test_output[l]);
            match = 0;
        }
    }

    printf(match ? "Test passed!\n" : "Test failed!\n");

    free(test_input);
    free(test_output);
    free(expected_output);
}

int main() {
    test_logic_net();
    return 0;
}
