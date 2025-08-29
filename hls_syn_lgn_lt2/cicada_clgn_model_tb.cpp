#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>

#include "cicada_clgn_model.h"

#define N_BITS_ENCODING 4

// Function to read input from a file
ap_uint<504> *read_input_from_file(const char *filename, size_t n_events) {
    printf("in read_input_from_file\n");
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s for reading.\n", filename);
        return NULL;
    }

    // allocate array of n_events ap_uint<504>
    ap_uint<504> *inputs = (ap_uint<504> *)malloc(n_events * sizeof(ap_uint<504>));
    if (!inputs) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }
    
    size_t event = 0;
    size_t bit_idx = 0;
    int bit;
    ap_uint<504> current = 0;
    
    while (fscanf(file, "%d", &bit) == 1) {
        if (bit != 0 && bit != 1) {
            fprintf(stderr, "Invalid bit: %d at event %zu, bit %zu\n", bit, event, bit_idx);
            free(inputs);
            fclose(file);
            return NULL;
        }

        if (bit) {
            current.set(bit_idx);
        }
        bit_idx++;

        if (bit_idx == 504) {
            // finished one event
            inputs[event] = current;
            event++;
            if (event >= n_events) break;

            // reset for next event
            bit_idx = 0;
            current = 0;
        }
    }

    fclose(file);
    return inputs;

}

// Function to read exactly n_events lines from reference file
ap_uint<12> *read_expected_output_from_file(const char *filename, size_t n_events) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    ap_uint<12> *output = (ap_uint<12> *)malloc(n_events * sizeof(ap_uint<12>));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    size_t index = 0;
    int tmp;  // fscanf reads into a regular int
    while (index < n_events && fscanf(file, "%d", &tmp) == 1) {
        if (tmp < 0 || tmp > 4095) { // 12-bit range check
            fprintf(stderr, "Warning: value %d out of range for ap_uint<12>\n", tmp);
        }
        output[index] = (ap_uint<12>)tmp;
        index++;
    }

    if (index < n_events) {
        fprintf(stderr, "Warning: Expected %zu values, but only read %zu from file.\n", n_events, index);
    }

    fclose(file);

    return output;
}

void apply_logic_net(ap_uint<504> *inp, ap_uint<12> *out, size_t len) {
    for (size_t i = 0; i < len; i++) {

        // Call top function for this event
        ap_uint<12> result;
        top_logic_net(inp[i], result);

        // Store the result
        out[i] = result;
    }
}

// Test bench function
void test_logic_gate_net() {
    size_t n_events = 5;

    printf("n_events = %ld\n", n_events);

    ap_uint<504> *test_input = read_input_from_file("x_val.txt", n_events);

    ap_uint<12> *test_output = (ap_uint<12> *)malloc(n_events * sizeof(ap_uint<12>));
    apply_logic_net(test_input, test_output, n_events);

    printf("test_output = \n");
    for (size_t j = 0; j < n_events; ++j) { 
        printf("%u ", (unsigned int)test_output[j]);
        if ((j + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    ap_uint<12> *expected_output = read_expected_output_from_file("y_val_ref.txt", n_events);
    printf("expected_output = \n");
    for (size_t k = 0; k < n_events; ++k) { 
        printf("%u ", (unsigned int)expected_output[k]);
        if ((k + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    int match = 1;
    for (size_t l = 0; l < n_events; ++l) {
        if (test_output[l] != expected_output[l]) {
            fprintf(
                stderr, 
                "Mismatch at index %zu: expected %u, got %u\n", 
                l, 
                (unsigned int)expected_output[l], 
                (unsigned int)test_output[l]);
            match = 0;
        }
    }

    printf(match ? "Test passed!\n" : "Test failed!\n");

    free(test_input);
    free(test_output);
    free(expected_output);
}

int main() {
    printf("In main\n");
    test_logic_gate_net();
    return 0;
}