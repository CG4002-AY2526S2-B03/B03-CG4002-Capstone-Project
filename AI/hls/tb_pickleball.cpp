// PickleballNet HLS C-Simulation Testbench (INT8)
// Validates HLS output against quantized PyTorch golden reference.

#include "pickleball_model.h"
#include "weights.h"
#include "test_vectors.h"
#include <cstdio>
#include <cmath>

union float_uint32_tb {
    float f;
    unsigned int u;
};

void push_float(hls::stream<axis_pkt_t> &s, float val, bool last = false) {
    axis_pkt_t pkt;
    float_uint32_tb conv;
    conv.f = val;
    pkt.data = conv.u;
    pkt.keep = 0xF;
    pkt.strb = 0xF;
    pkt.last = last ? 1 : 0;
    s.write(pkt);
}

float pop_float(hls::stream<axis_pkt_t> &s) {
    axis_pkt_t pkt = s.read();
    float_uint32_tb conv;
    conv.u = pkt.data;
    return conv.f;
}

int main() {
    int n_match_golden = 0;
    int n_correct_cls = 0;
    float total_reg_mse = 0.0f;
    float total_cls_mse = 0.0f;

    const char* class_names[] = {"Drive", "Drop", "Dink", "Lob", "SpeedUp", "HandBattle"};

    printf("=====================================================================\n");
    printf("  PickleballNet HLS C-Simulation Testbench (INT8 Quantized)\n");
    printf("  Architecture: %d hidden layers x %d units, ReLU6, BN-fused\n", N_LAYERS, HIDDEN);
    printf("  Test samples: %d\n", N_TESTS);
    printf("=====================================================================\n\n");

    for (int t = 0; t < N_TESTS; t++) {
        hls::stream<axis_pkt_t> in_stream("input");
        hls::stream<axis_pkt_t> out_stream("output");

        // Feed raw input
        for (int i = 0; i < IN_DIM; i++) {
            push_float(in_stream, test_inputs[t][i], (i == IN_DIM - 1));
        }

        // Run HLS kernel
        pb_predict(in_stream, out_stream);

        // Read outputs
        float reg_result[OUT_REG];
        float cls_result[OUT_CLS];
        for (int i = 0; i < OUT_REG; i++) reg_result[i] = pop_float(out_stream);
        for (int i = 0; i < OUT_CLS; i++) cls_result[i] = pop_float(out_stream);

        // Argmax
        int pred_cls = 0;
        float max_logit = cls_result[0];
        for (int i = 1; i < OUT_CLS; i++) {
            if (cls_result[i] > max_logit) {
                max_logit = cls_result[i];
                pred_cls = i;
            }
        }

        // Compare with golden reference
        float reg_mse = 0.0f;
        for (int i = 0; i < OUT_REG; i++) {
            float diff = reg_result[i] - expected_reg[t][i];
            reg_mse += diff * diff;
        }
        reg_mse /= OUT_REG;
        total_reg_mse += reg_mse;

        float cls_mse = 0.0f;
        for (int i = 0; i < OUT_CLS; i++) {
            float diff = cls_result[i] - expected_cls_logits[t][i];
            cls_mse += diff * diff;
        }
        cls_mse /= OUT_CLS;
        total_cls_mse += cls_mse;

        bool cls_matches_golden = (pred_cls == expected_pred_cls[t]);
        // Tolerance for int8 FP32 rounding diffs
        bool reg_matches_golden = (reg_mse < 0.01f);
        bool both_match = cls_matches_golden && reg_matches_golden;

        if (both_match) n_match_golden++;
        if (pred_cls == expected_true_cls[t]) n_correct_cls++;

        const char* match_str = both_match ? "OK" : "MISMATCH";
        const char* true_name = class_names[expected_true_cls[t]];
        const char* pred_name = class_names[pred_cls];
        const char* gold_name = class_names[expected_pred_cls[t]];

        printf("Test %2d: true=%-10s hls=%-10s gold=%-10s %s | reg_mse=%.6f\n",
               t, true_name, pred_name, gold_name, match_str, reg_mse);

        if (!both_match) {
            printf("         HLS reg:  [");
            for (int i = 0; i < OUT_REG; i++) printf("%.4f%s", reg_result[i], i < 5 ? ", " : "");
            printf("]\n");
            printf("         Gold reg: [");
            for (int i = 0; i < OUT_REG; i++) printf("%.4f%s", expected_reg[t][i], i < 5 ? ", " : "");
            printf("]\n");
        }
    }

    printf("\n=====================================================================\n");
    printf("  VALIDATION SUMMARY\n");
    printf("=====================================================================\n");
    printf("  Architecture:           %d layers x %d units, ReLU6\n", N_LAYERS, HIDDEN);
    printf("  Samples tested:         %d\n", N_TESTS);
    printf("  HLS vs Quantized golden:  %d/%d (%s)\n",
           n_match_golden, N_TESTS, (n_match_golden == N_TESTS) ? "PASS" : "FAIL");
    printf("  HLS accuracy (vs true): %d/%d (%.1f%%)\n",
           n_correct_cls, N_TESTS, (100.0f * n_correct_cls) / N_TESTS);
    printf("  Avg reg MSE vs golden:  %.6f\n", total_reg_mse / N_TESTS);
    printf("  Avg cls MSE vs golden:  %.6f\n", total_cls_mse / N_TESTS);
    printf("=====================================================================\n");

    if (n_match_golden == N_TESTS) {
        printf("\n  >>> ALL TESTS PASSED - HLS matches quantized golden reference <<<\n\n");
        return 0;
    } else {
        printf("\n  >>> %d MISMATCHES - investigate int8 quantization or precision <<<\n\n",
               N_TESTS - n_match_golden);
        return 1;
    }
}
