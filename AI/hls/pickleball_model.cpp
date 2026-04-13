// PickleballNet Vitis HLS (INT8 quantized, AXI-Stream)
// Input: 6 x float32 (raw) -> Output: 12 x float32 (6 reg + 6 cls logits)
// BN fused at export. Weights quantized to int8.
//
// Pipeline notes:
//   LAYER0 (6->512): outer loop pipelined II=1, inner loop fully unrolled.
//     6 parallel fmuls + adder tree -> no loop-carried dependency -> II=1 clean.
//   LAYER1 + HEAD_L0 (512->512, 512->256): 16 independent partial accumulators
//     (s0..s15), j+=16, II=4. Float fadd latency ~11 cycles; with 16 chains
//     HLS achieves II=4 cleanly. cyclic=16 BRAM provides 16 concurrent reads.
//   Heads: DATAFLOW parallelism via run_heads(). REG and CLS pipelines execute
//     concurrently after LAYER1 completes. Each uses separate reg_buf/cls_buf
//     (required for HLS DATAFLOW fan-out of h_b).
//   HEAD_L1 (256->6): 16 accumulators II=4, only 16 inner iterations -- fine.
//
// Memory:
//   BRAM (rom_2p, cyclic=16): trunk_1, reg/cls_head_0 weights
//   LUTRAM: trunk_0, head_1 weights, biases, scalers
//   h_a/h_b: cyclic=16, impl chosen by HLS (FFs after partition)
//   reg_buf/cls_buf: cyclic=16 inside run_heads
//   Regs (complete partition): input_raw, input_scaled, reg_out, cls_out

#include "pickleball_model.h"
#include "weights.h"
#include <hls_math.h>

union float_uint32 {
    float f;
    unsigned int u;
};

static inline float relu6(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 6.0f) return 6.0f;
    return x;
}

// Canonical DATAFLOW requires only variable declarations and function calls at top level.
// REG and CLS heads are split into dedicated functions so run_heads() is purely function calls.

static void run_reg_head(float h_b[HIDDEN], float reg_out[OUT_REG]) {
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16
    float reg_buf[HEAD_HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=reg_buf cyclic factor=16

    // REG Head Layer 0 (512 -> 256), 16 accumulators II=4
    REG_HEAD_L0:
    for (int i = 0; i < HEAD_HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        REG_H0_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 0] * h_b[j+ 0];
            s1  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 1] * h_b[j+ 1];
            s2  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 2] * h_b[j+ 2];
            s3  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 3] * h_b[j+ 3];
            s4  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 4] * h_b[j+ 4];
            s5  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 5] * h_b[j+ 5];
            s6  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 6] * h_b[j+ 6];
            s7  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 7] * h_b[j+ 7];
            s8  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 8] * h_b[j+ 8];
            s9  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 9] * h_b[j+ 9];
            s10 += (float)reg_head_0_weight_q[i*HIDDEN+j+10] * h_b[j+10];
            s11 += (float)reg_head_0_weight_q[i*HIDDEN+j+11] * h_b[j+11];
            s12 += (float)reg_head_0_weight_q[i*HIDDEN+j+12] * h_b[j+12];
            s13 += (float)reg_head_0_weight_q[i*HIDDEN+j+13] * h_b[j+13];
            s14 += (float)reg_head_0_weight_q[i*HIDDEN+j+14] * h_b[j+14];
            s15 += (float)reg_head_0_weight_q[i*HIDDEN+j+15] * h_b[j+15];
        }
        reg_buf[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * reg_head_0_qscale + reg_head_0_bias[i]);
    }

    // REG Head Layer 1 (256 -> 6) + inverse scaler, 16 accumulators II=4
    REG_HEAD_L1:
    for (int i = 0; i < OUT_REG; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        REG_H1_MAC:
        for (int j = 0; j < HEAD_HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 0] * reg_buf[j+ 0];
            s1  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 1] * reg_buf[j+ 1];
            s2  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 2] * reg_buf[j+ 2];
            s3  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 3] * reg_buf[j+ 3];
            s4  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 4] * reg_buf[j+ 4];
            s5  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 5] * reg_buf[j+ 5];
            s6  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 6] * reg_buf[j+ 6];
            s7  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 7] * reg_buf[j+ 7];
            s8  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 8] * reg_buf[j+ 8];
            s9  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 9] * reg_buf[j+ 9];
            s10 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+10] * reg_buf[j+10];
            s11 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+11] * reg_buf[j+11];
            s12 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+12] * reg_buf[j+12];
            s13 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+13] * reg_buf[j+13];
            s14 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+14] * reg_buf[j+14];
            s15 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+15] * reg_buf[j+15];
        }
        float y_norm = (s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * reg_head_1_qscale + reg_head_1_bias[i];
        reg_out[i] = y_norm * y_scaler_scale[i] + y_scaler_mean[i];
    }
}

static void run_cls_head(float h_b[HIDDEN], float cls_out[OUT_CLS]) {
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16
    float cls_buf[HEAD_HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=cls_buf cyclic factor=16

    // CLS Head Layer 0 (512 -> 256), 16 accumulators II=4
    CLS_HEAD_L0:
    for (int i = 0; i < HEAD_HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        CLS_H0_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 0] * h_b[j+ 0];
            s1  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 1] * h_b[j+ 1];
            s2  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 2] * h_b[j+ 2];
            s3  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 3] * h_b[j+ 3];
            s4  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 4] * h_b[j+ 4];
            s5  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 5] * h_b[j+ 5];
            s6  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 6] * h_b[j+ 6];
            s7  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 7] * h_b[j+ 7];
            s8  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 8] * h_b[j+ 8];
            s9  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 9] * h_b[j+ 9];
            s10 += (float)cls_head_0_weight_q[i*HIDDEN+j+10] * h_b[j+10];
            s11 += (float)cls_head_0_weight_q[i*HIDDEN+j+11] * h_b[j+11];
            s12 += (float)cls_head_0_weight_q[i*HIDDEN+j+12] * h_b[j+12];
            s13 += (float)cls_head_0_weight_q[i*HIDDEN+j+13] * h_b[j+13];
            s14 += (float)cls_head_0_weight_q[i*HIDDEN+j+14] * h_b[j+14];
            s15 += (float)cls_head_0_weight_q[i*HIDDEN+j+15] * h_b[j+15];
        }
        cls_buf[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * cls_head_0_qscale + cls_head_0_bias[i]);
    }

    // CLS Head Layer 1 (256 -> 6), 16 accumulators II=4
    CLS_HEAD_L1:
    for (int i = 0; i < OUT_CLS; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        CLS_H1_MAC:
        for (int j = 0; j < HEAD_HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 0] * cls_buf[j+ 0];
            s1  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 1] * cls_buf[j+ 1];
            s2  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 2] * cls_buf[j+ 2];
            s3  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 3] * cls_buf[j+ 3];
            s4  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 4] * cls_buf[j+ 4];
            s5  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 5] * cls_buf[j+ 5];
            s6  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 6] * cls_buf[j+ 6];
            s7  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 7] * cls_buf[j+ 7];
            s8  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 8] * cls_buf[j+ 8];
            s9  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 9] * cls_buf[j+ 9];
            s10 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+10] * cls_buf[j+10];
            s11 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+11] * cls_buf[j+11];
            s12 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+12] * cls_buf[j+12];
            s13 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+13] * cls_buf[j+13];
            s14 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+14] * cls_buf[j+14];
            s15 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+15] * cls_buf[j+15];
        }
        cls_out[i] = (s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * cls_head_1_qscale + cls_head_1_bias[i];
    }
}

// Canonical DATAFLOW wrapper: only function calls at top level -> no HLS 214-114 warnings.
static void run_heads(
    float h_b[HIDDEN],
    float reg_out[OUT_REG],
    float cls_out[OUT_CLS]
) {
    #pragma HLS DATAFLOW
    run_reg_head(h_b, reg_out);
    run_cls_head(h_b, cls_out);
}

void pb_predict(
    hls::stream<axis_pkt_t> &input_stream,
    hls::stream<axis_pkt_t> &output_stream
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Registers (fully partitioned -> flip-flops)
    float input_raw[IN_DIM];
    float input_scaled[IN_DIM];
    float reg_out[OUT_REG];
    float cls_out[OUT_CLS];
    #pragma HLS ARRAY_PARTITION variable=input_raw    complete
    #pragma HLS ARRAY_PARTITION variable=input_scaled complete
    #pragma HLS ARRAY_PARTITION variable=reg_out      complete
    #pragma HLS ARRAY_PARTITION variable=cls_out      complete

    // Activation buffers: cyclic=16 to support 16 simultaneous reads per MAC iter.
    // After partition into 16 banks each bank is tiny -> HLS uses FFs.
    float h_a[HIDDEN];
    float h_b[HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=h_a cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16

    // Large weights -> BRAM rom_2p, cyclic=16
    // 16 banks -> 16 concurrent reads per cycle to match s0..s15 partial sums.
    #pragma HLS BIND_STORAGE variable=trunk_1_weight_q    type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=reg_head_0_weight_q type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=cls_head_0_weight_q type=rom_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=trunk_1_weight_q    cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=reg_head_0_weight_q cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=cls_head_0_weight_q cyclic factor=16

    // Small weights -> LUTRAM
    #pragma HLS BIND_STORAGE variable=trunk_0_weight_q    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_1_weight_q type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_1_weight_q type=rom_1p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=trunk_0_weight_q    cyclic factor=6
    #pragma HLS ARRAY_PARTITION variable=reg_head_1_weight_q cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=cls_head_1_weight_q cyclic factor=16

    #pragma HLS BIND_STORAGE variable=trunk_0_bias    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=trunk_1_bias    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_0_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_1_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_0_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_1_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=x_scaler_mean   type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=x_scaler_scale  type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=y_scaler_mean   type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=y_scaler_scale  type=rom_1p impl=lutram

    // STAGE 1: AXI-Stream input
    READ_INPUT:
    for (int i = 0; i < IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt = input_stream.read();
        float_uint32 conv;
        conv.u = pkt.data;
        input_raw[i] = conv.f;
    }

    // STAGE 2: StandardScaler
    SCALE_INPUT:
    for (int i = 0; i < IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        input_scaled[i] = (input_raw[i] - x_scaler_mean[i]) / x_scaler_scale[i];
    }

    // STAGE 3: Trunk Layer 0 (6 -> 512)
    // Outer loop pipelined II=1, inner loop fully unrolled (IN_DIM=6).
    // No loop-carried accumulator dependency -> II=1 achievable.
    // 6 parallel fmuls + adder-tree reduction, critical path < 10 ns.
    LAYER0:
    for (int i = 0; i < HIDDEN; i++) {
        #pragma HLS PIPELINE
        float sum = 0.0f;
        L0_MAC:
        for (int j = 0; j < IN_DIM; j++) {
            #pragma HLS UNROLL
            sum += (float)trunk_0_weight_q[i * IN_DIM + j] * input_scaled[j];
        }
        h_a[i] = relu6(sum * trunk_0_qscale + trunk_0_bias[i]);
    }

    // STAGE 4: Trunk Layer 1 (512 -> 512)
    // 16 independent partial accumulators (s0..s15), j+=16: 32 iterations.
    // II=4; cyclic=16 BRAM provides 16 concurrent reads per cycle.
    // Float fadd latency ~11 cycles limits minimum achievable II to ~4 with 16 chains.
    LAYER1:
    for (int i = 0; i < HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        L1_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)trunk_1_weight_q[i*HIDDEN+j+ 0] * h_a[j+ 0];
            s1  += (float)trunk_1_weight_q[i*HIDDEN+j+ 1] * h_a[j+ 1];
            s2  += (float)trunk_1_weight_q[i*HIDDEN+j+ 2] * h_a[j+ 2];
            s3  += (float)trunk_1_weight_q[i*HIDDEN+j+ 3] * h_a[j+ 3];
            s4  += (float)trunk_1_weight_q[i*HIDDEN+j+ 4] * h_a[j+ 4];
            s5  += (float)trunk_1_weight_q[i*HIDDEN+j+ 5] * h_a[j+ 5];
            s6  += (float)trunk_1_weight_q[i*HIDDEN+j+ 6] * h_a[j+ 6];
            s7  += (float)trunk_1_weight_q[i*HIDDEN+j+ 7] * h_a[j+ 7];
            s8  += (float)trunk_1_weight_q[i*HIDDEN+j+ 8] * h_a[j+ 8];
            s9  += (float)trunk_1_weight_q[i*HIDDEN+j+ 9] * h_a[j+ 9];
            s10 += (float)trunk_1_weight_q[i*HIDDEN+j+10] * h_a[j+10];
            s11 += (float)trunk_1_weight_q[i*HIDDEN+j+11] * h_a[j+11];
            s12 += (float)trunk_1_weight_q[i*HIDDEN+j+12] * h_a[j+12];
            s13 += (float)trunk_1_weight_q[i*HIDDEN+j+13] * h_a[j+13];
            s14 += (float)trunk_1_weight_q[i*HIDDEN+j+14] * h_a[j+14];
            s15 += (float)trunk_1_weight_q[i*HIDDEN+j+15] * h_a[j+15];
        }
        h_b[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * trunk_1_qscale + trunk_1_bias[i]);
    }

    // STAGE 5+6: REG and CLS heads in parallel via DATAFLOW
    run_heads(h_b, reg_out, cls_out);

    // STAGE 7: AXI-Stream output
    WRITE_REG:
    for (int i = 0; i < OUT_REG; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt;
        float_uint32 conv;
        conv.f = reg_out[i];
        pkt.data = conv.u;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = 0;
        output_stream.write(pkt);
    }

    WRITE_CLS:
    for (int i = 0; i < OUT_CLS; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt;
        float_uint32 conv;
        conv.f = cls_out[i];
        pkt.data = conv.u;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = (i == OUT_CLS - 1) ? 1 : 0;
        output_stream.write(pkt);
    }
}
