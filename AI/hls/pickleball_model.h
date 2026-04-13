#ifndef PICKLEBALL_MODEL_H
#define PICKLEBALL_MODEL_H

#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <ap_fixed.h>

// PickleballNet HLS Ultra96 ZU3EG
// Input(6) -> Scaler -> 2x(Linear+ReLU6) -> RegHead(2) + ClsHead(2)
// AXI-Stream: 6 float32 in, 12 float32 out (6 reg + 6 cls logits)
//
// Resource targets (ZU3EG: 70560 LUT / 216 BRAM36K / 360 DSP):
//   BRAM: ~134 BRAM36K (62%) -- trunk_1 + reg/cls_head_0 weights
//   LUTRAM: small weights + all activation buffers
//   Pipeline: LAYER0 II=1 (unrolled), LAYER1/heads II=4 (16 accumulators), heads DATAFLOW parallel

#define IN_DIM        6
#define HIDDEN        512
#define HEAD_HIDDEN   256
#define N_LAYERS      2
#define OUT_REG       6
#define OUT_CLS       6
#define OUT_TOTAL     (OUT_REG + OUT_CLS)
#define UNROLL_FACTOR 4

typedef ap_axis<32, 0, 0, 0> axis_pkt_t;

void pb_predict(
    hls::stream<axis_pkt_t> &input_stream,
    hls::stream<axis_pkt_t> &output_stream
);

#endif // PICKLEBALL_MODEL_H
