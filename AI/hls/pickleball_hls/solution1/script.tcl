############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project pickleball_hls
set_top pb_predict
add_files pickleball_model.cpp
add_files pickleball_model.h
add_files weights.h
add_files -tb tb_pickleball.cpp -cflags "-I. -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb test_vectors.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xczu3eg-sbva484-1-e}
create_clock -period 10 -name default
config_interface -m_axi_latency 0
config_export -description {PickleballNet MTL MLP - 3x512 ReLU6 BN-fused} -display_name PickleballNet -format ip_catalog -rtl verilog -vendor cg4002 -version 4.0 -vivado_clock 10
source "./pickleball_hls/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -rtl verilog -format ip_catalog
