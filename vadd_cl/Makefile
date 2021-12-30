# Compiler
CC := gcc

# Compilation flags
CFLAGS := -O3 -Wall -Wextra -g -std=c99 -fopenmp

# EXE
EXE := test_vecadd.exe

# Directories
INC_DIRS := ${XILINX_XRT}/include/
LIB_DIRS := ${XILINX_XRT}/lib/

# Files
SRCS := main.c cl_utils.c
LIBS := OpenCL rt

sw_emu: clean exe_build
	emconfigutil --platform xilinx_u200_xdma_201830_2 --nd 1
	v++ -c vadd.cl -o vadd.xo --kernel vadd --target sw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target sw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	env XCL_EMULATION_MODE=sw_emu ./$(EXE) vadd.xclbin 10

hw_emu: clean exe_build
	emconfigutil --platform xilinx_u200_xdma_201830_2 --nd 1
	v++ -c vadd.cl -o vadd.xo --kernel vadd --target hw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target hw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	env XCL_EMULATION_MODE=hw_emu ./$(EXE) vadd.xclbin 10

hw: clean exe_build
	v++ -c vadd.cl -o vadd.xo --kernel vadd --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	srun -u -p alveo -w alveo01 ./$(EXE) vadd.xclbin 10

exe_build:
	$(CC) $(CFLAGS) $(SRCS) -o $(EXE) $(foreach D,$(INC_DIRS),-I$D) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)

.PHONY: clean
clean:
	rm -rf $(EXE) vadd.xclbin* v++_* vadd.xo* vadd.ltx _x xcd.log xrc.log emconfig.json xilinx_u200_xdma_201830_2-0-* emulation_debug.log .ipcache/ .run/ .Xil/
