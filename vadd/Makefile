# Compiler
CXX := g++

# Compilation flags
CXXFLAGS := -Wall -g -std=c++11

# EXE
EXE := app.exe

# Directories
INC_DIRS := ${XILINX_XRT}/include/
LIB_DIRS := ${XILINX_XRT}/lib/

# Files
SRCS := host.cpp
LIBS := OpenCL pthread rt stdc++

sw_emu: clean exe_build
	emconfigutil --platform xilinx_u200_xdma_201830_2 --nd 1
	v++ -c vadd.cpp -o vadd.xo --kernel vadd --target sw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target sw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	env XCL_EMULATION_MODE=sw_emu ./app.exe

hw_emu: clean exe_build
	emconfigutil --platform xilinx_u200_xdma_201830_2 --nd 1
	v++ -c vadd.cpp -o vadd.xo --kernel vadd --target hw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target hw_emu --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	env XCL_EMULATION_MODE=hw_emu ./app.exe

hw: clean exe_build
	v++ -c vadd.cpp -o vadd.xo --kernel vadd --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l vadd.xo -o vadd.xclbin --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	srun -u -p alveo -w alveo01 ./app.exe

exe_build:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(EXE) $(foreach D,$(INC_DIRS),-I$D) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)

.PHONY: clean
clean:
	rm -rf $(EXE) vadd.xclbin* v++_* vadd.xo* vadd.ltx _x xcd.log xrc.log emconfig.json xilinx_u200_xdma_201830_2-0-* emulation_debug.log .ipcache/ .run/ .Xil/
