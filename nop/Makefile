hw: clean
	v++ -c nop.cl -o nop.xo --kernel nop --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg
	v++ -l nop.xo -o nop.xclbin --target hw --platform xilinx_u200_xdma_201830_2 --config u200.cfg

.PHONY: clean
clean:
	rm -rf $(EXE) nop.xclbin* v++_* nop.xo* nop.ltx _x xcd.log xrc.log emconfig.json xilinx_u200_xdma_201830_2-0-* emulation_debug.log .ipcache/ .run/ .Xil/
