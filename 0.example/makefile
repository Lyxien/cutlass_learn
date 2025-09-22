# conda activate mycutlass
export CUTLASS_ROOT=/home/li/cutlass

0.example:
	nvcc 0.example.cu -I$$CUTLASS_ROOT/include -I$$CUTLASS_ROOT/tools/util/include -std=c++17 -O3 -o 0.example