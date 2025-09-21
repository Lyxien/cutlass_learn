# cutlass_learn

## cutlass的安装（RTX 5060Ti）

conda create -n mycutlass python=3.12 -y
conda activate mycutlass
conda install -c nvidia cuda-toolkit=12.8 -y

mkdir build && cd build
export CUDACXX=/home/li/anaconda3/envs/mycutlass/bin/nvcc
cmake .. -DCUTLASS_NVCC_ARCHS=120
