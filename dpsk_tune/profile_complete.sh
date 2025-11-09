#!/bin/bash
#SBATCH --job-name=dpsk
#SBATCH --partition=hpcai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:8
#SBATCH --output=dpsk_%j.out
#SBATCH --error=dpsk_%j.err

source /home/apacsc40/anaconda3/etc/profile.d/conda.sh
conda activate xbx_dpsk

source /scratch/public/intel/oneapi/setvars.sh

export CUDA_HOME=/scratch/public/nvidia/cuda/cuda-12.4
export PATH=$CUDA_HOME/bin:/home/apacsc40/nsys_2024.2/pkg/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export NVCC=$CUDA_HOME/bin/nvcc
export TRITON_CC=/usr/bin/gcc

# ❌ 删掉这些会“抢位”的 include 覆盖
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset LIBRARY_PATH


export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

export DIST_INIT_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg_${SLURM_JOB_ID}"
export FLASHINFER_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flashinfer_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_CACHE_DIR"

export SGLANG_TORCH_PROFILER_DIR="/work/apacsc40/profiler/${SLURM_JOB_ID}"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"


time mpirun -np 2 \
  -ppn 1 \
  -genv OMP_NUM_THREADS 112 \
  -genv NCCL_DEBUG INFO \
  -genv DIST_INIT_ADDR ${DIST_INIT_ADDR} \
  -genv TRITON_CACHE_DIR ${TRITON_CACHE_DIR} \
  -genv XDG_CACHE_HOME ${XDG_CACHE_HOME} \
  -genv SGLANG_TORCH_PROFILER_DIR ${SGLANG_TORCH_PROFILER_DIR} \
  -genv FLASHINFER_CACHE_DIR ${FLASHINFER_CACHE_DIR} \
  bash -c 'export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}/${OMPI_COMM_WORLD_RANK}; \
  export XDG_CACHE_HOME=${XDG_CACHE_HOME}/${OMPI_COMM_WORLD_RANK}; \
  export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR}/${OMPI_COMM_WORLD_RANK}; \
  export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}/${OMPI_COMM_WORLD_RANK}; \
  mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$FLASHINFER_CACHE_DIR" "$SGLANG_TORCH_PROFILER_DIR"; \
  /home/apacsc40/nsys_2024.2/pkg/bin/nsys profile --stats=true --trace-fork-before-exec=true --cuda-graph-trace=node --trace=cuda,nvtx --cuda-memory-usage=true \
  python3 -u \
  -m sglang.bench_offline_throughput \
  --profile \
  --model-path deepseek-ai/DeepSeek-R1 \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --load-format dummy \
  --seed 2025 \
  --dtype bfloat16 \
  --tp 16 \
  --nnodes 2 \
  --trust-remote-code \
  --dist-init-addr ${DIST_INIT_ADDR}:5000 \
  --node-rank ${PMI_RANK} '
