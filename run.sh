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
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export NVCC=$CUDA_HOME/bin/nvcc
export TRITON_CC=nvcc
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

export DIST_INIT_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg_${SLURM_JOB_ID}"
export FLASHINFER_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/flashinfer_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_CACHE_DIR"

# export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG_SUBSYS=ALL


# 1023 haibin changed
# 在你的 run.sh 里（conda等加载之后）加上：
export NCCL_IB_HCA=mlx5_0:1
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_CROSS_NIC=1      # 跨网卡也能凑，但HCA仍会按 NCCL_IB_HCA 限定

#export NCCL_ALGO=RING


export NCCL_PROTO=LL128

iblinkinfo
# 检查GPU内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

time mpirun -np 2 \
  -ppn 1 \
  -genv OMP_NUM_THREADS 56 \
  -genv NCCL_DEBUG INFO \
  -genv DIST_INIT_ADDR ${DIST_INIT_ADDR} \
  -genv TRITON_CACHE_DIR ${TRITON_CACHE_DIR} \
  -genv XDG_CACHE_HOME ${XDG_CACHE_HOME} \
  -genv FLASHINFER_CACHE_DIR ${FLASHINFER_CACHE_DIR} \
  bash -c '
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}/${OMPI_COMM_WORLD_RANK};
    export XDG_CACHE_HOME=${XDG_CACHE_HOME}/${OMPI_COMM_WORLD_RANK};
    export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR}/${OMPI_COMM_WORLD_RANK};
    mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$FLASHINFER_CACHE_DIR";
    python3 -u \
      -m sglang.bench_offline_throughput \
      --model-path deepseek-ai/DeepSeek-R1 \
      --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 2000 \
      --load-format dummy \
      --seed 2025 \
      --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
      --dtype bfloat16 \
      --tp 16 \
      --enable-dp-attention \
      --pipeline-parallel-size 1 \
      --data-parallel-size 2 \
      --nnodes 2 \
      --trust-remote-code \
      --do-not-exit --sleep-on-idle --dist-timeout 3600 \
      --dist-init-addr ${DIST_INIT_ADDR}:5000 \
      --node-rank ${PMI_RANK} \
      --result-filename ${SLURM_SUBMIT_DIR:-$HOME}/bench_${SLURM_JOB_ID}.json
  '
