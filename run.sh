#!/bin/bash
# Launcher for rccl_debug.py via torchrun.
#
# Single node:
#   ./run.sh                                  # all 8 GPUs, all ranks
#   ./run.sh --ranks 0,3,4,7                  # forward args to rccl_debug.py
#   NPROC_PER_NODE=4 ./run.sh
#
# Multi-node (Slurm: one task per node, torchrun fans out to local GPUs):
#   srun --ntasks-per-node=1 ./run.sh [args]
#
# Env overrides:
#   ITERATIONS       how many back-to-back runs (default: 10)
#                    iter 1 = --first-run 1 (generate inputs + save reference)
#                    iter 2..N = --first-run 0 (compare against saved reference)
#   NPROC_PER_NODE   GPUs per node              (default: 8)
#   NNODES           number of nodes            (default: $SLURM_NNODES or 1)
#   NODE_RANK        this node's rank           (default: $SLURM_NODEID or 0)
#   MASTER_ADDR      master node hostname/IP    (default: first SLURM hostname or 127.0.0.1)
#   MASTER_PORT      torchrun rendezvous port   (default: 29500)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ITERATIONS="${ITERATIONS:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    else
        MASTER_ADDR="127.0.0.1"
    fi
fi

# RCCL / NCCL fabric configuration (AMD MI355 + RoCE NICs + amd-anp plugin).
# Defaults are tuned for the current cluster:
#   * The 8 GPU-side RoCE NICs are listed explicitly below (port 1 each).
#   * GID index 1 is the routable IPv6 ULA on the GPU NICs (idx 3 doesn't exist).
#   * The data NIC enp193s0f0np0 blocks inter-host TCP, so bootstrap / sockets
#     must use the mgmt NIC enp193s0f1np1.
# Use `${VAR:-default}` so callers can override any of these from the shell.
# export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep9s0:1,rocep25s0:1,rocep105s0:1,rocep121s0:1,rocep137s0:1,rocep153s0:1,rocep233s0:1,rocep249s0:1}"
export IP_INTERFACE="${IP_INTERFACE:-enp193s0f1np1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f1np1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0f1np1}"

export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-librccl-anp.so}"
export NCCL_IB_TC="${NCCL_IB_TC:-104}"
export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-192}"
export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
export NCCL_MAX_P2P_CHANNELS="${NCCL_MAX_P2P_CHANNELS:-56}"
export NCCL_IB_SL="${NCCL_IB_SL:-0}"
export NET_OPTIONAL_RECV_COMPLETION="${NET_OPTIONAL_RECV_COMPLETION:-1}"
export NCCL_IB_USE_INLINE="${NCCL_IB_USE_INLINE:-1}"
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING="${RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING:-0}"
export NCCL_GDR_FLUSH_DISABLE="${NCCL_GDR_FLUSH_DISABLE:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:/workspace/rccl/build/release:/workspace/amd-anp/build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir -p logs
LOG_FILE="logs/rank${NODE_RANK}.log"
: > "${LOG_FILE}"

echo "[run.sh] node=${NODE_RANK}/${NNODES} gpus=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} iterations=${ITERATIONS}" | tee -a "${LOG_FILE}"

for i in $(seq 1 "${ITERATIONS}"); do
    if [ "${i}" -eq 1 ]; then
        FIRST_RUN=1
    else
        FIRST_RUN=0
    fi

    printf '\n===== [run.sh] iteration %d/%d (--first-run %d) =====\n' \
        "${i}" "${ITERATIONS}" "${FIRST_RUN}" | tee -a "${LOG_FILE}"

    if ! torchrun \
            --nnodes="${NNODES}" \
            --nproc-per-node="${NPROC_PER_NODE}" \
            --node-rank="${NODE_RANK}" \
            --master-addr="${MASTER_ADDR}" \
            --master-port="${MASTER_PORT}" \
            "${SCRIPT_DIR}/rccl_debug.py" --first-run "${FIRST_RUN}" "$@" \
            2>&1 | tee -a "${LOG_FILE}"; then
        echo "[run.sh] iteration ${i} FAILED, aborting" | tee -a "${LOG_FILE}"
        exit 1
    fi
done

echo "[run.sh] all ${ITERATIONS} iterations completed" | tee -a "${LOG_FILE}"
