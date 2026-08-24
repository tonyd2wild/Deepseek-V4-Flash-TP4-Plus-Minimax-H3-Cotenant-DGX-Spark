#!/usr/bin/env bash
# DeepSeek-V4-Flash on FOUR DGX Spark (GB10) — vLLM multi-node TP=4, tuned to CO-TENANT with H3.
#
# This is a co-tenancy-flavored variant of the DS4 TP=4 launcher. The full DS4-on-Spark
# derivation (RDMA/NCCL deep-dive, worker-first startup, head-count rule, the C1-C6 speed
# benchmarks, the public image, the "NVFP4 is a mirage / weights are FP8" note) lives in the
# sibling repo — read it first if DS4-on-Spark is new to you:
#
#   https://github.com/tonyd2wild/deepseek-v4-flash-tp4-4x-dgx-spark
#   https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX   (two-Spark base)
#
# What THIS script adds: the gmu / context knobs that make room for a co-resident MiniMax-H3
# render lane on every node (KEYS' co-tenancy budget), plus the strict "DS4 first" load order.
#
# SCRUBBED: uses the PUBLIC DeepSeek-V4-Flash model id and a placeholder image. Our production
# stack runs an internal model path behind a private DSpark-spec image with the k=5 dspark
# drafter + fused-Markov-argmax + the drafter-sizes patch (that stack works fine at TP=4). It is
# NOT published here. Fill in <PLACEHOLDERS> for your rig.
#
# Run this SAME script on ALL FOUR nodes. args:
#   arg1 = NODE_RANK   0 = head (serves :8000) | 1,2,3 = workers (--headless)
#   arg2 = this node's own IP on the switched RoCE fabric (e.g. 192.168.192.2)
# STARTUP ORDER: workers (ranks 3,2,1) FIRST, then head (rank 0).

set -uo pipefail
NODE_RANK="${1:?usage: ds4-tp4-cotenant-launch.sh <0|1|2|3> <THIS_NODE_FABRIC_IP>}"
HOST_IP="${2:?usage: ds4-tp4-cotenant-launch.sh <0|1|2|3> <THIS_NODE_FABRIC_IP>}"
HEADLESS_FLAG=""
[ "$NODE_RANK" != "0" ] && HEADLESS_FLAG="--headless"

# --- rig-specific placeholders -------------------------------------------------------------
DS4_IMAGE="<YOUR_DS4_VLLM_GB10_IMAGE>"   # e.g. the public aidendle94/sparkrun-vllm-ds4-gb10:production-ready (see sibling)
DS4_MODEL="deepseek-ai/DeepSeek-V4-Flash"  # public model id; swap for your local path if pre-downloaded
MASTER_ADDR="<HEAD_FABRIC_IP>"           # rank 0's IP on the switched RoCE fabric (rendezvous master)
MASTER_PORT="29640"
FABRIC_SUBNET="192.168.192.0/24"         # the subnet all four Sparks share on the switch
CTRL_IF="enp1s0f0np0"                    # TCP control-plane NIC (confirm: ip link)
ROCE_HCA="rocep1s0f0"                    # the RoCE HCA on the switched fabric (confirm: ibstatus)

# --- CO-TENANCY BUDGET (KEYS) --------------------------------------------------------------
# At TP=4 each node holds only ~38 GiB of weights (1/4 the model) vs ~80 GiB at TP=2, so you
# have real headroom and can co-tenant H3 without dropping gmu at all:
#
#   GMU=0.78  CTX=300000   -> ~6.9M-token KV pool, ~30 GB/node still free for the H3 int8 draft.
#                             This is the comfortable default; H3 fits in the leftover slack.
#
# KEYS' original razor-edge budget (from the TP=2 era, where the model ate ~80 GiB/node and H3
# lived inside a 0.02 gmu delta) was gmu 0.76 at 909312 (~888K) context. You do NOT need to go
# that tight at TP=4 — but if you want to deliberately reserve MORE room for H3 (or push more
# concurrent render lanes), drop gmu the KEYS way and shorten context to afford it:
#
#   GMU=0.76  CTX=909312    -> KEYS' budget: ~888K ctx, extra headroom carved out for H3.
#
# Fleet hard cap is gmu 0.85; never exceed it. Shorten CONTEXT to afford a lower GMU, don't just
# lower gmu at a fixed context (see the KV/util curve in the README).
GMU="${GMU:-0.78}"
CTX="${CTX:-300000}"
SEQS="${SEQS:-6}"

docker rm -f vllm_ds4_tp4 2>/dev/null || true

docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --name vllm_ds4_tp4 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$ROCE_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ADDR_RANGE="$FABRIC_SUBNET" -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_SOCKET_IFNAME="$CTRL_IF" -e GLOO_SOCKET_IFNAME="$CTRL_IF" -e TP_SOCKET_IFNAME="$CTRL_IF" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  "$DS4_IMAGE" \
  -lc "exec vllm serve $DS4_MODEL \
    --served-model-name deepseek-v4-flash-spark deepseek-v4-flash \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len $CTX --max-num-seqs $SEQS --max-num-batched-tokens 8192 \
    --gpu-memory-utilization $GMU --enable-prefix-caching \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' \
    --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --nnodes 4 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT $HEADLESS_FLAG"

echo "launched vllm_ds4_tp4 rank=$NODE_RANK host=$HOST_IP gmu=$GMU ctx=$CTX headless='$HEADLESS_FLAG' rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_ds4_tp4 || echo "WARN: not in ps (docker logs vllm_ds4_tp4)"

# ---------------------------------------------------------------------------
# COPY-PASTE (worker-first: ranks 3, 2, 1, then head 0). drop_caches on ALL nodes first!
#
#   sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # on every node, BEFORE launching
#   ./ds4-tp4-cotenant-launch.sh 3 192.168.192.4
#   ./ds4-tp4-cotenant-launch.sh 2 192.168.192.3
#   ./ds4-tp4-cotenant-launch.sh 1 192.168.192.2
#   # wait ~20s, then the HEAD:
#   ./ds4-tp4-cotenant-launch.sh 0 192.168.192.1
#   docker logs -f vllm_ds4_tp4        # on the head; wait for "Application startup complete"
#   curl http://<HEAD_NODE_IP>:8000/v1/models
#
# THEN drop_caches again on the render nodes and bring up H3 (h3-comfy-cotenant.sh).
# JSON args are single-quoted so they survive the container's inner `bash -lc` — keep them that
# way or all four nodes exit(2) instantly. Fresh `docker run` only; DS4 does NOT survive restart.
# ---------------------------------------------------------------------------
