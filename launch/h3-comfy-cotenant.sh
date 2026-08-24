#!/usr/bin/env bash
# MiniMax-H3 (ComfyUI) as a CO-RESIDENT render lane next to a DeepSeek-V4-Flash TP=4 rank on a
# DGX Spark (GB10). This is the "draft" tier: the int8 H3 build, co-tenant with DS4 LIVE.
#
# The three flags that make it fit next to DS4 (KEYS' budget, container half):
#   --reserve-vram 48        caps H3's grab on the UNIFIED pool so DS4's KV survives a render.
#   --oom-score-adj 800      makes THIS container the kernel's OOM victim (H3 dies, never DS4).
#   --disable-pinned-memory  stops ComfyUI page-locking ~90% of host RAM into the shared pool.
#
# Plus the entrypoint override and the PATH-PRESERVING mount (see the symlink trap below).
#
# SCRUBBED: image tag, model dir, and username are placeholders. On the GB10, --reserve-vram is
# CORRECT (unified memory). On a discrete-GPU RAM-poor box (e.g. a 3090) it is the OPPOSITE
# advice — do not carry this flag across box types. See GB10-HYGIENE.md.
#
# usage: h3-comfy-cotenant.sh <NODE_INDEX>      # 0..3, just names the container per node

set -uo pipefail
NODE_INDEX="${1:?usage: h3-comfy-cotenant.sh <NODE_INDEX 0..3>}"

# --- rig-specific placeholders -------------------------------------------------------------
H3_IMAGE="<YOUR_COMFYUI_H3_IMAGE>"       # a ComfyUI image w/ native MiniMax-H3 (ComfyUI >= v0.30.1)
                                         # built on a torch/cu130 base (do NOT pip into host globals).
H3_PY="/opt/env/bin/python3"             # the python INSIDE that image (the image ENTRYPOINT is `sleep`)
H3_USER="<user>"                         # the host user that owns the H3 tree (for the path-preserving mount)
H3_HOME="/home/${H3_USER}/h3"            # H3 home on the HOST. Mounted at the SAME absolute path (trap below).
H3_MODELS_NFS="<H3_MODELS_NFS_EXPORT>"   # e.g. /srv/h3-models — the single NFS export the model
                                         # symlinks point into (weights are NOT duplicated per node).
PORT="8188"

# ⚠️ THE SYMLINK TRAP: the files under $H3_HOME/ComfyUI/models/<type>/*.safetensors are
# ABSOLUTE-path symlinks into $H3_MODELS_NFS. If you mount H3_HOME at a DIFFERENT container path,
# every symlink dangles, the loaders show empty, and H3 dies in ~0.01s. So mount it at the SAME
# absolute path and set -w to the ComfyUI dir underneath it. Make sure the NFS export is actually
# mounted + readable on this node first.

docker rm -f "comfy-h3-${NODE_INDEX}" 2>/dev/null || true

docker run -d --name "comfy-h3-${NODE_INDEX}" \
  --gpus all --network host --ipc host --shm-size 8g \
  --oom-score-adj 800 \
  --restart no \
  -v "${H3_HOME}:${H3_HOME}" \
  -v "${H3_MODELS_NFS}:${H3_HOME}/ComfyUI/models/_nfs:ro" \
  -w "${H3_HOME}/ComfyUI" \
  --entrypoint "${H3_PY}" \
  "${H3_IMAGE}" \
  main.py --listen 0.0.0.0 --port "${PORT}" \
    --disable-pinned-memory --fp16-intermediates \
    --reserve-vram 48
    # optional (KEYS H3 launcher; may be a custom flag on your build): --vram-headroom 10

echo "launched comfy-h3-${NODE_INDEX} on :${PORT}  (reserve-vram 48, oom-score-adj 800) rc=$?"
sleep 2
docker ps --format '{{.Names}} | {{.Status}}' | grep "comfy-h3-${NODE_INDEX}" \
  || echo "WARN: not in ps (docker logs comfy-h3-${NODE_INDEX})"

# ---------------------------------------------------------------------------
# BEFORE THIS SCRIPT, on this node:  DS4 must already be UP (whoever loads first wins memory),
# then drop caches:   sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
#
# One H3 job per box. RestartPolicy is `no` on purpose: `unless-stopped` would let H3 start
# BEFORE DS4 on a reboot and win the memory race. Teardown order is the reverse of bring-up:
# stop H3 on every node, THEN DS4.
#
# FINALS (bf16) DO NOT run co-tenant — the bf16 model is ~40 GB and is too big to share a Spark
# with DS4. For a polished final, pause DS4 and fan bf16 across the freed Sparks (~3x). See
# FARM-MODE.md. This script is the int8 DRAFT co-tenant only.
# ---------------------------------------------------------------------------
