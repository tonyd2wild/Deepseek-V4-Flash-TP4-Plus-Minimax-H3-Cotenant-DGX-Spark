#!/usr/bin/env bash
# Ordered bring-up for DS4 (TP=4) + MiniMax-H3 co-tenancy on four DGX Sparks (GB10).
#
# This encodes the ONE thing you cannot get wrong: LOAD ORDER. On unified memory "whoever loads
# first wins the memory," so this is a correctness requirement, not etiquette.
#
#   drop_caches (all 4)  ->  DS4 TP=4 (workers first, then head)  ->  drop_caches  ->  H3 lanes
#
# This is a DRIVER you run FROM a control host that can SSH to all four Sparks. It is intentionally
# explicit rather than clever — read it, then run the pieces by hand the first time. Fill in the
# node list + the fabric IPs for your rig. SCRUBBED: no real hostnames/IPs.

set -uo pipefail

# --- your four Sparks: "ssh-target fabric-ip node-rank" (rank 0 = head/API) ----------------
# Replace ssh targets and fabric IPs with YOUR rig. Fabric IPs are on the switched RoCE subnet.
NODES=(
  "spark0  192.168.192.1  0"   # HEAD (serves DS4 API on :8000)
  "spark1  192.168.192.2  1"
  "spark2  192.168.192.3  2"
  "spark3  192.168.192.4  3"
)
DS4_LAUNCH="/path/to/ds4-tp4-cotenant-launch.sh"   # where this repo's script lives ON each node
H3_LAUNCH="/path/to/h3-comfy-cotenant.sh"
RENDER_NODES=(spark0 spark1 spark2 spark3)         # which nodes should also run an H3 render lane
HEAD_SSH="spark0"
DROP="sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"

echo "== 1) drop_caches on ALL four nodes (page cache starves the UMA allocator) =="
for entry in "${NODES[@]}"; do read -r ssh_t _ _ <<<"$entry"; ssh "$ssh_t" "$DROP"; done

echo "== 2) DS4 TP=4 — WORKERS FIRST (ranks 3,2,1), then HEAD (rank 0) =="
# workers first, in reverse rank order
for entry in "${NODES[@]}"; do
  read -r ssh_t fab rank <<<"$entry"
  [ "$rank" = "0" ] && continue
  echo "  -> worker rank $rank on $ssh_t ($fab)"
  ssh "$ssh_t" "bash $DS4_LAUNCH $rank $fab"
done
echo "  ...waiting ~20s for workers to listen..."
# (foreground sleep is fine here; adjust to taste)
for i in $(seq 1 20); do sleep 1; done
for entry in "${NODES[@]}"; do
  read -r ssh_t fab rank <<<"$entry"
  [ "$rank" = "0" ] || continue
  echo "  -> HEAD rank 0 on $ssh_t ($fab)"
  ssh "$ssh_t" "bash $DS4_LAUNCH 0 $fab"
done

echo "== 3) wait for DS4 to serve, then VERIFY a real completion =="
echo "  watch:  ssh $HEAD_SSH 'docker logs -f vllm_ds4_tp4'   # wait for 'Application startup complete'"
echo "  verify: curl http://<HEAD_NODE_IP>:8000/v1/models"
echo "  (do not proceed to H3 until DS4 answers a real request — it must win the memory first)"
read -r -p "  DS4 verified up? press ENTER to bring up H3, or Ctrl-C to stop... " _

echo "== 4) drop_caches AGAIN on render nodes (loading DS4's weights refilled the page cache) =="
for n in "${RENDER_NODES[@]}"; do ssh "$n" "$DROP"; done

echo "== 5) H3 render lanes — one per render node, co-resident with that node's DS4 rank =="
idx=0
for n in "${RENDER_NODES[@]}"; do
  echo "  -> H3 lane $idx on $n"
  ssh "$n" "bash $H3_LAUNCH $idx"
  idx=$((idx+1))
done

echo "== done. Teardown is the STRICT REVERSE: H3 down on every node, THEN DS4. =="
echo "   (to restart DS4 you must stop H3 first; DS4 needs a fresh 'docker run', not restart)"
