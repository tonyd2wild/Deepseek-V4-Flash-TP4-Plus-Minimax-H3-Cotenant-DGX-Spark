# DeepSeek-V4-Flash (TP=4, four DGX Sparks) **co-tenant with** MiniMax-H3 image/video

> **Unofficial community recipe.** Not affiliated with, endorsed by, or supported by
> DeepSeek, NVIDIA, MiniMax, the vLLM project, or ComfyUI. All names and trademarks belong to
> their owners. Every number labeled "our measurement" is from our own rig — treat it as
> *relative*, not gospel, and benchmark your own mix. Nothing here is a guarantee it fits on
> your boxes; the whole point is that the memory budget is tight and you have to measure it.

This is the **"we ran a full 671B-class LLM AND a video-diffusion model on the same four
boxes"** recipe. On **four NVIDIA DGX Spark (GB10)** nodes we serve **DeepSeek-V4-Flash** as a
single tensor-parallel group (**TP=4**) with vLLM, and — on the *same four nodes, at the same
time* — run **MiniMax-H3** (omni-modal image/video diffusion, ComfyUI) as a co-resident render
farm. One endpoint for a live agent fleet, plus in-house image/video generation, zero
per-asset cost, all local.

It works because **TP=4 changes the memory math.** At TP=2 the model eats ~80 GB of each
node's ~120 GB unified memory and co-tenancy lives on a razor edge. At TP=4 each node holds
only **¼ of the model (~38 GB)**, which frees roughly **40–50 GB/node** — and *that* is the
room H3 co-tenants in.

---

## Credit — KEYS

**The co-tenancy budget at the heart of this recipe is KEYS' work.** KEYS tuned the deliberate
GPU-memory-utilization drop — **`gmu 0.76` at `909312` (~888K) context** — that carves out the
headroom H3 needs, paired with H3's **`--reserve-vram 48`** cap and an **earlyoom + choom
victim net** so that an H3 render is *always* the memory sacrifice and DeepSeek can **never** be
the process that dies. That gmu-drop budget is what first made **DS4 + H3 co-tenancy fit on the
razor edge at TP=2**; this repo carries KEYS' budget discipline onto the roomier **TP=4**
topology. If you take one thing from this repo, take KEYS' budget. **Thanks, KEYS.**

---

## Why TP=4 is what makes H3 fit (the memory math)

GB10 is **unified CPU+GPU memory (~120–121 GiB/node)** — the page cache, the model weights, and
the KV pool all draw from the *same* pool. So "does H3 fit next to DS4?" is pure arithmetic on
one number per node.

| | **TP=2** (2 Sparks) | **TP=4** (4 Sparks) |
|---|---|---|
| DS4 weights **per node** | **79.51 GiB** (½ the model) | **38.29 GiB** (¼ the model) |
| KV mem per node @ gmu 0.78 | ~11 GiB | **50.65 GiB** |
| GPU KV pool | ~1.46M tokens | **6.90M tokens (~4.7×)** |
| Concurrency @ ~888K/req | ~1.6× | **7.58× (~4.7×)** |
| Free per node beyond vLLM's alloc | razor-thin | **~30 GB+** |
| Boot time | ~12 min | **~5 min** (¼ weights/node) |

The key line: **dropping from ~80 GB to ~38 GB of weights per node frees ~40 GB/node.** At TP=2
that headroom didn't exist — H3 lived inside a **0.02 gmu delta** (see the KV/util curve below),
which is exactly why KEYS' budget had to be so precise. At TP=4 the same H3 render drops into
~30–50 GB of genuine slack. Same H3 workload, far more comfortable.

> **"Whoever loads first wins the memory."** On unified memory the first big allocator to start
> defines the split. Always bring up **DS4 first**, then H3. Teardown is the strict reverse.

### The KV / util curve — why 0.76 and 888K are ONE setting (TP=2 origin)

At TP=2 you do **not** simply lower gmu to make room for H3 — you **shorten context to afford
lowering gmu**:

| context | gmu | KV needed vs available |
|---|---|---|
| 1M (`1048576`) | **0.78 required** | needs 7.54 GiB; 0.76 gives only ~7.38 → **fails** |
| 888K (`909312`) | **0.76 works** | needs ~6.99 GiB; 0.75 gives ~6.76 → **fails** |

Fleet hard cap **gmu 0.85**, never exceed. The **0.02 util delta is exactly where H3 lives** at
TP=2. This is KEYS' budget, measured. At **TP=4** the razor edge is gone: you can run DS4 at its
standalone winner gmu (0.78, ~6.9M-token KV pool) **and** still have ~30 GB/node free for the H3
int8 draft — or apply KEYS' gmu-drop to reserve even more. The *discipline* carries over even
though the pressure doesn't.

---

## Two modes: **int8 "draft" co-tenant** vs **bf16 "finals" farm-mode**

This is the workflow, and it is the most important operational decision in the repo. Full
detail in [`FARM-MODE.md`](FARM-MODE.md).

| | **DRAFT — int8 H3** | **FINALS — bf16 H3** |
|---|---|---|
| Runs **co-tenant with DS4 live?** | ✅ yes — DS4 stays up | ❌ no — model is ~40 GB, too big to share a Spark with DS4 |
| Model | int8 DiT (largest single component ~20.9 GB; Comfy loads/frees sequentially) | full **bf16** (~40 GB resident) |
| Speed | ~8.8 min/span, co-tenant | ~3× faster across the **freed** Sparks |
| Quality | slightly stylized / posterized (that's the int8 tier) | native, properly-real |
| DS4 impact | DS4 keeps serving (a live render ~halves DS4 throughput; idle H3 costs ~6–15%) | **DS4 paused ~20–30 min** (farm mode) |
| Use for | iteration, previews, drafts | the one polished final |

**The play:** iterate in **draft co-tenant** (DeepSeek never drops) → then **one farm-mode
pass** (DeepSeek paused ~20–30 min, H3 fans across all four freed Sparks at ~3× speed) for the
polished final. Pausing DS4 is a human call every time — it drops the agent fleet to a fallback
model for the duration.

---

## Launch order (the whole thing hinges on this)

Because "whoever loads first wins the memory," order is a correctness requirement, not
etiquette. See [`launch/cotenant-bringup.sh`](launch/cotenant-bringup.sh) for the orchestrated
version.

1. **`drop_caches` on all four nodes** — `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
   On unified memory the page cache competes directly with the model and **nothing evicts it
   automatically**; skipping this is the #1 cause of a "CUDA out of memory" at startup on a box
   that looks half-empty. (See [`GB10-HYGIENE.md`](GB10-HYGIENE.md).)
2. **DS4 TP=4 first**, worker-first: ranks 3, 2, 1 (`--headless`), wait ~20 s, then rank 0
   (head). Wait for `Application startup complete` and a real `curl /v1/models`.
   ([`launch/ds4-tp4-cotenant-launch.sh`](launch/ds4-tp4-cotenant-launch.sh))
3. **`drop_caches` again** on the nodes you're about to render on — loading DS4's own weights
   refills the page cache just as thoroughly as copying files would.
4. **H3 second**, one ComfyUI/H3 container per node you want to render on, co-resident with that
   node's DS4 rank. ([`launch/h3-comfy-cotenant.sh`](launch/h3-comfy-cotenant.sh))

**Teardown is the strict reverse:** H3 down on every node, *then* DS4. To restart DS4 you must
stop H3 first.

---

## Topology (as it actually runs)

```
        switched RoCE fabric — 192.168.192.0/24 (200GbE, RoCEv2, MTU 9000, GID idx 3)
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   │              │              │              │
┌──┴───────────┐ ┌┴─────────────┐ ┌┴────────────┐ ┌┴────────────┐
│  Spark 0     │ │  Spark 1     │ │  Spark 2    │ │  Spark 3    │
│  .1  (HEAD)  │ │  .2 (worker) │ │  .3 (worker)│ │  .4 (worker)│
│              │ │              │ │             │ │             │
│  DS4 rank 0  │ │  DS4 rank 1  │ │  DS4 rank 2 │ │  DS4 rank 3 │  ← ONE TP=4 group
│  :8000 API   │ │  --headless  │ │  --headless │ │  --headless │    (¼ weights each)
│              │ │              │ │             │ │             │
│  H3 ComfyUI  │ │  H3 ComfyUI  │ │  H3 ComfyUI │ │  H3 ComfyUI │  ← co-resident render lanes
│  :8188       │ │  :8188       │ │  :8188      │ │  :8188      │    (int8 draft, reserve-vram)
└──────┬───────┘ └──────────────┘ └─────────────┘ └─────────────┘
       │ serves DS4 API
       ▼
  http://<HEAD_NODE_IP>:8000/v1   ← agents / clients   |   :8188 per node ← render jobs
```

Per node = **two containers**: a DS4 TP-rank + a ComfyUI/H3 lane, co-resident. Across the fleet:
**one** DS4 TP=4 lane (four rank containers, rank 0 serves the API) **plus** up to four H3 render
lanes. In our earlier dual-TP2-lane era we ran the same shape — DS4 on all four boxes + H3 on
all four, co-tenant — and it held for days; TP=4 just makes the per-node budget roomier.

---

## Hardware

| Component | Requirement |
|---|---|
| **Compute** | **4× NVIDIA DGX Spark** (GB10 Grace-Blackwell, `sm_121` / arch `12.1a`), one GPU each. |
| **Interconnect** | A **switched** 200GbE RoCE/RDMA fabric — all four Sparks on one subnet (we use `192.168.192.0/24`) through a RoCE switch. TP=4 needs every node to reach every other node (a direct QSFP cable only does TP=2). |
| **Unified memory** | GB10 = **unified CPU+GPU memory (~120–121 GiB/node)**. This is the whole game — weights, KV pool, page cache, and H3 all draw from it. |
| **Disk** | DS4 weights (~150 GB) + DS4 image (~36 GB) per node; H3 weights (int8 draft ~40 GB of components / bf16 finals ~124 GB full checkpoint) — see below. |
| **OS-level RDMA** | Confirm `ibstatus` ACTIVE + a valid RoCEv2 GID (`show_gids`) on **all four** nodes before you touch a container. |

DS4 side (image, RDMA/NCCL env, worker-first startup, head-count rule, benchmarks) is fully
documented in the **TP=4 DS4 sibling repo** — this repo does not re-derive it:

➡️ **[`tonyd2wild/deepseek-v4-flash-tp4-4x-dgx-spark`](https://github.com/tonyd2wild/deepseek-v4-flash-tp4-4x-dgx-spark)**
(and the two-Spark base: [`Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX`](https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX)).

---

## The H3 side (MiniMax-H3 on ComfyUI, co-resident)

MiniMax-H3 is an **omni-modal video-diffusion** model (text/image/video/audio in one sequence,
24 fps / 32 kHz), run on **ComfyUI / diffusers — not vLLM.** ComfyUI loads and *frees* its
components in sequence (encode prompt → free encoder → run DiT → free → VAE decode), so the
binding memory constraint is the **largest single component**, not the sum — which is why an
int8 draft (largest piece ~20.9 GB DiT) co-tenants comfortably in the TP=4 headroom.

Co-tenant launch essentials (full script + scrubbed placeholders in
[`launch/h3-comfy-cotenant.sh`](launch/h3-comfy-cotenant.sh)):

- **`--reserve-vram 48`** — the piece that makes it fit. Caps H3's grab on the unified pool so
  DS4's KV pool survives an H3 render. (We also carry `--vram-headroom 10`, raised from 8 after
  a real UMA OOM.) **⚠️ box-type dependent:** on a discrete-GPU, RAM-poor box (e.g. a 3090)
  `--reserve-vram` is *counterproductive* — it pushes more into host RAM. On the GB10's unified
  memory it's exactly what protects the co-tenant. Don't copy the flag across box types blindly.
- **`--oom-score-adj 800`** — makes the H3 container the kernel's OOM victim, so if memory ever
  spikes **ComfyUI dies, not DS4.** This is the container-level half of KEYS' victim net; the
  fleet-level half is **earlyoom + choom** tuned to sacrifice an H3 span first.
- **`--disable-pinned-memory`** — the one flag every source agrees on: stops ComfyUI page-locking
  up to 90% of host RAM and eating the KV pool. Non-negotiable when co-tenanting.
- **Entrypoint override.** The H3 ComfyUI image ships `ENTRYPOINT sleep`, so you must override
  it: `--entrypoint <python> ... main.py --listen 0.0.0.0 --port 8188 --disable-pinned-memory
  --fp16-intermediates`.
- **RestartPolicy `no`** (not `unless-stopped`) — `unless-stopped` would let H3 start *before*
  DS4 on boot and win the memory race.
- **Weights are NOT duplicated.** Each node's H3 model dir is a symlink to a single NFS export;
  the `ComfyUI/models/<type>/*.safetensors` entries are symlinks into it. This means the mount
  trap below is fatal if you get it wrong.

**⚠️ The NFS-mount + VAE symlink trap:** the H3 model files are absolute-path symlinks. If you
bind-mount the H3 home to a *different* container path (e.g. `-v ~/h3:/root/h3`), every symlink
dangles, the loaders show empty, and **H3 dies almost instantly** with no useful error. Use a
**path-preserving** mount — `-v /home/<user>/h3:/home/<user>/h3 -w /home/<user>/h3/ComfyUI` —
so the absolute symlink targets still resolve inside the container. Details in
[`GB10-HYGIENE.md`](GB10-HYGIENE.md).

---

## Mandatory GB10 hygiene (the gotchas that will bite you)

Full list with symptoms/fixes in [`GB10-HYGIENE.md`](GB10-HYGIENE.md). The short version:

1. **`drop_caches` before every H3 render and before every DS4 boot.** Page cache starves the
   UMA allocator; watch `free -g` **column 4 (free)**, not column 7 (available).
2. **`--oom-score-adj 800` on every H3 container** + **earlyoom/choom** so H3 is always the OOM
   victim and DS4 never dies.
3. **Path-preserving mount** or the H3 model symlinks dangle and it dies in ~0.01 s.
4. **`--disable-pinned-memory`** always; never `--reserve-vram`/`--high-ram`/`--cache-lru` on a
   discrete-GPU box (opposite of the GB10 co-tenant rule — know your box type).
5. **Fresh `docker run`, load order DS4→H3, teardown H3→DS4.** DS4 does not survive
   `docker restart`.
6. **Thermal:** on a thermally-marginal box, keep H3 to short (~8 s) clips even clamped; route
   long clips to the healthy nodes. Watch temps + interrupt long renders.

---

## Measured cost of a live H3 render on DS4 throughput

Measured at **TP=2 co-tenancy** (our rig, their bench harness) — carry as *relative*, and
re-measure at TP=4 on your mix (the TP=4 headroom is larger, so expect this to be gentler, but
we have **not** separately benchmarked the TP=4 co-tenant render curve yet):

| DS4 concurrency | DS4 idle | +1 H3 render | +2 H3 renders |
|---|---|---|---|
| C1 | 88.87 t/s | **40.98** | **28.48** |
| C6 | 285.95 t/s | 130.77 | 100.79 |

**A live H3 render roughly halves DS4 throughput; an *idle* H3 lane costs only ~6–15%.** Worth
knowing before you schedule video against a live agent fleet — this is the argument for the
draft/finals split: iterate cheap in draft, and take the big quality pass in farm-mode when you
can afford to pause DS4.

---

## Files in this repo

- [`FARM-MODE.md`](FARM-MODE.md) — the int8-draft-co-tenant vs bf16-finals-farm-mode workflow in
  full: when to pause DS4, how the ~3× fan-out works, the quality tiers.
- [`GB10-HYGIENE.md`](GB10-HYGIENE.md) — every unified-memory gotcha with symptom + fix:
  drop_caches, the OOM victim net, the NFS/VAE symlink trap, pinned memory, thermal, load order.
- [`launch/cotenant-bringup.sh`](launch/cotenant-bringup.sh) — the ordered orchestration:
  drop_caches → DS4 TP=4 (worker-first) → drop_caches → H3 lanes.
- [`launch/ds4-tp4-cotenant-launch.sh`](launch/ds4-tp4-cotenant-launch.sh) — DS4 TP=4 launcher,
  scrubbed, with KEYS' gmu budget documented as a co-tenancy option.
- [`launch/h3-comfy-cotenant.sh`](launch/h3-comfy-cotenant.sh) — the H3 ComfyUI co-tenant
  container (int8 draft, `--reserve-vram 48`, `--oom-score-adj 800`, `--disable-pinned-memory`,
  entrypoint override, path-preserving mount), scrubbed with placeholders.

## See also

- **DS4 TP=4 base recipe:** [`tonyd2wild/deepseek-v4-flash-tp4-4x-dgx-spark`](https://github.com/tonyd2wild/deepseek-v4-flash-tp4-4x-dgx-spark) — the LLM half of this stack, with the full RDMA/NCCL deep-dive and the C1–C6 speed benchmarks.
- **DS4 TP=2 base recipe:** [`tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX`](https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX) — the two-Spark starting point.

---

## A note on numbers & scrubbing

- Numbers come from our own runs and are labeled by the topology they were measured on
  (**TP=2** vs **TP=4**). Where we don't have a TP=4 measurement, it says so — we didn't invent
  one. Test on your own rig; **no absolutes.**
- This repo uses the **public** DeepSeek-V4-Flash model id and public image references, and
  **placeholders** for anything site-specific (internal image tags, model paths, host IPs,
  usernames). The private production stack — an abliterated model on an internal path behind a
  private DSpark-spec image, plus our finals node set — is described in prose only, never
  published. Fill in your own.

---

*MIT licensed. Unofficial — a community recipe, not an official DeepSeek / NVIDIA / MiniMax /
vLLM / ComfyUI artifact. Co-tenancy budget by **KEYS**.*
