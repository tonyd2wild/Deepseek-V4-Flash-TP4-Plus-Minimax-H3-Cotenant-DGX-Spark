# GB10 co-tenancy hygiene — the gotchas that will bite you

Everything here is a consequence of one fact: **the DGX Spark (GB10) has unified CPU+GPU
memory.** There is no separate VRAM. The model weights, the vLLM KV pool, MiniMax-H3's render
buffers, **and the OS page cache** all draw from the *same* ~120–121 GiB per node. Most
co-tenancy failures are that single pool being spent somewhere you didn't expect.

Ordered roughly by how often they bite.

> **Unofficial community recipe.** Symptoms/fixes are from our own rig. Verify on yours.

---

## 0. `drop_caches` before every DS4 boot AND every H3 render

**Symptom:** `torch.AcceleratorError: CUDA error: out of memory` at startup — on a box that
`free`/`available` says is half-empty. Or H3 dies at `cudaMallocAsync` right after staging
weights, before sampler step 1.

**Cause:** on unified memory the **page cache competes directly with the model**, and nothing
evicts it automatically. Reading a big file — *including the model loading its own weights* —
refills the page cache. So even the innocent sequence "start DS4 → start H3" fails on the second
step unless you drop caches *between* them, because loading DS4's ~150 GB of weights just filled
the cache.

**Fix — on the node, before you start anything on the GPU:**

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

**Watch the right column.** `free -g` **column 4 (free)** is what the GPU allocator needs, not
column 7 (available). We've watched *available* sit at 18 GiB while *free* fell to 1 GiB and H3
OOM'd — available is not the number.

**⚠️ Needs passwordless sudo, and it fails soft.** If the drop is wrapped in a bring-up script
and hits an interactive sudo prompt, you get a bring-up with **no cache drop** and a start
failure minutes later that looks completely unrelated. Make sure sudo is non-interactive for
this command on every node.

This is a **standard step of every bring-up**, on both/all nodes, before DS4 launches — not a
break-glass fix. Any hand-restart of DS4 that bypasses the bring-up script skips it, which is
exactly how you get bitten.

---

## 1. The OOM victim net — H3 must die before DS4, always

**Goal:** if memory ever spikes, the process the kernel kills is **an H3 render**, never
DeepSeek. This is KEYS' victim net and it has two halves:

- **Container level:** launch every H3 ComfyUI container with **`--oom-score-adj 800`** so the
  kernel picks it as the OOM victim first.
- **Fleet level:** **earlyoom + choom**, tuned so an H3 span is the memory sacrifice and DS4 can
  **never** be the one that dies. This is the box's protection for the whole co-tenancy.

With the victim net in place, an H3 OOM is a *recoverable render failure*, not a DS4 outage. If
H3 OOMs co-tenant, the fix is on the **H3** side (missing `--reserve-vram 48`, wrong mount, or
you're accidentally trying bf16 co-tenant) — it is **not** "DS4 is too big, shrink it." Do not
shrink DS4 to fix an H3 OOM.

---

## 2. The NFS-mount + VAE symlink trap (H3 dies in ~0.01 s)

**Symptom:** H3 container starts and dies almost instantly; loaders show **empty** model lists;
no useful error.

**Cause:** H3 weights are **not duplicated per node** — each node's H3 model dir is a symlink to
a single **NFS export**, and the `ComfyUI/models/<type>/*.safetensors` entries are **absolute-path
symlinks** into it. If you bind-mount the H3 home to a *different* path inside the container
(e.g. `-v ~/h3:/root/h3`), every absolute symlink target now points somewhere that doesn't
exist in the container → all dangle → loaders empty → death.

**Fix — path-preserving mount** so the absolute targets still resolve:

```bash
-v /home/<user>/h3:/home/<user>/h3   -w /home/<user>/h3/ComfyUI
```

Mount the H3 home at the **same absolute path** it has on the host. If the model dir is an NFS
export from one node, make sure that export is actually mounted and readable on every render
node *before* you start H3 — a missing mount is indistinguishable from this symlink failure.

---

## 3. `--disable-pinned-memory` always; and don't cross box types with `--reserve-vram`

**`--disable-pinned-memory`** is the one flag **every** source agrees on: without it ComfyUI
page-locks up to ~90% of host RAM, which on unified memory eats straight into the KV pool (and
pinned pages are unswappable/unreclaimable, so the kernel's only lever is the OOM-killer). Always
launch H3 with it when co-tenanting.

**`--reserve-vram 48` is right on the GB10 co-tenant, WRONG on a discrete-GPU box.** This is a
genuine, easy-to-miss reversal:

| Box type | `--reserve-vram` verdict |
|---|---|
| **GB10 / unified memory, co-tenant with DS4** | **USE `--reserve-vram 48`** — it caps H3's grab on the shared pool so DS4's KV survives. This is what makes co-tenancy fit. (We also carry `--vram-headroom 10`, raised from 8 after a real UMA OOM.) |
| **Discrete GPU, RAM-poor host (e.g. a 3090)** | **AVOID** `--reserve-vram` / `--high-ram` / `--cache-lru` — on that box they push *more* into host RAM and cause the OOM-killer. |

Same flag, opposite effect, because one box has unified memory and the other doesn't. Know your
box before copying a launch line.

---

## 4. Load order is a memory technique, not etiquette

**"Whoever loads first wins the memory."** On unified memory the first big allocator to start
defines the split, and it cuts both ways.

- **Bring-up:** `drop_caches` (all nodes) → **DS4 first** (worker-first: ranks 3,2,1 headless,
  wait ~20 s, then head rank 0), verify a real `/v1/models` → `drop_caches` again → **H3 second**.
- **Teardown:** strict reverse — **H3 down on every node, then DS4.** To restart DS4 you must
  stop H3 first, or the leftover H3 wins the memory and DS4 can't load.
- **RestartPolicy `no` on H3 containers** (not `unless-stopped`) — `unless-stopped` would let H3
  start *before* DS4 on a reboot and win the race.
- **Fresh `docker run` for DS4, never `docker restart`** — the multi-node DS4 cluster does not
  survive a restart; tear down on all four and re-run.

See [`launch/cotenant-bringup.sh`](launch/cotenant-bringup.sh) for the whole sequence.

---

## 5. A failed/idle H3 render leaks memory — restart the lane before a batch

**Symptom:** renders that used to fit start OOM'ing; a node's free memory sits low for a long
time even with nothing rendering.

**Cause:** when a render throws inside the sampler, ComfyUI never runs its unload path and the
staged model buffers stay mapped **forever** — one crash poisons every job after it on that
lane, and even clean runs leak over a long session.

**Fixes:**

- **Restart the H3 container before any big batch** — `docker restart <h3-container>`. DS4 is a
  **separate container** and is never touched by this.
- **ComfyUI `/free` reclaims memory** mid-session: `POST /free {"unload_models":true,
  "free_memory":true}`. Useful but **not sufficient alone** for a large render — combine with the
  restart-before-batch rule.
- **Guard the fan-out:** check `GET /system_stats` → `devices[0].vram_free` and skip a lane
  that's low before dispatching to it. Treat a failed probe as *unknown*, never as "healthy."
- **One driver per box.** Two schedulers POSTing to the same node's `:8188` collide and worsen a
  wedge.

---

## 6. Thermal — short clips only on a marginal box

**Symptom:** a long H3 clip climbs a node's temperature over a sustained render (we saw
68 → 71 → 82 °C on a thermally-marginal box) even with a clock clamp; a short clip on the same
clamped box stays cool.

**Fix:**

- On any thermally-marginal node, run **only short (~8 s) H3 clips** even clamped; **route long
  clips to the healthy nodes.**
- Keep an **independent watchdog** polling temp + DS4 every ~60 s; on a threshold (≥82 °C)
  **`POST /interrupt`** to that node's ComfyUI — it kills the render and the temp drops straight
  back, and DS4 never drops.
- A node whose DS4 rank is pinned under live agent load has little real memory left, so **H3
  won't fit there** regardless of thermals — render on the least-loaded boxes and skip the busy
  one.

---

## 7. Misc H3 render facts worth knowing

- **`720` is not a legal height.** Width/height step is **32** (720/32 = 22.5). Use **1280×704**.
  Frame count is **17n+5** (e.g. 124 = ~5.2 s).
- **Restarting an H3 lane blanks ComfyUI's in-memory queue-history panel** — the *files on disk
  are safe* (they're a host bind-mount), but the panel resets and from a user's side that looks
  like data loss. Say so before you restart a lane.
- **`docker inspect` shows `ExitCode=0 OOMKilled=false` on a kernel OOM-kill** — Docker doesn't
  flag it. Check `sudo dmesg -T | grep -i oom-kill`, don't trust `State.OOMKilled`.
- **Poll `history[pid].status.status_str == "success"`**, not mere membership in `/history` —
  a crashed render also appears in `/history` and can look "finished."

---

*Related: [`README.md`](README.md) (memory math + KEYS budget + launch order),
[`FARM-MODE.md`](FARM-MODE.md) (draft vs finals).*
