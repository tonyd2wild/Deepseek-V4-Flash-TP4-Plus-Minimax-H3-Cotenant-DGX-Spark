# Two modes: int8 "draft" co-tenant vs bf16 "finals" farm-mode

The single most important operational decision when you run MiniMax-H3 next to
DeepSeek-V4-Flash on four Sparks is **which of two modes you're in.** They are not two quality
knobs on the same pipeline — they are two *different memory postures*, and only one of them
keeps DS4 serving.

> **Unofficial community recipe.** Numbers are our own measurements, labeled by the topology
> they were taken on. No guarantees; benchmark your own rig.

---

## Mode 1 — DRAFT: int8 H3, co-tenant with DS4 live

**This is the default.** DeepSeek stays up serving the agent fleet; H3 renders alongside it.

- **Model:** the **int8** H3 build. Because ComfyUI loads and *frees* its components in
  sequence (encode → free encoder → run DiT → free → VAE decode), the binding memory constraint
  is the **largest single component** — the int8 DiT at ~20.9 GB — not the sum of everything.
  That's what lets it drop into the TP=4 co-tenant headroom.
- **Fit:** governed by KEYS' budget — H3 launched with **`--reserve-vram 48`** so it can't grow
  into DS4's KV pool, plus **`--oom-score-adj 800`** + earlyoom/choom so an H3 span is the OOM
  sacrifice. See [`launch/h3-comfy-cotenant.sh`](launch/h3-comfy-cotenant.sh).
- **Speed:** ~8.8 min per span, co-tenant.
- **Quality:** slightly **stylized / posterized** — that look *is* the int8 tier, not a bug and
  not just the encoder. Perfectly good for iterating on composition, motion, timing, prompt
  wording.
- **Cost to DS4:** a **live** H3 render roughly **halves** DS4 throughput while it runs; an
  **idle** H3 lane costs only ~6–15%. DS4 never drops — it just goes slower for the duration of
  a render. (Render-cost table measured at TP=2 co-tenancy; see the README.)

Use draft mode for **all iteration**: previews, blocking, dialing in the prompt, checking a
likeness. You can burn as many draft spans as you want without ever taking DeepSeek down.

---

## Mode 2 — FINALS: bf16 H3, farm-mode (DS4 paused)

**This is the polished-final pass, and it requires pausing DS4.**

- **Model:** the full **bf16** H3 model — ~40 GB resident. That is **too big to share a Spark
  with DeepSeek**, full stop. There is no `--reserve-vram` number that makes a 40 GB bf16 model
  and a 38 GB DS4 rank and a usable KV pool all fit in ~120 GB alongside the page cache.
- **The move — FARM MODE:** briefly **pause DeepSeek** on the nodes you're rendering on, which
  frees ~38 GB/node of weights *plus* the whole KV pool, then run bf16 H3 across the **freed**
  Sparks at roughly **~3× the draft speed** (parallel fan-out, one heavy job per box), then
  **bring DeepSeek right back.** Typical pause is **~20–30 minutes.**
- **Quality:** native, properly-real — the reason you'd pause a live LLM fleet at all.

**Pausing DS4 is a human call every single time.** For the ~20–30 min of a farm-mode pass the
agent fleet must fall back to another model (a smaller local model, or a hosted fallback). Never
pause DS4 farm-mode silently or on a timer — decide it, announce it, do the pass, restore.

---

## The play (how the two modes fit together)

```
  iterate ────────────────────────────────►  ONE farm-mode pass ──►  done
  (DRAFT, int8, co-tenant)                     (FINALS, bf16)
  DeepSeek STAYS UP the whole time             DeepSeek PAUSED ~20–30 min
  ~8.8 min/span, stylized                      ~3× speed across freed Sparks, native quality
```

1. Do **all** your iteration in **draft co-tenant** — DeepSeek never drops. Nail the
   composition, motion, prompt, likeness, timing.
2. When the draft is locked, take **one farm-mode pass**: pause DS4 on the render nodes, fan the
   bf16 finals across the freed Sparks, ~20–30 min, then restore DS4.

Iterate cheap where it's free (draft); spend the expensive resource (a live LLM fleet's uptime)
exactly once, on the final.

---

## Why farm-mode is ~3× (and how the fan-out works)

Two independent multipliers stack:

1. **Freed weights + KV.** Tearing DS4 down on a node returns ~38 GB of weights and the entire
   KV allocation to the unified pool — now the bf16 H3 model has the whole box.
2. **Parallel across boxes.** With DS4 out of the way you run **one heavy H3 job per Spark** in
   parallel across all the freed nodes instead of squeezing a single int8 draft into shared
   headroom. H3 is **byte-deterministic across nodes** (same seed + same graph → sha256-identical
   output), so a multi-clip final splits cleanly across boxes with no cross-node drift.

**⚠️ One heavy H3 job per Spark.** A "fleet concurrency of 2" means two *boxes*, not two jobs on
one box — a second heavy render on the same node will wedge on memory. And **one driver per
box**: don't point two schedulers at the same node's `:8188`, they collide and worsen a wedge.

---

## Quick decision table

| You want to… | Mode | DS4 | H3 model |
|---|---|---|---|
| Iterate / preview / check a likeness | **DRAFT** | **stays up** (co-tenant) | int8 |
| Ship the polished final | **FINALS** | **paused ~20–30 min** | bf16 |
| Render while the agent fleet is busy | **DRAFT** only | stays up (goes ~½ speed while a render runs) | int8 |
| Batch many clips at best quality | **FINALS** | paused, fan out ~3× | bf16 |

---

*Related: [`README.md`](README.md) (memory math + KEYS budget), [`GB10-HYGIENE.md`](GB10-HYGIENE.md)
(the unified-memory gotchas that apply to both modes).*
