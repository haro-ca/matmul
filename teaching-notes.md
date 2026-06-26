# Teaching Notes — Semana 09: Tiled Matrix Multiplication on GPUs

Reference for teaching the "Matmul tiled" review deck (30 slides, Numba/CUDA, A10G). Source app: 09-matmul.vercel.app. This doc goes deeper than the slides: it adds the derivations, the hardware reasoning, instructor talking points, and the common student misconceptions for each section.

---

## The one-sentence thesis (say this first, repeat it last)
Tiling does not speed up the arithmetic — it speeds up data delivery. You load a block into shared memory once and reuse it across many threads, cutting trips to global memory by a factor of T (≈ 16× with TILE=16).

Everything in the deck is a consequence of this. If a student remembers only one thing, this is it.

## Table of contents

- Part 0 — Framing (slides 1–5)
- Part 1 — The problem: naive matmul is memory-bound (slides 6–7)
- Part 2 — The diagnostic: the roofline (slide 8)
- Part 3 — The tiling idea & reuse (slides 9–11)
- Part 4 — The kernel: V1, indices, double sync, V2 (slides 12–16)
- Part 5 — Trade-offs: tile size, occupancy, prediction (slides 17–20)
- Part 6 — The ceiling: Tensor Cores & precisions (slides 21–22)
- Part 7 — Debugging & connections (slides 23–25)
- Part 8 — Wrap-up (slides 26–30)
- The 6 transferable principles
- Key numbers cheat-sheet
- Self-check questions
## Part 0 — Framing (slides 1–5)

### Slide 1 — Título: "Matmul tiled: el kernel que define el deep learning"
The claim. Neural networks are, mechanically, mostly matrix multiplications: every dense/linear layer is Y = XW, and the heart of a transformer's attention (QKᵀ, then ·V) is matmul. Convolutions are lowered to matmul (im2col) or to closely related GEMM-like kernels. So the throughput of deep learning is the throughput of GEMM (GEneral Matrix Multiply) on the GPU. Master this single kernel and you understand the performance ceiling of essentially all of DL.

The root principle ("El principio raíz", one sentence): 16× menos viajes a la memoria global.

Tiling doesn't make the multiply-adds faster — the GPU already has FLOPs to spare.
It makes data delivery cheaper: load a block of A and B into shared memory once, then reuse it across the T threads that need it, instead of each thread re-reading from slow global memory.
Where 16× comes from: with TILE=16, every value loaded is reused by 16 threads → 16× fewer global reads. (Quantified later: arithmetic intensity goes 0.25 → 4.)
Tags as a syllabus: shared memory, tiling, roofline, doble syncthreads (red = the one that produces the worst bugs).

Teach-this: open by asking "what operation dominates a forward pass of an MLP or a transformer?" Lead them to "matrix multiply," then to "so making matmul fast = making DL fast."

### Slide 2 — Mapa: "Seis paradas, un principio en cada una"

The roadmap. The review reorganizes the week around durable ideas rather than re-reading code line by line. The six stops:

| # | Stop | The point |
|---|------|-----------|
| 1 | El problema | naive matmul is memory-bound: AI = 0.25 |
| 2 | La herramienta | tiling + shared memory raise arithmetic intensity |
| 3 | El modelo | roofline: memory-bound or compute-bound? |
| 4 | El kernel | build it: V1 → V2, indices, and the double sync |
| 5 | El trade-off | tile size: reuse vs occupancy |
| 6 | El techo | Tensor Cores and why cuBLAS still wins |
The success criterion (callout): by the end a student can explain, without looking at the code, (a) why tiling works, (b) when it stops helping, and (c) what separates their kernel from cuBLAS.

Slide 3 — Principios: "Si te llevas algo, que sean estos seis"
The 6 transferable principles (full list at the bottom). They apply to any kernel/GPU/framework, not just matmul. Principle 4 (synchronization) is highlighted in red because it is the most error-prone.

Heads-up / known typo: in the source data the synchronization card is Principle 4, but the rendered slide labels it "Principio 6". It's a label bug in the deck — the content is the 4th principle. So the set is genuinely 1–6, with #4 (sync) being the critical one.

### Slide 4 — Puente desde Semana 08: "Ya tienes todas las piezas"

Tiled matmul introduces no new concept — it orchestrates three things students already practiced. The difficulty is combining them, not any single one.

| From | Piece | What it gives you |
|------|-------|-------------------|
| S05 | Shared memory + cuda.syncthreads() | declaring shared arrays, barriers, shared-vs-global |
| S08 | 2D indexing | mapping threads to (row, col) via blockIdx / threadIdx |
| S08 | The tile concept | splitting a big problem into sub-blocks that fit in shared memory |
| S09 | New this week | accumulating partial products while iterating over K-tiles |
Teach-this: "If a piece is shaky, the kernel is impossible to debug — the bug almost always lives in the weak piece. Shore it up first." Diagnose which of the three a struggling student is weak on before debugging their kernel.

### Slide 5 — Objetivos de aprendizaje

The goal is reasoning the kernel, not memorizing 20 lines. Concretely, by the end a student can:

- Implement tiled matmul with shared memory (error < 1e-3 vs NumPy).
- Handle boundary conditions for arbitrary matrix sizes.
- Measure & compare GFLOPS of naive / tiled / cuBLAS, and the % of peak.
- Select a tile size from the reuse-vs-occupancy trade-off.
- Place a kernel on the roofline to diagnose its bottleneck.
## Part 1 — The problem: naive matmul is memory-bound (slides 6–7)
### Slide 6 — ¿Por qué tiling? Naive wastes bandwidth

```python
@cuda.jit
def matmul_naive(C, A, B):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        acc = 0.0
        for k in range(A.shape[1]):
            acc += A[row, k] * B[k, col]   # 2K global reads per output element
        C[row, col] = acc
```
Each thread reads an entire row of A and an entire column of B straight from global memory (off-chip DRAM, ~400–800 cycle latency). The GPU spends almost all its time waiting for data, not multiplying. This is memory-bound.

**Thread-by-thread breakdown for 4×4 naive (TILE=2):**

| Thread | Computes | Reads from A | Reads from B |
|--------|----------|--------------|--------------|
| (0,0) | C[0,0] | Row 0: [a00,a01,a02,a03] | Col 0: [b00,b10,b20,b30] |
| (1,0) | C[0,1] | Row 0: [a00,a01,a02,a03] | Col 1: [b01,b11,b21,b31] |
| (0,1) | C[1,0] | Row 1: [a10,a11,a12,a13] | Col 0: [b00,b10,b20,b30] |
| (1,1) | C[1,1] | Row 1: [a10,a11,a12,a13] | Col 1: [b01,b11,b21,b31] |

**The problem:** Threads (0,0) and (1,0) both read the same Row 0 of A, but each fetches it independently from global memory. No reuse → massive redundancy.

### Slide 7 — Intensidad aritmética: AI = FLOPs / bytes
Arithmetic intensity (AI) = useful FLOPs ÷ bytes moved from global memory. It's the single number that decides whether you're memory- or compute-bound.

Derivation for naive, per output element C[i,j]:

FLOPs: the inner product over K is K multiplies + K adds = 2K FLOPs.
Bytes: read row i of A (K floats) + column j of B (K floats) = 2K floats × 4 B = 8K bytes. (The single write of C[i,j] is negligible and usually excluded.)
AI = 2K / 8K = 0.25 FLOP/byte. Terrible — far below any GPU's break-even.
Why tiling raises it. Tiling does not change the FLOPs (2K per element, always). It changes the bytes: each value loaded into shared memory is reused by T threads, so bytes read drop by a factor T.

naive    : 8K   bytes/element  → AI = 0.25
tiled T  : 8K/T bytes/element  → AI = T/4

T=8  → AI = 2
T=16 → AI = 4    (16× fewer reads)
T=32 → AI = 8    (32× fewer reads)
Block-level cross-check of AI = T/4: to compute one TILE×TILE block of C you load K/TILE tiles each of A and B, each tile TILE² floats: bytes = 2 · (K/TILE) · TILE² · 4 = 8K·TILE, for TILE² outputs → 8K/TILE bytes/element → AI = 2K / (8K/TILE) = TILE/4. ✓

Common misconception: students think tiling "does fewer multiplications." It does the exact same multiplications. It just reads each input fewer times. Hammer: same FLOPs, fewer bytes.

The supermarket analogy (slide 6): without tiling, 16 cooks each drive to the store separately (16 trips). With tiling, one trip brings ingredients to the shared kitchen and all 16 cook from them. The cooking (arithmetic) is identical; only the shopping (memory traffic) changed.

## Part 2 — The diagnostic: the roofline (slide 8)
The roofline answers "what should I even optimize?" before you touch code.

The model. Plot achievable performance vs arithmetic intensity (log-log):

attainable_GFLOPS = min( peak_compute,  AI × memory_bandwidth )
ridge_point (FLOP/byte) = peak_compute / memory_bandwidth
AI < ridge → memory-bound: you sit on the sloped part (AI × BW). The ceiling is bandwidth → reduce memory accesses (more tiling, better reuse).
AI > ridge → compute-bound: you sit under the flat part (peak FLOPs) → use better instructions/hardware (e.g. Tensor Cores), not less traffic.
For the A10G: ridge ≈ 31,200 GFLOPS ÷ 600 GB/s ≈ 52 FLOP/byte.

Key reality check: even T=32 (AI=8) is still below 52 → your tiled kernel is still memory-bound, just much less so. Tiling moves you toward the ridge; it doesn't cross it. The thing that crosses it (cuBLAS) does so by raising the ceiling with Tensor Cores, not by raising AI.

The vertical gap between your point and the roofline is the performance you're leaving on the table.

Teach-this: the roofline is a prediction tool, not decoration. Always classify a kernel (memory vs compute) before optimizing — it tells you which lever even matters.

## Part 3 — The tiling idea & reuse (slides 9–11)

### Slide 9 — La idea del tiling: the 6-step recipe
Every tiled kernel has this shape. A block of TILE×TILE threads owns one sub-block of C; to compute it, the block sweeps the K dimension, loading paired tiles and accumulating partial products.

C[0:T,0:T] = A[0:T,0:T]·B[0:T,0:T] + A[0:T,T:2T]·B[T:2T,0:T] + ...
(It's literally the definition of matmul, evaluated tile by tile.)

The recipe — memorize it; if the kernel breaks, walk it step by step:

- Iterate over K-tiles — split the shared dimension K into chunks of TILE.
- Load sA, sB — each thread copies one element of A and one of B into shared memory.
- syncthreads() — SYNC 1 — wait until the whole tile is loaded before reading it.
- Accumulate — acc += sA[ty,k] * sB[k,tx] for k in 0..TILE.
- syncthreads() — SYNC 2 — wait until everyone finished reading before overwriting.
- Write C[row,col] — after all K-tiles, dump the accumulator to global memory.

### Slide 10 — Tile a tile (worked example)
4×4 matrices, TILE=2, block (0,0) computes C[0:2,0:2] in 2 iterations:

t=0: load A[:,0:2] and B[0:2,:] → sync → accumulate partial product #1.
t=1: load A[:,2:4] and B[2:4,:] → sync → accumulate partial product #2.
Result: C[0:2,0:2] = sum of the two partial products.

**Detailed thread breakdown for Block (0,0) during iteration t=0:**

| Thread | Loads to shared memory | Reads from shared memory |
|--------|----------------------|-------------------------|
| (0,0) | sA[0,0]=a00, sB[0,0]=b00 | Reads sA[0,0], sA[0,1] and sB[0,0], sB[1,0] |
| (1,0) | sA[0,1]=a01, sB[0,1]=b01 | Reads sA[0,0], sA[0,1] and sB[0,1], sB[1,1] |
| (0,1) | sA[1,0]=a10, sB[1,0]=b10 | Reads sA[1,0], sA[1,1] and sB[0,0], sB[1,0] |
| (1,1) | sA[1,1]=a11, sB[1,1]=b11 | Reads sA[1,0], sA[1,1] and sB[0,1], sB[1,1] |

**The win:** Thread (0,0) loads `a00` once, but Thread (1,0) also reads it from shared memory. 1 global load → 2 uses. With TILE=16: 1 global load → 16 uses.

**Accumulator progression:**

After iteration t=0 (first partial product):
```
Thread (0,0): acc = a00*b00 + a01*b10  (PARTIAL)
Thread (1,0): acc = a00*b01 + a01*b11  (PARTIAL)
Thread (0,1): acc = a10*b00 + a11*b10  (PARTIAL)
Thread (1,1): acc = a10*b01 + a11*b11  (PARTIAL)
```

After iteration t=1 (adding second partial product):
```
Thread (0,0): acc = (a00*b00 + a01*b10) + (a02*b20 + a03*b30)  (COMPLETE)
Thread (1,0): acc = (a00*b01 + a01*b11) + (a02*b21 + a03*b31)  (COMPLETE)
Thread (0,1): acc = (a10*b00 + a11*b10) + (a12*b20 + a13*b30)  (COMPLETE)
Thread (1,1): acc = (a10*b01 + a11*b11) + (a12*b21 + a13*b31)  (COMPLETE)
```

**Key:** The accumulator `acc` starts at 0 and accumulates across iterations — it's not reset between tiles.

Teach-this: make students hand-trace this 4×4 / TILE=2 case. It is the single best cure for index confusion and for "why two syncs." If they can trace one block by hand, they understand the kernel.

### Slide 11 — Reuso: the actual saving
This is where the win lives. Each thread loads one element into sA and one into sB, but then reads an entire row of sA and an entire column of sB:

A row of sA was loaded by a single thread but is read by all T threads in that row of C.
A column of sB was loaded by a single thread but is read by all T threads in that column.
So: 1 global load → T uses. Without tiling each thread fetched its own row/column from global memory (T redundant loads); with tiling they share one load. Saving = factor T in global reads. That factor T is exactly the ×T improvement in AI.

Why shared memory makes this worth it: shared memory is on-chip SRAM, per-SM, ~100 KB on the A10G, with ~20–30 cycle latency and very high bandwidth — vs global DRAM at ~400–800 cycles. Reusing from shared memory instead of re-reading global is the whole game.

## Part 4 — The kernel: V1, indices, double sync, V2 (slides 12–16)

### Slide 12 — Kernel V1 (clean skeleton; assumes sizes are multiples of TILE)

```python
from numba import cuda, float32

TILE = 16   # must be a compile-time constant so Numba can size the shared arrays

@cuda.jit
def matmul_tiled_v1(C, A, B):
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x          # local column
    ty = cuda.threadIdx.y          # local row
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty

    _, K = A.shape
    acc = 0.0
    for t in range(K // TILE):     # assumes K is a multiple of TILE
        sA[ty, tx] = A[row, t * TILE + tx]
        sB[ty, tx] = B[t * TILE + ty, col]
        cuda.syncthreads()         # SYNC 1: everyone has loaded
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()         # SYNC 2: nobody overwrites yet
    C[row, col] = acc
```

Line-by-line reasons (not just what, but why):

- `cuda.shared.array((TILE,TILE), float32)` — reserves on-chip shared memory visible to the whole block; sA/sB are the tiles we reuse. TILE must be a compile-time constant.
- `col/row = blockIdx*TILE + threadIdx` — maps each thread to its global element of C. Forgetting the blockIdx*TILE offset is the #1 bug.
- `for t in range(K // TILE)` — sweep the K-tiles. V1 assumes K divisible by TILE.
- the two `sX[ty,tx] = ...` — each thread loads exactly one element of each tile.
- SYNC 1 — barrier so nobody reads shared memory until all loads finished.
- inner for k loop — partial product: row ty of sA × column tx of sB, accumulated in a register (acc).
- SYNC 2 — barrier so nobody overwrites the tile until all reads finished.
- `C[row,col] = acc` — after sweeping all K-tiles, write the result once.
- Launch config matters: threadsperblock = (TILE, TILE). The block must match the tile, or the shared tile isn't fully populated → corruption / CUDA_ERROR_ILLEGAL_ADDRESS.

### Slide 13 — Mapa de índices (the #1 bug)

~80% of bugs are mis-computed indices. The crucial invariants:

- row and col are fixed for a given thread (its output element of C).
- What changes with iteration t is which column of A it loads (a_col = t*TILE + tx) and which row of B it loads (b_row = t*TILE + ty).
- After the sync, the thread computes sum(sA[ty,k]·sB[k,tx] for k in 0..TILE) and finally writes C[row,col].
Teach-this: before any coding, have students fill a tiny table for, say, block (1,0), TILE=16, tx, ty, and a chosen t: what C[?, ?] do I write, what A[?, ?] and B[?, ?] do I read? The deck's index-tracer (sliders) does exactly this — replicate it on paper.

### Slide 14 — Doble syncthreads (THE exam question)

Why two barriers? Because they protect opposite hazards on the shared tile.

| Barrier | Hazard | Protects | If you omit it |
|---------|--------|----------|----------------|
| SYNC 1 (after load) | RAW (read-after-write) | the read | a thread reads sA[i,k] before another wrote it → reads garbage |
| SYNC 2 (after compute) | WAR (write-after-read) | the write | a fast thread loads tile t+1, overwriting sA, while a slow thread still reads tile t → two tiles mix |
Why this is the worst kind of bug: without the barrier the kernel sometimes works (if threads happen to run at similar speed) and sometimes yields small errors or NaNs on large matrices. It's an intermittent race condition — run the verification ~10× to expose it.

Golden rule: syncthreads() must never sit inside an if that not all threads in the block reach → deadlock (the barrier waits forever for threads that branched away).

Teach-this: ask "which line could one thread run that corrupts another thread's data, and when?" Lead them to: (1) reading before a neighbor's write, (2) overwriting before a neighbor's read. Each barrier kills one. Principle 4 in action: every barrier protects a concrete shared datum.

### Slide 15 — Boundary handling (arbitrary sizes)

If a matrix isn't a multiple of TILE, the last tile is partial. Reading out of bounds on a GPU is undefined behavior. Fix = ceil division in the K-loop + bounds-checked loads with zero padding (because 0 × x = 0 doesn't affect the sum).

```python
a_col = t * TILE + tx
if row < M and a_col < K:
    sA[ty, tx] = A[row, a_col]
else:
    sA[ty, tx] = 0.0          # safe padding

# ...and only write inside the matrix:
if row < M and col < N:
    C[row, col] = acc
```
Two changes vs V1: range(K//TILE) → range((K + TILE - 1)//TILE) (ceil), and the bounds ifs with zero padding.

Why zero padding is elegant: you need no special-case code for the last tile — the padded zeros accumulate harmlessly. 0·anything = 0.

### Slide 16 — Kernel V2 (production; any size)

V2 = V1 + boundary checks. This is the version used in the lab. Read it as a diff against V1: the only new things are the bounds ifs and the ceil division.

```python
@cuda.jit
def matmul_tiled_v2(C, A, B):
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x; ty = cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty
    M, K = A.shape; _, N = B.shape
    acc = 0.0

    for t in range((K + TILE - 1) // TILE):   # ceil division
        a_col = t * TILE + tx
        if row < M and a_col < K:
            sA[ty, tx] = A[row, a_col]
        else:
            sA[ty, tx] = 0.0                  # zero padding

        b_row = t * TILE + ty
        if b_row < K and col < N:
            sB[ty, tx] = B[b_row, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = acc
```
Robustness checklist: ✓ ceil division in the K-loop · ✓ if + 0.0 padding when loading sA/sB · ✓ if before writing C. These three together make it correct for all M, K, N.

## Part 5 — Trade-offs: tile size, occupancy, prediction (slides 17–20)
### Slide 17 — Tamaño de tile: reuse vs occupancy

There is no universal best tile size. Bigger tile = more reuse (higher AI) but more shared memory per block → fewer blocks fit per SM → fewer warps to hide latency → lower occupancy.

shared memory per block = 2 × TILE² × 4 bytes   (a tile of A + a tile of B, float32)

| TILE | Shared memory | Reuse | AI |
|------|---------------|-------|-----|
| 8    | 512 B         | 8     | 2   |
| 16   | 2 KB          | 16    | 4   | ← good default |
| 32   | 8 KB          | 32    | 8   |
Practical guidance: TILE=16 is a solid default (2 KB, good balance); use TILE=32 if the GPU has shared memory to spare. Principle 5: don't optimize blindly — measure.

### Slide 18 — Ocupancia: how many blocks fit on an SM?

An SM's shared memory is finite. If each block asks for a lot, fewer blocks run concurrently, leaving fewer warps to hide memory latency.

```python
smem_block = 2 * TILE**2 * 4            # bytes;  TILE=16 → 2 KB,  TILE=32 → 8 KB
blocks_per_SM = smem_of_SM // smem_block # shared-memory limit only
```
With the A10G's ~100 KB shared/SM: TILE=32 (8 KB) fits about half as many blocks as TILE=16 (2 KB). More reuse, less parallelism.

Important caveat: shared memory is only one occupancy limiter. The others are registers per thread and max warps/blocks per SM. The smallest limit wins. E.g. a (32,32) block is 1024 threads = 32 warps; with a 64-warp/SM cap, only 2 such blocks fit by the warp limit alone, regardless of shared memory.

### Slide 19 — GFLOPS esperados (validation table, A10G)

Use this to sanity-check a lab implementation. A10G peak ≈ 31.2 TFLOPS FP32; cuBLAS reaches ~15–20 TFLOPS (~50–65% of peak) using Tensor Cores. Tiled should be 10–30× faster than naive.

| Implementation | AI (FLOP/byte) | N=512 | N=1024 | N=2048 | zone |
|----------------|----------------|-------|--------|--------|------|
| Naive          | 0.25           | 10–40 | 15–50  | 15–50  | memory-bound |
| Tiled T=16     | 4.0            | 100–300 | 150–500 | 200–600 | transition |
| Tiled T=32     | 8.0            | 200–500 | 300–800 | 400–1000 | near the ceiling |
| cuBLAS         | 8–16           | 2000–8000 | 5000–12000 | 8000–15000 | Tensor Cores |
If your tiled kernel hits > 50% of cuBLAS, that's excellent — comparable to 2000s-era GPU implementations. The gap to cuBLAS is not your fault: NVIDIA spent years on it (Tensor Cores, register tiling, vectorized loads, warp-level primitives). Understanding why the gap exists is worth as much as closing it.

### Slide 20 — Predice antes de medir (the key habit)

Before running, predict with the roofline; then compare. For tiled, N=1024, TILE=16:

| Step | Reasoning |
|------|-----------|
| Theoretical AI | AI = T/4 = 16/4 = 4 FLOP/byte |
| Memory or compute? | 4 ≪ 52 (ridge) → memory-bound |
| Memory-bound ceiling | 600 GB/s × 4 = 2,400 GFLOPS theoretical |
| Realistic range | 10–25% efficiency → ~150–600 GFLOPS |
If your measurement misses the range, ask what you didn't account for: syncthreads overhead, shared-memory bank conflicts, SM occupancy. That gap is the lesson.

## Part 6 — The ceiling: Tensor Cores & precisions (slides 21–22)
### Slide 21 — Tensor Cores: cuBLAS uses different hardware

cuBLAS doesn't optimize memory better than a good tiled kernel — it runs on different compute units.

```python
# CUDA Core (your kernel): 1 scalar FMA per cycle, FP32
acc += sA[ty, k] * sB[k, tx]

# Tensor Core (cuBLAS): D = A·B + C, where A,B,C,D are small matrices (e.g. 4×4)
#   → ~64 multiply-adds in one cycle
```
A Tensor Core does a small matrix multiply-accumulate per op (≈ 64 MACs in the time a CUDA Core does 1). Principle 6: the 5–10× gap closes by changing compute unit (CUDA → Tensor Cores), a different roofline ceiling — not by better memory.

Why cuBLAS wins on A10G (Ampere): 3rd-gen Tensor Cores · TF32 by default for FP32 inputs (no code change) · register tiling (another reuse level, in registers) · 128-bit vectorized loads. Peak: ~31 TFLOPS FP32 (CUDA Cores) vs ~125+ TFLOPS TF32 (Tensor Cores).

Reaching Tensor Cores directly needs the WMMA API or Triton — out of scope. The point is to understand why the gap exists.

### Slide 22 — Precisiones: the roofline has many ceilings

Fewer bits = more throughput on the Tensor Cores → the roofline has one ceiling per format:

| Format | Unit | bits | ~TFLOPS (A10G) | vs FP32 | Use |
|--------|------|------|----------------|---------|-----|
| FP32   | CUDA Cores | 32 | ~31 | 1× | your tiled kernel; full precision, 23-bit mantissa |
| TF32   | Tensor Cores | 19 | ~125 | ~4× | cuBLAS default on Ampere; FP32 range, 10-bit mantissa |
| FP16   | Tensor Cores | 16 | ~250 | ~8× | mixed-precision training; reduced range, overflow risk |
| BF16   | Tensor Cores | 16 | ~250 | ~8× | LLM training; FP32 range, fewer mantissa bits |
| INT8   | Tensor Cores | 8 | ~500 | ~16× | quantized inference; integers, not floats |
TF32 is the silent trick: same exponent range as FP32, 10-bit mantissa; cuBLAS uses it by default on Ampere, so you see ~4× without changing a line. The roofline isn't invalidated — only the ceiling you compare against moved (from ~31 to ~125 TFLOPS).

## Part 7 — Debugging & connections (slides 23–25)
### Slide 23 — Bugs comunes (the 6 — keep next to you in the lab)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Missing 2nd syncthreads() | sometimes-correct, small errors / NaN on big matrices | two syncs per iteration (after load, after compute) |
| 2 | Bad boundary handling | passes for multiples of TILE, fails for arbitrary sizes | ceil division + if+0.0 padding + if before write |
| 3 | threads != TILE at launch | corruption / CUDA_ERROR_ILLEGAL_ADDRESS | threadsperblock = (TILE, TILE) |
| 4 | Wrong tile indices | allclose fails everywhere, large diffs | row=by*TILE+ty, col=bx*TILE+tx; trace a block by hand |
| 5 | Normal array instead of shared | tiled is slower than naive | use cuda.shared.array((TILE,TILE), float32) |
| 6 | TILE not a compile-time const | Numba can't allocate shared mem / compile error | define TILE as a global constant; re-run on change |
The most treacherous is #1 (the second sync): a race condition that sometimes passes verification. Run ~10× to catch it.

### Slide 24 — Conexiones (this kernel is the synthesis of half the course)

| Prior week | Connection |
|------------|------------|
| S05 — tiled transpose | both stage tiles in shared memory and need a sync between load and use. Difference: transpose uses each datum once (just relocates it); matmul reuses each datum TILE times → matmul gains far more from tiling. |
| S06 — reductions | acc += sA*sB is a reduction, but needs no shared memory because each thread accumulates in its own local acc (no inter-thread communication). In S06 the reduction needed shared memory because many threads contributed to the same result; here each C[i,j] is independent. |
| S04 — SAXPY (BLAS1) vs GEMM (BLAS3) | SAXPY has no reuse (each datum used once → memory-bound, AI≈0.167). GEMM reuses each datum O(N) times; tiling is exactly what exploits that and pushes GEMM toward compute-bound. |
The master pattern: load to fast memory → sync → reuse → sync → write. Seen in S05, S06, S08; matmul is its most complete form.

### Slide 25 — ¿Custom o cuBLAS? (the practical decision)

You wrote the kernel to understand it, not to beat cuBLAS.

| Case | Use | Why |
|------|-----|-----|
| Standard matmul FP32/FP16 | cuBLAS | Tensor Cores + years of tuning; you won't match it with a hand kernel |
| Learning how the GPU works | custom | writing tiled matmul teaches shared memory, sync, roofline — huge pedagogical value |
| matmul + fused activation (epilogue) | custom | fusing avoids extra DRAM passes a library call would incur |
| Production, large matrices | cuBLAS / cuDNN | what PyTorch calls under the hood |
What PyTorch actually does: alternates — cuBLAS/cuDNN for matmul & conv, custom kernels for activations, normalization, attention. Exactly the decision you just learned to make.

## Part 8 — Wrap-up (slides 26–30)
### Slide 26 — Hacia el lab (~5 hours, Modal + A10G)

Deliverables: matmul_tiled.py (V1 and V2 complete) + lab09_respuestas.md (GFLOPS table + answers P1–P7).

| Step | Time | What |
|------|------|------|
| 0. Code tracing | 30 min | trace the tile loop by hand (no GPU) |
| 1. Implement tiled | 120 min | complete V1 and V2 from the skeleton |
| 1.5 Predict | 15 min | roofline reasoning before measuring |
| 2. Naive vs tiled vs cuBLAS | 90 min | GFLOPS table |
| 3. Tile size | 60 min | TILE 8/16/32 and its effect |
| 4. Roofline (optional) | 60 min | place your measurements |
| 5. Debugging (optional) | 30 min | race conditions by hand |
Checkpoint emocional: if tiled passes np.allclose and shows speedup, you completed the hardest optimization in the course. It's normal for it to take hours and several attempts — this is code that senior NVIDIA engineers debug professionally.

### Slide 27 — Preview S10: from writing kernels to measuring them
Next week shifts from optimizing to knowing whether it's optimal: profiling with Nsight and cupyx.profiler (find exactly where time goes, no guessing), libraries (cuBLAS/cuFFT), and bank conflicts. "The rest of the course is integration and tooling, not new algorithms of this complexity — the hardest part is behind you."

### Slide 28 — 3 ideas to take away

- Tiling turns memory-bound into (closer to) compute-bound — reusing in shared memory raises AI by a factor T; TILE=16 goes from 0.25 → 4 FLOP/byte (16× more memory-efficient).
- The two syncthreads() are mandatory — first: data loaded before reading; second: nobody overwrites before all have read. Omit either → intermittent race condition.
- Understanding the cuBLAS gap is worth as much as closing it — tiled is 10–30× over naive, but cuBLAS is 5–10× ahead via Tensor Cores + register tiling: different hardware, not better memory.
### Slide 29 — Principles to take away (self-assessment)

The six principles transcend matmul — they let you optimize any kernel. Checklist before the lab:

- [ ] I can explain why tiling improves AI and by what factor (T).
- [ ] I understand why two syncthreads() and what each protects.
- [ ] I can handle partial edge tiles with zero padding.
- [ ] I can hand-trace which indices of A and B each thread loads.
- [ ] I can distinguish CUDA Cores from Tensor Cores and why cuBLAS is ~4–8× faster.
- [ ] I can place a kernel on the roofline and say memory- vs compute-bound.
- [ ] I know the TILE trade-off (big = reuse, small = occupancy).

### Slide 30 — Cierre

You wrote the same pattern cuBLAS and the Tensor Cores use inside. Tiling with shared memory is the heart of every fast matmul on the planet. You understand it, can reason about it, and can debug it — that's what separates someone who uses the GPU from someone who programs it.

---

## The 6 transferable principles

- Mueve menos, no calcules menos — Move less, don't compute less. The bottleneck is feeding the compute units, not the arithmetic. Optimize = reduce memory traffic.
- Reuso = intensidad aritmética — Reuse = arithmetic intensity. Each global read should serve many threads; reuse raises AI by a factor T and pushes the kernel toward the ceiling.
- El roofline te dice qué optimizar — The roofline tells you what to optimize. Memory-bound → cut accesses; compute-bound → better instructions. Diagnose before coding.
- Sincronizar protege datos compartidos — (critical) Synchronizing protects shared data. Each barrier guards a concrete datum; forget one = race condition.
- Todo trade-off tiene dos lados — Every trade-off has two sides. Big tile = more reuse but less occupancy. No universal best — measure.
- Conoce el hardware bajo la abstracción — Know the hardware under the abstraction. The cuBLAS gap closes by switching compute units (CUDA → Tensor Cores), not by better memory.

---

## Key numbers cheat-sheet (A10G)

| Quantity | Value |
|----------|-------|
| Naive AI | 0.25 FLOP/byte (= 2K FLOPs / 8K bytes) |
| Tiled AI | T/4 → T=16 ⇒ 4, T=32 ⇒ 8 |
| Roofline ridge point | ≈ 52 FLOP/byte (= 31.2 TFLOPS ÷ 600 GB/s) |
| Peak FP32 (CUDA Cores) | ~31 TFLOPS |
| Peak TF32 (Tensor Cores) | ~125+ TFLOPS |
| Memory bandwidth | ~600 GB/s |
| Shared memory per SM | ~100 KB |
| Shared memory per block | 2 × TILE² × 4 B → T=16 ⇒ 2 KB, T=32 ⇒ 8 KB |
| Default tile | TILE=16 |
| Tiled vs naive | 10–30× faster |
| cuBLAS vs tiled | still 5–10× ahead |

---

## Self-check questions

- Why does tiling raise arithmetic intensity, and by exactly what factor? (Answer: each loaded value is reused by T threads → bytes drop by T → AI = T/4.)
- Why two syncthreads(), and what hazard does each prevent? (SYNC1 = RAW, read before neighbor's write; SYNC2 = WAR, overwrite before neighbor's read.)
- How do you make the kernel correct for arbitrary sizes? (ceil division + bounds if + 0.0 padding + if before write.)
- Your tiled kernel at N=1024, TILE=16 — predict its GFLOPS range and justify with the roofline. (AI=4 ≪ 52 → memory-bound → 600×4=2400 theoretical → 10–25% → ~150–600 GFLOPS.)
- Why is cuBLAS ~4–8× faster even with perfect memory optimization on your side? (Different compute unit: Tensor Cores + TF32, a higher roofline ceiling.)
- When would you actually write a custom matmul in production instead of calling cuBLAS? (When fusing with a custom epilogue, or when the op doesn't exist in a library.)