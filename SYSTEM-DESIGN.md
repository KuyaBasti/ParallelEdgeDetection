# Parallel Edge Detection — system design

> How a photograph becomes a line drawing, three ways.
>
> A PNG is flattened to floating-point grayscale, smoothed with a **5×5
> Gaussian**, differentiated with **Sobel** operators, and thresholded with
> **hysteresis** into a binary edge map. The same five stages run on three
> engines: a plain **sequential** loop nest (the correctness oracle), an
> **OpenMP + AVX** CPU engine that vectorizes the blur and gradient stencil
> interiors 8 pixels at a time, and a **CUDA** engine that gives every pixel its own GPU thread
> and never lets the data leave the device mid-pipeline. A GoogleTest harness
> races most of the CPU variants and demands output byte-identical to the
> sequential engine's (compared as 8-bit images, channel 0) from every variant
> it races and from both parallel engines.

This document is the developer-facing map of the whole system — every
component and how data moves between them. The companion
[README](README.md) covers building, running, and per-engine detail.

---

## End-to-end flowchart

<p align="center"><img src="docs/system-design-flowchart.svg" alt="Parallel Edge Detection end-to-end flowchart. testfiles/Lena_2048.png (2048×2048 RGB, 16 bits per channel) is decoded by stbi_load into an 8-bit RGB host Image (12 MB). That Image is the sequential engine and correctness oracle; it is copied into a ParallelImage (OpenMP + AVX, copy constructor default mode 2) and uploaded with cudaMemcpy into a CudaImage (CUDA, one thread per pixel, 32×32 blocks, __constant__ Gaussian and Sobel tables). All three engines run the same five virtual stages: convert to float grayscale, 5×5 Gaussian blur, 3×3 Sobel gradient, hysteresis edges(low, high) with 0.3 / 0.7 in the tests, and convert back to 8-bit grayscale (4 MB). ParallelImage defaults: convert mode 4 of 0–5, blur mode 4 of 0–4 (AVX), gradient mode 3 of 0–3 (AVX), edges mode 2 of 0–2; CudaImage launches one kernel per stage. ImageTest.TestBlur writes PNGs after stages 1–4 of the sequential run. The 8-bit results meet an exact-match gate (operator== on channel 0) against the sequential reference; the CUDA result first comes back through to_host(), the only device-to-host copy. ParallelTest.FullPerformance times Image against ParallelImage and prints an integer t0/t1 and the thread count; ParallelTest.Time* times each ParallelImage mode except the AVX gradient and the float-to-8-bit convert; TestCudaImage.TestEach times the upload, the five stages and to_host() for CudaImage's single mode, with nothing raced. CacheNuke (no test calls it) and the edgedetect main.cpp stub are present but unused. A legend explains the box colours and line styles." width="100%"></p>

---

## How to read it: the three ideas that matter

1. **One algorithm, three engines, one oracle.** The pipeline stages are
   virtual methods on `Image`; `ParallelImage` and `CudaImage` override them.
   The sequential implementation isn't legacy code — it's the *specification*.
   Every parallel path must reproduce its output **byte-for-byte** after
   quantization to 8-bit, which holds because every variant accumulates each
   pixel's stencil taps in the same order — keeping the CPU paths bit-identical
   — while on the CUDA path, whose test compares only the binary edge map, the
   0.3 / 0.7 thresholds absorb the low-order rounding differences it can
   introduce (nvcc contracts multiply-adds to FMA by default), unless a
   gradient lands within rounding error of a threshold. The exact-match gate
   turns subtle races and boundary bugs into hard test failures instead of
   slightly-wrong pictures.

2. **The pipeline is a `shared_ptr` chain, and that shape does real work.**
   `img->convert(...)->blur()->gradient()->edges(...)` frees every intermediate
   automatically when the statement ends, with no manual cleanup. On the CUDA engine the same
   shape becomes a *residency* guarantee: each stage allocates its output on
   the device, so the whole pipeline is one host→device upload, five kernel
   launches, and one download at `to_host()` — zero intermediate transfers.

3. **Optimization is measured, not assumed.** Each stage carries numbered
   mode variants — naive OpenMP, accessor-free pointer indexing, flat
   single-loop, AVX — and the harness races them back-to-back per run and
   prints microsecond costs. Each stage's call without a mode argument dispatches to a
   fixed default variant; the AVX gradient and the float→8-bit convert
   defaults are checked for correctness but never timed on their own.
   The design hypothesis the races are built to test: parallelism (OpenMP)
   buys the first factor, *memory access discipline* (raw indexing, then
   8-wide vector loads) buys the rest.

---

## Deep dive 1 — anatomy of a stencil, and the AVX split

Every stage except `convert` is a stencil: output pixel `(x, y)` is a
function of an input neighborhood (5×5 for blur, 3×3 for gradient and
hysteresis). Stencils near the border index outside the image, handled by
clamping coordinates to the nearest valid pixel (border replication).

That clamp is a branch — four comparisons per tap — and it poisons
vectorization. The AVX modes remove it structurally instead of predicating it:

<p align="center"><img src="docs/stencil-and-avx.svg" alt="Parallel Edge Detection, AVX interior and clamped border, for blur mode 4 (5×5 Gaussian, 25 taps) and gradient mode 3 (3×3 Sobel ×2, 9 taps), drawn on a 22 × 10 image. Both run two OpenMP parallel for loops over rows. Pass 1 covers the interior, a margin of kernel radius r from every edge (2 for blur, 1 for gradient): 8-pixel AVX blocks, then a scalar remainder, with no bounds checks. Pass 2 visits every pixel, skips the 2-pixel interior with continue and computes the border ring with clamped coordinates, so for the gradient it recomputes the 1-pixel ring pass 1 already wrote. At 2048 wide a row gets 255 AVX blocks plus 4 scalar pixels (blur) or 6 (gradient). Inner loop, per tap: _mm256_loadu_ps loads 8 adjacent floats of row y + j, _mm256_set1_ps broadcasts the kernel weight, and _mm256_mul_ps plus _mm256_add_ps accumulate (no FMA); after the taps the gradient takes _mm256_sqrt_ps of xsum² + ysum², and _mm256_storeu_ps writes the 8 results." width="100%"></p>

Row-major layout makes the 8-wide load natural: 8 horizontally-consecutive
pixels are 8 consecutive floats. OpenMP parallelizes over rows (`y`), so
threads own disjoint bands and never contend on writes. A scalar remainder
loop finishes each row when the interior width isn't a multiple of 8, and the
two passes overlap on a one-pixel ring for the gradient — recomputing a few
identical values in exchange for simple bounds.

The CUDA kernels keep the clamp: with one thread per pixel there's no vector
lane to poison, and the uniform-stencil shape means adjacent threads read and
write adjacent addresses — coalesced by construction. Kernel coefficient
tables (`gaussian`, `xdir`, `ydir`) live in `__constant__` memory, which is
cached and broadcast-optimized for the case where every thread reads the same
value at the same time — exactly what a convolution does.

## Deep dive 2 — the CUDA pipeline, one upload to one download

<p align="center"><img src="docs/cuda-pipeline.svg" alt="CUDA pipeline as run by TestCudaImage.TestEach on the 2048 × 2048 test image, in three lanes: testbinary (host, user_tests.cpp), CudaImage (host side, cuda_image.cu) and the GPU (default stream, device memory). Part 0 constructs a CudaImage from lena: cudaMalloc plus cudaMemcpy HostToDevice of 12 MB of 8-bit RGB, the only upload. Parts 1–5, convert(floatgrayscale), blur(), gradient(), edges(0.3f, 0.7f) and convert(grayscale), each cudaMalloc and cudaMemset a new output (16 MB float, or 4 MB 8-bit for the last) and asynchronously launch one kernel (convertRGBtoGRAYSCALE, blur_kernel, gradient_kernel, edges_kernel, convertFLOATINGGRAYSCALEtoGRAYSCALE) on a 64 × 64 grid of 32 × 32 blocks, one thread per pixel; reassigning test_each drops the previous image, whose cudaFree waits for the kernel still reading it. There is no explicit synchronization call in the code. Part 6, to_host() on last, does a blocking cudaMemcpy DeviceToHost of the 4 MB edge map into a new host Image. Then, untimed, the test compares it byte-exact with the sequential reference. A caveat notes that the grid size rounds down, so a side that is not a multiple of 32 leaves an edge strip with no threads; 2048 divides exactly." width="100%"></p>

Three consequences worth knowing:

- **The chain leans on `cudaFree`'s implicit synchronization.** Reassigning
  the `shared_ptr` frees the input buffer of a kernel that is still in
  flight; CUDA's blocking free waits for it, turning what would be a
  use-after-free into serialization.
- **Per-stage timings are muddied, not clean launch latencies.** Each timed
  stage includes the output's `cudaMalloc` + `cudaMemset`, the async launch,
  *and* the synchronizing free of the input — so it folds in roughly that
  stage's kernel execution plus memory-management overhead. The unambiguous
  figure is the pipeline total, ending at `to_host()`'s blocking copy.
- **The default stream serializes the stages** — each kernel sees its
  predecessor's completed output without explicit synchronization. Correct by
  construction, at the cost of no inter-stage overlap.

---

## Component inventory

| Component | Layer | Tech | Provenance | Where |
|---|---|---|---|---|
| `Image` base class + sequential stages | Oracle | C++20 | course scaffolding | [image.hpp](image.hpp) / [image.cpp](image.cpp) |
| STB integration (PNG-only) | I/O | C | third-party + scaffolding | [stb_instantiation.cpp](stb_instantiation.cpp) |
| `ParallelImage` mode variants | CPU engine | OpenMP + AVX intrinsics | ✅ implemented here | [parallel_image.hpp](parallel_image.hpp) |
| `CudaImage` + 5 kernels | GPU engine | CUDA | ✅ implemented here | [cuda_image.cu](cuda_image.cu) |
| Timing utilities (`GetTiming`, `CacheNuke`) | Harness | C++ / chrono | course scaffolding | [parallel_utils.cpp](parallel_utils.cpp) |
| Correctness + mode-race tests | Harness | GoogleTest | course scaffolding | [edgedetect_tests.cpp](edgedetect_tests.cpp) |
| CUDA pipeline test | Harness | GoogleTest | ✅ implemented here | [user_tests.cpp](user_tests.cpp) |
| Build (C++20, `-mavx`, CUDA 75/89, gtest FetchContent) | Build | CMake | course scaffolding | [CMakeLists.txt](CMakeLists.txt) |
| `edgedetect` CLI | — | — | ⬜ stub only ("Hello world!") | [main.cpp](main.cpp) |

---

## The numbers that matter

| Value | What it is |
|---|---|
| 5×5 = 25 taps | Gaussian blur kernel (weights sum ≈ 1.0) |
| 3×3 ×2 = 18 taps | Sobel X + Y convolutions per gradient pixel |
| 0.3 / 0.7 | weak / strong hysteresis thresholds used by the tests |
| 8 | float lanes per 256-bit AVX register — pixels per vector op |
| 2 px / 1 px | interior margin for the AVX blur / gradient passes |
| 32×32 = 1024 | threads per CUDA block; grid is (width/32, height/32) |
| 7.5, 8.9 | CUDA compute capabilities compiled for (Turing, Ada) |
| 2048×2048 | test image — 12 MB as RGB, 16 MB as float grayscale, 4 MB as 8-bit output |
| 1 + 1 | host↔device transfers for the whole GPU pipeline (upload + download) |
| /(255·3) | RGB→gray mapping: equal-weight channel mean into [0,1] |
| 6 / 5 / 4 / 3 | implementation variants per stage: convert / blur / gradient / edges — all raced except the AVX gradient (mode 3) and the float→8-bit convert direction (`TimeConvert` races only RGB→float), which run only via default dispatch (gradient mode 3, convert mode 4; the other float→8-bit convert modes are never called) |
| µs | timing resolution (`GetTiming`, std::chrono) |
| 0 | tolerated output difference — parallel results must match the oracle byte-for-byte (8-bit, channel 0) |

---

## Verification workflow

| Stage | Test | What it proves |
|---|---|---|
| 1 | `ImageTest.*` | PNG load/write (a bad path must throw), shrink vs reference PNG, convert round-trip, sequential pipeline runs end to end and writes its stage PNGs (`TestBlur` / `MakeSmall` assert nothing about their contents; see stage 5) |
| 2 | `ParallelTest.TimeCopy / TimeConvert / TimeBlur / TimeGradient / TimeEdge` | the OpenMP/AVX mode races: each variant timed and byte-identical to the sequential oracle — except the AVX gradient (mode 3) and the float→8-bit convert (only its default mode 4 runs), both covered only indirectly via default dispatch, e.g. in `TimeEdge` / `FullPerformance` |
| 3 | `ParallelTest.FullPerformance` | whole-pipeline sequential vs parallel race; prints speedup (integer-truncated `t0 / t1`) + thread count |
| 4 | `TestCudaImage.TestEach` | full GPU pipeline with per-stage timings; final output byte-identical to the oracle |
| 5 | visual | stage PNGs (`Lena_blurred/gradient/edges.png`) written to the build dir for eyeball verification |

---

## Design trade-offs & sharp edges

- **Exact-match testing over tolerance testing** — brutal but unambiguous;
  it forces every variant to keep the same accumulation order, which rules
  out reduction-reordering optimizations but makes "correct" binary. (One
  blind spot: `operator==` compares only channel 0 — complete for the
  single-channel pipeline outputs, but the RGB copy race effectively checks
  just the red channel.)
- **Grid truncation** — CUDA grids use integer division (`width/32`), so
  non-multiple-of-32 images would leave a black remainder strip. The kernels
  already carry per-thread bounds guards; ceil-division in the grid
  computation is the only missing piece.
- **Fallback traps** — `CudaImage`'s unsupported conversions and non-zero
  modes fall through to base-class CPU code that either touches a device
  pointer or, for conversions it lacks (e.g. `rgb→grayscale`), throws
  `"Not Implemented (yet)"`; the supported path
  (`rgb→floatgrayscale`, `floatgrayscale→grayscale`, mode 0) is the only safe
  one. On the CPU side, an out-of-range `ParallelImage` mode returns an
  allocated but never-written image for the stencil stages, and
  `ParallelImage::to_host()` throws rather than inheriting the base no-op.
- **Not full Canny** — no non-maximum suppression (edges are thick) and
  single-pass hysteresis (no iterative edge tracking). The pipeline optimizes
  for comparable parallel workloads, not publication-grade edge maps.
- **Warm-cache timings** — the mode races run back-to-back on the same image;
  `CacheNuke` exists to fix that but isn't wired into the current tests.

---

## Provenance

Course-provided scaffolding (image framework, oracle implementation, test and
build harness) with the parallel engines implemented on top: the OpenMP + AVX
mode system in `parallel_image.hpp` and the CUDA engine in `cuda_image.cu` /
`user_tests.cpp`. Image I/O via [STB](https://github.com/nothings/stb); test
image is the [ethically sourced Lena recreation](https://mortenhannemose.github.io/lena/)
by Morten Rieger Hannemose; tests via GoogleTest.
