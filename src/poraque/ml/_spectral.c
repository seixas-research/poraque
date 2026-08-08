/* -*- coding: utf-8 -*- */
/* file: _spectral.c */

/* This code is part of Poraquê.
 * MIT License
 *
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 */

/*
 * The spectral-convolution contraction, in C, for CPU inference.
 *
 * One kernel, the einsum "bixyz,ioxyz->boxyz" of SpectralConv3d:
 *
 *     out[b, o, m] = sum_i  x[b, i, m] * w[i, o, m]
 *
 * with the three mode axes (x, y, z) flattened into a single contiguous run
 * `m` of length M = m1*m2*m3. Flattening is exact -- the axes are the fastest
 * varying in both operands and the contraction is elementwise across all
 * three -- and it turns a five-index gather into the loop below, whose inner
 * iteration walks three contiguous arrays in lockstep and vectorises.
 *
 * Why this is worth a C file at all
 * ---------------------------------
 * The cost of this contraction does not depend on the grid: it is set by the
 * retained mode count, which is a model hyper-parameter. The FFTs around it
 * scale as N^3 log N. So on a 96^3 field the contraction is ~10% of the
 * spectral layer and PyTorch's FFT dominates, but at the 32^3 resolution the
 * training cache is normally built at, the same contraction is ~74% of it --
 * and `torch.einsum` runs it at roughly 3.4 GFLOP/s, because its planner
 * reduces this pattern to a batched product over a layout with poor locality.
 *
 * Ordering the loops (b, o, i, m) instead makes `m` innermost and contiguous
 * in all three operands, so the accumulator row stays in L1 across the whole
 * `i` sweep and the compiler can emit vector FMAs over it.
 *
 * No Python C API and no numpy headers: the entry points below are plain C,
 * called through ctypes from `poraque.ml.backend`. That keeps the extension
 * outside the build system entirely -- hatchling never has to learn to compile
 * anything, and a wheel without a compiled copy still works, because the
 * Python side falls back to `torch.einsum`.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef PORAQUE_NO_PTHREADS
#include <pthread.h>
#define PORAQUE_HAVE_PTHREADS 1
#else
#define PORAQUE_HAVE_PTHREADS 0
#endif

/* Bumped whenever the ABI of the functions below changes. `backend.py`
 * checks it and rebuilds a stale cached library rather than calling into one
 * whose argument list no longer matches. */
#define PORAQUE_SPECTRAL_ABI 3

/* Above this the thread setup costs more than it saves: one contraction is
 * ~0.6 ms at production sizes. */
#define PORAQUE_MAX_THREADS 64

int poraque_spectral_abi(void) { return PORAQUE_SPECTRAL_ABI; }

/*
 * Threading is POSIX threads, deliberately, not OpenMP.
 *
 * PyTorch already links an OpenMP runtime. Linking a second one into the same
 * process is what libomp reports as "OMP: Error #15 ... can degrade
 * performance or cause incorrect results", and the documented workaround is an
 * environment variable whose own documentation calls it unsafe. pthreads have
 * no runtime to collide with, are in libc on every platform this runs on, and
 * the parallelisation needed here -- independent output rows, no reduction --
 * does not benefit from anything OpenMP would add.
 */
int poraque_spectral_threaded(void) { return PORAQUE_HAVE_PTHREADS; }

/*
 * Complex interleaved (re, im) contraction.
 *
 * x   : (B, I, M) complex, contiguous
 * w   : (I, O, M) complex, contiguous
 * out : (B, O, M) complex, contiguous, overwritten (not accumulated)
 *
 * `out` is zeroed per (b, o) row and then accumulated over `i`, so the caller
 * need not pre-zero it. Writing rather than accumulating is deliberate: the
 * caller assembles the four spectral corners into separate buffers, and an
 * accumulating kernel would silently double any corner computed twice.
 */
#define PORAQUE_CONTRACT_BODY(REAL)                                           \
    const ptrdiff_t stride = (ptrdiff_t)M * 2;                                \
    ptrdiff_t b, o, i, m;                                                     \
    for (b = 0; b < B; ++b) {                                                 \
        for (o = 0; o < O; ++o) {                                             \
            REAL *acc = out + ((b * O + o) * stride);                         \
            memset(acc, 0, (size_t)stride * sizeof(REAL));                    \
            for (i = 0; i < I; ++i) {                                         \
                const REAL *xp = x + ((b * I + i) * stride);                  \
                const REAL *wp = w + ((i * O + o) * stride);                  \
                for (m = 0; m < M; ++m) {                                     \
                    const REAL xr = xp[2 * m], xi = xp[2 * m + 1];            \
                    const REAL wr = wp[2 * m], wi = wp[2 * m + 1];            \
                    acc[2 * m]     += xr * wr - xi * wi;                      \
                    acc[2 * m + 1] += xr * wi + xi * wr;                      \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }

void poraque_spectral_contract_c64(const float *x, const float *w, float *out,
                                   long B, long I, long O, long M)
{
    PORAQUE_CONTRACT_BODY(float)
}

void poraque_spectral_contract_c128(const double *x, const double *w,
                                    double *out,
                                    long B, long I, long O, long M)
{
    PORAQUE_CONTRACT_BODY(double)
}

/*
 * The same contraction, spread over the (b, o) output rows.
 *
 * Each accumulator row is written by exactly one thread, so there is no
 * sharing, no reduction and no barrier inside the work -- the split is over
 * independent outputs.
 *
 * The kernel is memory-bound rather than compute-bound: at batch 1 every
 * weight element is read exactly once for eight flops, an arithmetic
 * intensity of 1 FLOP/byte. Threads therefore buy aggregate memory bandwidth,
 * not aggregate arithmetic, and the useful thread count saturates well below
 * the core count -- measured here, at four of eight.
 *
 * MPI was tried and dropped. Splitting these same rows across ranks works and
 * is correct, but lands on the identical bandwidth ceiling while costing a
 * process and a full model replica per rank plus a gather on every one of the
 * sixteen contractions in a forward pass. The operands already share an
 * address space; moving them between processes spends the one resource that
 * is scarce. Across nodes it is hopeless anyway -- this kernel is 0.2 ms,
 * below the latency floor of any interconnect. Rank-level parallelism belongs
 * one level up, over structures, where it is embarrassingly parallel.
 */
#define PORAQUE_ROW_BODY(REAL)                                                \
    const ptrdiff_t stride = (ptrdiff_t)M * 2;                                \
    ptrdiff_t row;                                                            \
    for (row = begin; row < end; ++row) {                                     \
        const ptrdiff_t b = row / O, o = row % O;                             \
        REAL *acc = out + (row * stride);                                     \
        ptrdiff_t i, m;                                                       \
        memset(acc, 0, (size_t)stride * sizeof(REAL));                        \
        for (i = 0; i < I; ++i) {                                             \
            const REAL *xp = x + ((b * I + i) * stride);                      \
            const REAL *wp = w + ((i * O + o) * stride);                      \
            for (m = 0; m < M; ++m) {                                         \
                const REAL xr = xp[2 * m], xi = xp[2 * m + 1];                \
                const REAL wr = wp[2 * m], wi = wp[2 * m + 1];                \
                acc[2 * m]     += xr * wr - xi * wi;                          \
                acc[2 * m + 1] += xr * wi + xi * wr;                          \
            }                                                                 \
        }                                                                     \
    }

#define PORAQUE_DEFINE_RANGE(SUFFIX, REAL)                                    \
    static void poraque_rows_##SUFFIX(const REAL *x, const REAL *w, REAL *out,\
                                      long I, long O, long M,                 \
                                      ptrdiff_t begin, ptrdiff_t end)         \
    {                                                                         \
        PORAQUE_ROW_BODY(REAL)                                                \
    }

PORAQUE_DEFINE_RANGE(c64, float)
PORAQUE_DEFINE_RANGE(c128, double)

#if PORAQUE_HAVE_PTHREADS

#define PORAQUE_DEFINE_THREADED(SUFFIX, REAL)                                 \
    typedef struct {                                                          \
        const REAL *x, *w;                                                    \
        REAL *out;                                                            \
        long I, O, M;                                                         \
        ptrdiff_t begin, end;                                                 \
    } poraque_job_##SUFFIX;                                                   \
                                                                              \
    static void *poraque_worker_##SUFFIX(void *argument)                      \
    {                                                                         \
        poraque_job_##SUFFIX *job = (poraque_job_##SUFFIX *)argument;         \
        poraque_rows_##SUFFIX(job->x, job->w, job->out, job->I, job->O,       \
                              job->M, job->begin, job->end);                  \
        return NULL;                                                          \
    }                                                                         \
                                                                              \
    void poraque_spectral_contract_##SUFFIX##_mt(                             \
        const REAL *x, const REAL *w, REAL *out,                              \
        long B, long I, long O, long M, int threads)                          \
    {                                                                         \
        const ptrdiff_t rows = (ptrdiff_t)B * (ptrdiff_t)O;                   \
        pthread_t handle[PORAQUE_MAX_THREADS];                                \
        poraque_job_##SUFFIX job[PORAQUE_MAX_THREADS];                        \
        int started = 0, k;                                                   \
        ptrdiff_t chunk, cursor = 0;                                          \
                                                                              \
        if (threads > PORAQUE_MAX_THREADS) threads = PORAQUE_MAX_THREADS;     \
        if (threads > (int)rows) threads = (int)rows;                         \
        if (threads <= 1) {                                                   \
            poraque_rows_##SUFFIX(x, w, out, I, O, M, 0, rows);               \
            return;                                                           \
        }                                                                     \
                                                                              \
        chunk = (rows + threads - 1) / threads;                               \
        for (k = 0; k < threads && cursor < rows; ++k) {                      \
            ptrdiff_t stop = cursor + chunk;                                  \
            if (stop > rows) stop = rows;                                     \
            job[k].x = x; job[k].w = w; job[k].out = out;                     \
            job[k].I = I; job[k].O = O; job[k].M = M;                         \
            job[k].begin = cursor; job[k].end = stop;                         \
            /* A thread that fails to start is not an error: its rows are  */ \
            /* run on this thread below, so the result is complete either  */ \
            /* way and only the speed differs.                             */ \
            if (pthread_create(&handle[started], NULL,                        \
                               poraque_worker_##SUFFIX, &job[k]) == 0) {      \
                ++started;                                                    \
                cursor = stop;                                                \
            } else {                                                          \
                break;                                                        \
            }                                                                 \
        }                                                                     \
        if (cursor < rows) {                                                  \
            poraque_rows_##SUFFIX(x, w, out, I, O, M, cursor, rows);          \
        }                                                                     \
        for (k = 0; k < started; ++k) pthread_join(handle[k], NULL);          \
    }

#else  /* no pthreads: the entry points exist and run serially */

#define PORAQUE_DEFINE_THREADED(SUFFIX, REAL)                                 \
    void poraque_spectral_contract_##SUFFIX##_mt(                             \
        const REAL *x, const REAL *w, REAL *out,                              \
        long B, long I, long O, long M, int threads)                          \
    {                                                                         \
        (void)threads;                                                        \
        poraque_rows_##SUFFIX(x, w, out, I, O, M, 0,                          \
                              (ptrdiff_t)B * (ptrdiff_t)O);                   \
    }

#endif

PORAQUE_DEFINE_THREADED(c64, float)
PORAQUE_DEFINE_THREADED(c128, double)
