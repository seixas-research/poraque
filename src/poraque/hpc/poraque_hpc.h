#ifndef PORAQUE_HPC_H
#define PORAQUE_HPC_H
#include <complex.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* LP64 ScaLAPACK ABI. All collective calls must be issued by the MPI main
 * thread on every rank in the context. Input/output root matrices are dense
 * column-major complex128. No Python/NumPy or potential data is owned by C. */
typedef struct pq_context pq_context;
int pq_context_create(int comm_f, int ranks_per_k, int block, int threads, pq_context **out);
int pq_context_info(pq_context *, int *group, int *groups, int *rank, int *size);
int pq_context_communicator(pq_context *, int *comm_f);
int pq_consensus(pq_context *, int local_error);
int pq_solve_root(pq_context *, int n, const double complex *h, const double complex *s,
                  double *eigenvalues, double complex *vectors);
/* Advanced distributed interface: buffers already in this context's 2D block
 * cyclic layout; overwritten H and S, output Z remains distributed. */
int pq_local_shape(pq_context *, int n, int *rows, int *cols, int *lld);
int pq_solve_local(pq_context *, int n, double complex *h, double complex *s,
                   double *eigenvalues, double complex *z);
int pq_fft3(pq_context *, int nx, int ny, int nz, const double complex *input,
            double complex *output, int inverse);
/* Inclusive cumulative seconds: solve_root, solve_local, pzheevd,
 * fft_total, fft_plan, fft_execute, assemble, reserved. profile is group
 * collective; reduce/barrier span the original communicator (all k groups). */
int pq_profile(pq_context *, double *out_eight, int reset);
int pq_reduce_base(pq_context *, const double *, double *, int n, int sum);
int pq_barrier_base(pq_context *);
int pq_assemble(pq_context *, int n, int m, const double complex *local,
                const double *kinetic, const double complex *beta,
                const double complex *d, double complex *out);
int pq_context_destroy(pq_context *);
/* Explicit runtime shutdown only; finalizes MPI solely if this library
 * initialized it and no context remains. Never finalizes externally owned MPI. */
int pq_shutdown(void);
const char *pq_error_string(int status);
/* Available only when CMake PORAQUE_FFTW_MPI=ON. Distributed slab storage is
 * caller-accessible; close every slab before calling pq_shutdown(). */
typedef struct pq_fft_slab pq_fft_slab;
int pq_fft_slab_create(int comm_f,int nx,int ny,int nz,int threads,pq_fft_slab **out);
int pq_fft_slab_data(pq_fft_slab *,ptrdiff_t *n0,ptrdiff_t *start,void **data);
int pq_fft_slab_execute(pq_fft_slab *,int inverse);
int pq_fft_slab_destroy(pq_fft_slab *);
#ifdef __cplusplus
}
#endif
#endif
