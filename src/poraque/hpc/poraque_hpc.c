/* Hybrid reference backend: MPI k groups, BLACS block-cyclic distribution,
 * ScaLAPACK generalized Hermitian solves, OpenMP copies, threaded FFTW.
 * MIT license. No VASP implementation or potential data is included. */
#include "poraque_hpc.h"
#include <mpi.h>
#include <omp.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(int) == 4, "ScaLAPACK LP64 requires 32-bit int");
_Static_assert(sizeof(MPI_Fint) == sizeof(int), "MPI Fortran handle must fit the C interface");
extern int Csys2blacs_handle(MPI_Comm);
extern void Cfree_blacs_system_handle(int);
extern void Cblacs_gridinit(int *, const char *, int, int);
extern void Cblacs_gridinfo(int, int *, int *, int *, int *);
extern void Cblacs_gridexit(int);
extern int numroc_(const int *, const int *, const int *, const int *, const int *);
extern void descinit_(int *, const int *, const int *, const int *, const int *,
                      const int *, const int *, const int *, const int *, int *);
extern void pzpotrf_(const char *, const int *, double complex *, const int *, const int *, int *, int *);
extern void pzhegst_(const int *, const char *, const int *, double complex *, const int *, const int *,
                     int *, double complex *, const int *, const int *, int *, double *, int *);
extern void pzheevd_(const char *, const char *, const int *, double complex *, const int *, const int *,
                     int *, double *, double complex *, const int *, const int *, int *, double complex *,
                     const int *, double *, const int *, int *, const int *, int *);
extern void pztrsm_(const char *, const char *, const char *, const char *, const int *, const int *,
                    const double complex *, const double complex *, const int *, const int *, int *,
                    double complex *, const int *, const int *, int *);

enum { PQ_OK=0, PQ_INVALID=1, PQ_MEMORY=2, PQ_MPI=3, PQ_DESCRIPTOR=4,
       PQ_OVERLAP=5, PQ_EIGEN=6, PQ_FFT=7, PQ_THREAD=8 };
struct pq_context {
    MPI_Comm comm, base;
    double timers[8];
    int rank, size, group, groups, block, threads;
    int system, grid, nprow, npcol, myrow, mycol;
};
static int owns_mpi=0, active_contexts=0, fft_ready=0;

const char *pq_error_string(int status) {
    switch(status) {
    case PQ_OK: return "success";
    case PQ_INVALID: return "invalid dimensions, buffers, or collective arguments";
    case PQ_MEMORY: return "allocation or integer-overflow limit";
    case PQ_MPI: return "MPI initialization or communication failure";
    case PQ_DESCRIPTOR: return "invalid ScaLAPACK descriptor";
    case PQ_OVERLAP: return "overlap is not positive definite or reduction failed";
    case PQ_EIGEN: return "ScaLAPACK eigensolver/workspace failure";
    case PQ_FFT: return "FFTW planning/allocation failure";
    case PQ_THREAD: return "MPI main-thread/FUNNELED contract violated";
    default: return "unknown backend error";
    }
}
static int main_thread(void) {
    int flag=0;
    return MPI_Is_thread_main(&flag)==MPI_SUCCESS && flag;
}
int pq_consensus(pq_context *c, int local) {
    int global=PQ_MPI;
    if (!c || !main_thread()) return PQ_THREAD;
    if (MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,c->comm)!=MPI_SUCCESS) return PQ_MPI;
    return global;
}
static void *allocate(size_t count, size_t width) {
    if (width && count>SIZE_MAX/width) return NULL;
    return calloc(count ? count:1,width);
}
int pq_context_create(int comm_f, int ranks_per_k, int block, int threads, pq_context **out) {
    int initialized=0, finalized=0, provided=0, rank=0, size=0, dimensions[2]={0,0};
    MPI_Comm base;
    pq_context *c=NULL;
    if (!out || ranks_per_k<1 || block<1 || threads<1) return PQ_INVALID;
    *out=NULL;
    MPI_Finalized(&finalized);
    if (finalized) return PQ_MPI;
    MPI_Initialized(&initialized);
    if (!initialized) {
        if (MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided)!=MPI_SUCCESS) return PQ_MPI;
        owns_mpi=1;
    }
    if (!main_thread()) return PQ_THREAD;
    MPI_Query_thread(&provided);
    if (provided<MPI_THREAD_FUNNELED && threads>1) return PQ_THREAD;
    base=comm_f<0 ? MPI_COMM_WORLD:MPI_Comm_f2c((MPI_Fint)comm_f);
    MPI_Comm_rank(base,&rank); MPI_Comm_size(base,&size);
    if (size%ranks_per_k) return PQ_INVALID;
    c=allocate(1,sizeof(*c));
    int failed=c ? 0:PQ_MEMORY, any=0;
    MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,base);
    if (any) { free(c); return any; }
    if (MPI_Comm_dup(base,&c->base)!=MPI_SUCCESS) { free(c); return PQ_MPI; }
    c->group=rank/ranks_per_k; c->groups=size/ranks_per_k;
    c->block=block; c->threads=threads;
    if (MPI_Comm_split(base,c->group,rank,&c->comm)!=MPI_SUCCESS) { MPI_Comm_free(&c->base); free(c); return PQ_MPI; }
    MPI_Comm_set_errhandler(c->comm,MPI_ERRORS_RETURN);
    MPI_Comm_rank(c->comm,&c->rank); MPI_Comm_size(c->comm,&c->size);
    MPI_Dims_create(c->size,2,dimensions);
    c->nprow=dimensions[0]; c->npcol=dimensions[1];
    c->system=Csys2blacs_handle(c->comm); c->grid=c->system;
    Cblacs_gridinit(&c->grid,"R",c->nprow,c->npcol);
    Cblacs_gridinfo(c->grid,&c->nprow,&c->npcol,&c->myrow,&c->mycol);
    ++active_contexts;
    omp_set_num_threads(threads);
    if (!fft_ready) fft_ready=fftw_init_threads();
    if (pq_consensus(c,fft_ready ? 0:PQ_FFT)) { pq_context_destroy(c); return PQ_FFT; }
    *out=c;
    return PQ_OK;
}
int pq_context_info(pq_context *c,int *group,int *groups,int *rank,int *size) {
    if (!c || !group || !groups || !rank || !size) return PQ_INVALID;
    *group=c->group; *groups=c->groups; *rank=c->rank; *size=c->size;
    return PQ_OK;
}
int pq_context_communicator(pq_context *c,int *comm_f) {
    if (!c || !comm_f || !main_thread()) return PQ_INVALID;
    *comm_f=(int)MPI_Comm_c2f(c->comm);
    return PQ_OK;
}
int pq_local_shape(pq_context *c,int n,int *rows,int *cols,int *lld) {
    const int zero=0;
    if (!c || n<1 || !rows || !cols || !lld) return PQ_INVALID;
    *rows=numroc_(&n,&c->block,&c->myrow,&zero,&c->nprow);
    *cols=numroc_(&n,&c->block,&c->mycol,&zero,&c->npcol);
    *lld=*rows>0 ? *rows:1;
    return PQ_OK;
}
static int descriptor(pq_context *c,int n,int *desc,int *rows,int *cols,int *lld) {
    const int zero=0;
    int info=0;
    int status=pq_local_shape(c,n,rows,cols,lld);
    if (status) return status;
    descinit_(desc,&n,&n,&c->block,&c->block,&zero,&zero,&c->grid,lld,&info);
    return pq_consensus(c,info ? PQ_DESCRIPTOR:0);
}
int pq_solve_local(pq_context *c,int n,double complex *a,double complex *b,double *w,double complex *z) {
    double started=MPI_Wtime(), eigen_started=0;
    int d[9],rows=0,cols=0,lld=0,one=1,itype=1,info=0,status;
    int lwork=-1,lrwork=-1,liwork=-1,iwquery=0;
    double scale=1.0,rquery=0;
    double complex query=0,*work=NULL,alpha=1.0;
    double *rwork=NULL; int *iwork=NULL;
    status=pq_consensus(c,(!a || !w || !z || n<1) ? PQ_INVALID:0);
    if (status) return status;
    int arguments[2]={n,b!=NULL}, minimum[2], maximum[2];
    MPI_Allreduce(arguments,minimum,2,MPI_INT,MPI_MIN,c->comm);
    MPI_Allreduce(arguments,maximum,2,MPI_INT,MPI_MAX,c->comm);
    if (minimum[0]!=maximum[0] || minimum[1]!=maximum[1]) return PQ_INVALID;
    status=descriptor(c,n,d,&rows,&cols,&lld);
    if (status) return status;
    if (b) {
        pzpotrf_("L",&n,b,&one,&one,d,&info);
        status=pq_consensus(c,info ? PQ_OVERLAP:0);
        if (status) return status;
        pzhegst_(&itype,"L",&n,a,&one,&one,d,b,&one,&one,d,&scale,&info);
        status=pq_consensus(c,(info || !isfinite(scale) || scale<=0) ? PQ_OVERLAP:0);
        if (status) return status;
    }
    pzheevd_("V","L",&n,a,&one,&one,d,w,z,&one,&one,d,&query,&lwork,&rquery,&lrwork,&iwquery,&liwork,&info);
    status=pq_consensus(c,(info || !isfinite(creal(query)) || creal(query)>INT_MAX ||
                          !isfinite(rquery) || rquery>INT_MAX || iwquery<1) ? PQ_EIGEN:0);
    if (status) return status;
    lwork=(int)ceil(creal(query)); lrwork=(int)ceil(rquery); liwork=iwquery;
    if (lwork<1 || lrwork<1) status=PQ_EIGEN;
    work=allocate((size_t)(lwork>0?lwork:1),sizeof(*work));
    rwork=allocate((size_t)(lrwork>0?lrwork:1),sizeof(*rwork));
    iwork=allocate((size_t)(liwork>0?liwork:1),sizeof(*iwork));
    status=pq_consensus(c,(!work || !rwork || !iwork) ? PQ_MEMORY:status);
    if (status) goto cleanup;
    eigen_started=MPI_Wtime();
    pzheevd_("V","L",&n,a,&one,&one,d,w,z,&one,&one,d,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
    c->timers[2]+=MPI_Wtime()-eigen_started;
    status=pq_consensus(c,info ? PQ_EIGEN:0);
    if (status) goto cleanup;
    if (b) {
        /* C=L^-1 H L^-H; X=L^-H Z restores S-normalized PAW spinors. */
        pztrsm_("L","L","C","N",&n,&n,&alpha,b,&one,&one,d,z,&one,&one,d);
        #pragma omp parallel for
        for (int i=0;i<n;++i) w[i]/=scale;
    }
cleanup:
    free(work); free(rwork); free(iwork);
    c->timers[1]+=MPI_Wtime()-started;
    return status;
}
static int global_index(int local,int block,int coordinate,int processes) {
    return ((local/block)*processes+coordinate)*block+local%block;
}
static void root_pack(pq_context *c,int n,const double complex *full,double complex *packed,
                      const int *offset,int unpack) {
    int zero=0;
    #pragma omp parallel for schedule(static)
    for (int rank=0;rank<c->size;++rank) {
        int row=rank/c->npcol,col=rank%c->npcol;
        int nr=numroc_(&n,&c->block,&row,&zero,&c->nprow);
        int nc=numroc_(&n,&c->block,&col,&zero,&c->npcol);
        for (int j=0;j<nc;++j) for (int i=0;i<nr;++i) {
            size_t global=(size_t)global_index(i,c->block,row,c->nprow)+
                          (size_t)global_index(j,c->block,col,c->npcol)*n;
            size_t local=(size_t)offset[rank]+i+(size_t)j*nr;
            if (!unpack) packed[local]=full[global];
            else ((double complex *)full)[global]=packed[local];
        }
    }
}
int pq_solve_root(pq_context *c,int n,const double complex *h,const double complex *s,
                  double *values,double complex *vectors) {
    double started=MPI_Wtime();
    int status=0,has_s=0,rows=0,cols=0,lld=0,zero=0;
    int *counts=NULL,*offset=NULL;
    double complex *a=NULL,*b=NULL,*z=NULL,*packed=NULL;
    double *w=NULL;
    if (!c || !main_thread()) return PQ_THREAD;
    MPI_Bcast(&n,1,MPI_INT,0,c->comm);
    if (c->rank==0) has_s=s!=NULL;
    MPI_Bcast(&has_s,1,MPI_INT,0,c->comm);
    status=pq_consensus(c,(n<1 || (size_t)n*n>INT_MAX ||
        (c->rank==0 && (!h || !values || !vectors))) ? PQ_INVALID:0);
    if (status) return status;
    pq_local_shape(c,n,&rows,&cols,&lld);
    a=allocate((size_t)lld*(cols>0?cols:1),sizeof(*a));
    z=allocate((size_t)lld*(cols>0?cols:1),sizeof(*z));
    if (has_s) b=allocate((size_t)lld*(cols>0?cols:1),sizeof(*b));
    w=allocate(n,sizeof(*w)); counts=allocate(c->size,sizeof(*counts)); offset=allocate(c->size,sizeof(*offset));
    if (c->rank==0) packed=allocate((size_t)n*n,sizeof(*packed));
    status=pq_consensus(c,(!a || !z || !w || !counts || !offset || (has_s&&!b) ||
                            (c->rank==0&&!packed)) ? PQ_MEMORY:0);
    if (status) goto cleanup;
    for (int rank=0;rank<c->size;++rank) {
        int r=rank/c->npcol,p=rank%c->npcol;
        int nr=numroc_(&n,&c->block,&r,&zero,&c->nprow),nc=numroc_(&n,&c->block,&p,&zero,&c->npcol);
        counts[rank]=nr*nc; offset[rank]=rank?offset[rank-1]+counts[rank-1]:0;
    }
    if (c->rank==0) root_pack(c,n,h,packed,offset,0);
    status=MPI_Scatterv(packed,counts,offset,MPI_C_DOUBLE_COMPLEX,a,rows*cols,MPI_C_DOUBLE_COMPLEX,0,c->comm);
    if (pq_consensus(c,status==MPI_SUCCESS?0:PQ_MPI)) { status=PQ_MPI; goto cleanup; }
    if (has_s) {
        if (c->rank==0) root_pack(c,n,s,packed,offset,0);
        status=MPI_Scatterv(packed,counts,offset,MPI_C_DOUBLE_COMPLEX,b,rows*cols,MPI_C_DOUBLE_COMPLEX,0,c->comm);
        if (pq_consensus(c,status==MPI_SUCCESS?0:PQ_MPI)) { status=PQ_MPI; goto cleanup; }
    }
    status=pq_solve_local(c,n,a,b,w,z);
    if (status) goto cleanup;
    status=MPI_Gatherv(z,rows*cols,MPI_C_DOUBLE_COMPLEX,packed,counts,offset,MPI_C_DOUBLE_COMPLEX,0,c->comm);
    if (pq_consensus(c,status==MPI_SUCCESS?0:PQ_MPI)) { status=PQ_MPI; goto cleanup; }
    if (c->rank==0) {
        memcpy(values,w,(size_t)n*sizeof(*w)); root_pack(c,n,vectors,packed,offset,1);
    }
    status=PQ_OK;
cleanup:
    free(a); free(b); free(z); free(w); free(packed); free(counts); free(offset);
    c->timers[0]+=MPI_Wtime()-started;
    return status;
}
int pq_fft3(pq_context *c,int nx,int ny,int nz,const double complex *input,double complex *output,int inverse) {
    double started=MPI_Wtime(), mark;
    fftw_complex *buffer=NULL;
    fftw_plan plan=NULL;
    size_t total;
    if (!c || !main_thread()) return PQ_THREAD;
    if (nx<1 || ny<1 || nz<1 || !input || !output || (inverse!=0 && inverse!=1)) return PQ_INVALID;
    if ((size_t)nx>SIZE_MAX/(size_t)ny || (size_t)nx*ny>SIZE_MAX/(size_t)nz) return PQ_MEMORY;
    total=(size_t)nx*ny*nz;
    if (total>SIZE_MAX/sizeof(*buffer)) return PQ_MEMORY;
    buffer=fftw_alloc_complex(total);
    if (!buffer) return PQ_MEMORY;
    memcpy(buffer,input,total*sizeof(*buffer));
    fftw_plan_with_nthreads(c->threads);
    mark=MPI_Wtime();
    plan=fftw_plan_dft_3d(nx,ny,nz,buffer,buffer,inverse?FFTW_BACKWARD:FFTW_FORWARD,FFTW_ESTIMATE);
    if (!plan) { fftw_free(buffer); return PQ_FFT; }
    c->timers[4]+=MPI_Wtime()-mark;
    mark=MPI_Wtime();
    fftw_execute(plan);
    c->timers[5]+=MPI_Wtime()-mark;
    double normalization=inverse?1.0/(double)total:1.0;
    #pragma omp parallel for
    for (size_t i=0;i<total;++i) output[i]=buffer[i]*normalization;
    fftw_destroy_plan(plan); fftw_free(buffer);
    c->timers[3]+=MPI_Wtime()-started;
    return PQ_OK;
}
int pq_context_destroy(pq_context *c) {
    if (!c || !main_thread()) return PQ_THREAD;
    Cblacs_gridexit(c->grid); Cfree_blacs_system_handle(c->system);
    MPI_Comm_free(&c->comm); MPI_Comm_free(&c->base); free(c);
    if (active_contexts>0) --active_contexts;
    return PQ_OK;
}
int pq_shutdown(void) {
    if (active_contexts) return PQ_INVALID;
    if (fft_ready) { fftw_cleanup_threads(); fft_ready=0; }
    if (owns_mpi) { owns_mpi=0; return MPI_Finalize()==MPI_SUCCESS?PQ_OK:PQ_MPI; }
    return PQ_OK;
}

/* Timer slots are inclusive and cumulative; never add parent/child slots.
 * Each rank measures local elapsed intervals (no synchronized clocks needed).
 * Collectives below are explicit so unequal k-group workloads remain safe. */
int pq_profile(pq_context *c,double *out,int reset) {
    if (!c || !out || !main_thread()) return PQ_INVALID;
    int rc=MPI_Allreduce(c->timers,out,8,MPI_DOUBLE,MPI_MAX,c->comm);
    if (reset) memset(c->timers,0,sizeof(c->timers));
    return rc==MPI_SUCCESS?PQ_OK:PQ_MPI;
}
int pq_reduce_base(pq_context *c,const double *input,double *output,int n,int sum) {
    if (!c || !input || !output || n<1 || !main_thread()) return PQ_INVALID;
    return MPI_Allreduce(input,output,n,MPI_DOUBLE,sum?MPI_SUM:MPI_MAX,c->base)==MPI_SUCCESS?PQ_OK:PQ_MPI;
}
int pq_barrier_base(pq_context *c) {
    if (!c || !main_thread()) return PQ_INVALID;
    return MPI_Barrier(c->base)==MPI_SUCCESS?PQ_OK:PQ_MPI;
}
/* Root-local OpenMP construction: H = Vlocal + diag(T) + B D B^dagger.
 * Local and output are column major; B and Hermitian D are row major.
 * D may contain arbitrary couplings including spinor/SOC projector blocks.
 * This kernel does not generate projectors or PAW one-center corrections. */
int pq_assemble(pq_context *c,int n,int m,const double complex *local,
                const double *kinetic,const double complex *beta,
                const double complex *d,double complex *out) {
    if (!c || !main_thread() || n<1 || m<0 || !local || !kinetic || !out ||
        (m && (!beta || !d))) return PQ_INVALID;
    double started=MPI_Wtime();
    double complex *bd=allocate((size_t)n*(size_t)m,sizeof(*bd));
    if (!bd) return PQ_MEMORY;
    #pragma omp parallel for schedule(static)
    for (int i=0;i<n;++i) for (int a=0;a<m;++a) {
        double complex value=0;
        for (int b=0;b<m;++b) value+=beta[(size_t)i*m+b]*d[(size_t)b*m+a];
        bd[(size_t)i*m+a]=value;
    }
    #pragma omp parallel for schedule(static)
    for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
        double complex value=local[i+(size_t)j*n]+(i==j?kinetic[i]:0.0);
        for (int a=0;a<m;++a) value+=bd[(size_t)i*m+a]*conj(beta[(size_t)j*m+a]);
        out[i+(size_t)j*n]=value;
    }
    free(bd);
    c->timers[6]+=MPI_Wtime()-started;
    return PQ_OK;
}
