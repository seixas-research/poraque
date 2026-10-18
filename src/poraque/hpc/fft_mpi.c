/* Optional FFTW MPI slab interface. Requires FFTW MPI development headers.
 * MPI must already be initialized with FUNNELED or better. All calls are
 * collective on comm_f and must originate from the MPI main thread.
 * The caller owns lifecycle; close slabs before pq_shutdown(). */
#include "poraque_hpc.h"
#include <mpi.h>
#include <fftw3-mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

struct pq_fft_slab {
    MPI_Comm comm;
    ptrdiff_t nx,ny,nz,n0,start,allocated;
    fftw_complex *data;
    fftw_plan forward,backward;
};

int pq_fft_slab_create(int comm_f,int nx,int ny,int nz,int threads,pq_fft_slab **out) {
    int initialized=0,main=0,level=0,local=0,global=0;
    MPI_Initialized(&initialized);
    if (!initialized || !out || nx<1 || ny<1 || nz<1 || threads<1) return 1;
    if ((ptrdiff_t)nx>PTRDIFF_MAX/ny || (ptrdiff_t)nx*ny>PTRDIFF_MAX/nz) return 2;
    MPI_Is_thread_main(&main); MPI_Query_thread(&level);
    if (!main || level<MPI_THREAD_FUNNELED) return 8;
    *out=NULL;
    MPI_Comm comm=MPI_Comm_f2c((MPI_Fint)comm_f);
    pq_fft_slab *c=calloc(1,sizeof(*c));
    local=c?0:2; MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm);
    if (global) { free(c); return global; }
    MPI_Comm_dup(comm,&c->comm);
    c->nx=nx;c->ny=ny;c->nz=nz;
    local=fftw_init_threads()?0:7;
    MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,c->comm);
    if (global) goto failure;
    fftw_mpi_init();
    fftw_plan_with_nthreads(threads);
    c->allocated=fftw_mpi_local_size_3d(nx,ny,nz,c->comm,&c->n0,&c->start);
    c->data=fftw_alloc_complex(c->allocated>0?c->allocated:1);
    local=c->data?0:2; MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,c->comm);
    if (global) goto failure;
    c->forward=fftw_mpi_plan_dft_3d(nx,ny,nz,c->data,c->data,c->comm,FFTW_FORWARD,FFTW_ESTIMATE);
    c->backward=fftw_mpi_plan_dft_3d(nx,ny,nz,c->data,c->data,c->comm,FFTW_BACKWARD,FFTW_ESTIMATE);
    local=c->forward&&c->backward?0:7;
    MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,c->comm);
    if (global) goto failure;
    *out=c;return 0;
failure:
    if(c->forward)fftw_destroy_plan(c->forward);
    if(c->backward)fftw_destroy_plan(c->backward);
    if(c->data)fftw_free(c->data);
    MPI_Comm_free(&c->comm);free(c);return global;
}
int pq_fft_slab_data(pq_fft_slab *c,ptrdiff_t *n0,ptrdiff_t *start,void **data) {
    if(!c||!n0||!start||!data)return 1;
    *n0=c->n0;*start=c->start;*data=c->data;return 0;
}
int pq_fft_slab_execute(pq_fft_slab *c,int inverse) {
    int main=0;MPI_Is_thread_main(&main);
    if(!main||!c||(inverse!=0&&inverse!=1))return 1;
    fftw_execute(inverse?c->backward:c->forward);
    if(inverse){
        double scale=1.0/((double)c->nx*c->ny*c->nz);
        ptrdiff_t size=c->n0*c->ny*c->nz;
        #pragma omp parallel for
        for(ptrdiff_t i=0;i<size;++i)c->data[i]*=scale;
    }
    return 0;
}
int pq_fft_slab_destroy(pq_fft_slab *c) {
    if(!c)return 1;
    fftw_destroy_plan(c->forward);fftw_destroy_plan(c->backward);fftw_free(c->data);
    MPI_Comm_free(&c->comm);free(c);return 0;
}
