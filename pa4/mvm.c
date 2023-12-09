/* Parallel matrix-vector multiplication with 1D block decomposition
 * We assume the matrix a and vector x have already been distributed.
 */
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

void MatrixVectorMultiply(int n, double *a, double *x, double *y, MPI_Comm comm)
{
    int i, j;
    int nlocal;         /* Number of locally stored rows of A */
    double *xbuf;       /* Buffer that will store entire vector x */
    int npes, myrank;
    MPI_Status status;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    /* Allocate memory for x buffer */
    xbuf = (double *) malloc(n * sizeof(double));

    nlocal = n / npes;

    /* Gather entire vector x on each processor */
    /********************************************/
    MPI_Allgather(x, nlocal, MPI_DOUBLE, xbuf, nlocal, MPI_DOUBLE, comm);
    /********************************************/

    // Print x buffer for debugging
    printf("Process %d X Buffer:\n", myrank);
    for (i = 0; i < n; i++)
    {
        printf("%.1f\t", xbuf[i]);
    }
    printf("\n");

    /* Perform local matrix-vector multiplication */
    for (i = 0; i < nlocal; i++)
    {
        y[i] = 0;
        for (j = 0; j < n; j++)
            y[i] += a[i * n + j] * xbuf[j];
    }

    // Print local result vector y for debugging
    printf("Process %d Local Y:\n", myrank);
    for (i = 0; i < nlocal; i++)
    {
        printf("%.1f\t", y[i]);
    }
    printf("\n");

    free(xbuf);
}

