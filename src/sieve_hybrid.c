//HOW TO COMPILE: mpicc -O3 -fopenmp src/sieve_hybrid.c -o bin/sieve_hybrid -lm
//HOW TO RUN: OMP_NUM_THREADS=4 mpirun -np 4 bin/sieve_hybrid 10000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>    // === MPI header ===
#include <omp.h>    // === OpenMP header ===

int main(int argc, char *argv[]) {
    // ======== MPI initialization ========
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s N\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    long N = atol(argv[1]);
    long size = N - 1;                 // liczby od 2..N
    long chunk = size / nprocs;
    long rem   = size % nprocs;

    // Obliczamy zakres [low_value..high_value] dla procesu MPI rank
    long low_index, local_size;
    if (rank < rem) {
        local_size = chunk + 1;
        low_index  = rank * local_size;
    } else {
        local_size = chunk;
        low_index  = rem * (chunk + 1) + (rank - rem) * chunk;
    }
    long low_value  = 2 + low_index;
    long high_value = low_value + local_size - 1;

    // ======== MPI: proces 0 tworzy małe sito do sqrt(N) ========
    long limit = (long) sqrt(N);
    char *small_sieve = NULL;
    long  small_count = 0;
    long *primes = NULL;
    if (rank == 0) {
        small_sieve = malloc((limit+1)*sizeof(char));
        for (long i = 0; i <= limit; i++) small_sieve[i] = 1;
        small_sieve[0] = small_sieve[1] = 0;
        for (long p = 2; p*p <= limit; p++) {
            if (small_sieve[p])
                for (long m = p*p; m <= limit; m += p)
                    small_sieve[m] = 0;
        }
        // zbieramy małe liczby pierwsze
        for (long i = 2; i <= limit; i++)
            if (small_sieve[i]) small_count++;
        primes = malloc(small_count * sizeof(long));
        for (long i = 2, idx = 0; i <= limit; i++)
            if (small_sieve[i]) primes[idx++] = i;
    }

    // ======== MPI: broadcast small_count and primes to all processes ========
    MPI_Bcast(&small_count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) primes = malloc(small_count * sizeof(long));
    MPI_Bcast(primes, small_count, MPI_LONG, 0, MPI_COMM_WORLD);

    // ================================================
    // -------- ZSYNCHRONIZUJ WSZYSTKIE PROCESY I ROZPOCZNIJ POMIAR --------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();           // === MPI timer ===
    // ================================================

    // ======== Prepare local is_prime array ========
    char *is_prime = malloc(local_size * sizeof(char));
    for (long i = 0; i < local_size; i++) is_prime[i] = 1;

    // ======== Hybrid: OpenMP parallel region ========
    #pragma omp parallel for schedule(dynamic)
    for (long idx = 0; idx < small_count; idx++) {
        long p = primes[idx];
        long first = (low_value + p - 1) / p * p;
        if (first < p*p) first = p*p;
        for (long m = first; m <= high_value; m += p)
            is_prime[m - low_value] = 0;   // marking composites
    }

    // ======== MPI: reduction of counts and times ========
    long local_count = 0;
    for (long i = 0; i < local_size; i++)
        if (is_prime[i]) local_count++;
    long total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - t0; // optional intermediate time
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ================================================
    // -------- ZSYNCHRONIZUJ WSZYSTKIE PROCESY I ZAKOŃCZ POMIAR --------
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();           // === MPI timer ===
    // ================================================

    if (rank == 0) {
        printf("HYBRID: N=%ld procs=%d threads=%d full_time=%.6f s primes=%ld\n",
               N, nprocs, omp_get_max_threads(), t1 - t0, total_count);
    }

    // cleanup
    free(is_prime);
    free(primes);
    if (rank == 0) free(small_sieve);
    MPI_Finalize();                        // === MPI finalize ===
    return EXIT_SUCCESS;
}
