//HOW TO COMPILE: mpicc -O3 src/sieve_mpi.c -o bin/sieve_mpi -lm
//HOW TO RUN: mpirun -np 4 bin/sieve_mpi 10000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
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
    long size = N - 1;              // liczby od 2 do N inclusive
    long chunk = size / nprocs;     // podział na bloki
    long rem   = size % nprocs;

    // Każdy proces dostaje albo chunk+1, albo chunk elementów:
    long low_index, high_index, local_size;
    if (rank < rem) {
        local_size = chunk + 1;
        low_index  = rank * local_size;
    } else {
        local_size = chunk;
        low_index  = rem * (chunk + 1) + (rank - rem) * chunk;
    }
    high_index = low_index + local_size - 1;

    long low_value  = 2 + low_index;
    long high_value = 2 + high_index;

    // ****************************************************************************
    // 1) Proces 0 buduje listę małych liczb pierwszych <= sqrt(N)
    // ****************************************************************************
    long limit = (long) sqrt(N);
    char *small_sieve = NULL;
    long  small_count = 0;
    long *primes = NULL;

    if (rank == 0) {
        small_sieve = malloc((limit + 1) * sizeof(char));
        for (long i = 0; i <= limit; i++) small_sieve[i] = 1;
        small_sieve[0] = small_sieve[1] = 0;

        for (long p = 2; p * p <= limit; p++) {
            if (small_sieve[p]) {
                for (long m = p * p; m <= limit; m += p)
                    small_sieve[m] = 0;
            }
        }
        // zbierz wszystkie małe liczby pierwsze do tablicy
        for (long i = 2; i <= limit; i++)
            if (small_sieve[i]) small_count++;
        primes = malloc(small_count * sizeof(long));
        long idx = 0;
        for (long i = 2; i <= limit; i++)
            if (small_sieve[i]) primes[idx++] = i;
    }

    // ****************************************************************************
    // 2) Broadcastujemy small_count i tablicę primes do wszystkich procesów
    // ****************************************************************************
    MPI_Bcast(&small_count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) primes = malloc(small_count * sizeof(long));
    MPI_Bcast(primes, small_count, MPI_LONG, 0, MPI_COMM_WORLD);

    // -------- ZSYNCHRONIZUJ WSZYSTKIE PROCESY --------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // ****************************************************************************
    // 3) Każdy proces alokuje swoją pod-tablicę znaków i inicjalizuje ją na 1
    // ****************************************************************************
    char *is_prime = malloc(local_size * sizeof(char));
    for (long i = 0; i < local_size; i++)
        is_prime[i] = 1;

    // ****************************************************************************
    // 4) Zrównoleglone wymazywanie: każdy proces pracuje na swoim przedziale
    //    dla każdej małej liczby pierwszej p:
    //      - oblicza pierwszy wielokrotny >= low_value
    //      - wymazuje kolejne w krokach co p
    // ****************************************************************************
    for (long i = 0; i < small_count; i++) {
        long p = primes[i];
        // najmniejszy wielokrotny p >= low_value
        long first = (low_value + p - 1) / p * p;
        if (first < p * p) first = p * p;
        for (long m = first; m <= high_value; m += p) {
            is_prime[m - low_value] = 0;
        }
    }

    long local_count = 0;
    for (long i = 0; i < local_size; i++)
        if (is_prime[i]) local_count++;

    long total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - t0;  // not used for output
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("MPI: N=%ld procs=%d full_time=%.6f s primes=%ld\n",
               N, nprocs, t1 - t0, total_count);
    }

    // Sprzątanie
    free(is_prime);
    free(primes);
    if (rank == 0) free(small_sieve);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
