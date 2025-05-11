//HOW TO COMPILE: gcc -O3 -fopenmp sieve_omp.c -o ../bin/sieve_omp -lm
//HOW TO RUN: OMP_NUM_THREADS=4 ./bin/sieve_omp 10000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    long N = atol(argv[1]);

    // Start full execution timing
    double t0 = omp_get_wtime();

    // Alokujemy tablicę znaków (1 = potencjalnie pierwsza, 0 = złożona)
    char *is_prime = malloc((N + 1) * sizeof(char));
    if (!is_prime) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    // ****************************************************************************
    // 1) Inicjalizacja tablicy: na początku zakładamy, że wszystkie liczby [0..N]
    //    są pierwsze (ustawiamy 1), a potem ręcznie oznaczamy 0 i 1 jako nie-pierwsze
    // ****************************************************************************
    for (long i = 0; i <= N; i++) {
        is_prime[i] = 1;
    }
    is_prime[0] = is_prime[1] = 0;

    long limit = (long) sqrt(N);

    // ****************************************************************************
    // 2) Główna pętla sita:
    //    dla każdej liczby p od 2 do sqrt(N),
    //    jeśli p jest dalej oznaczone jako pierwsze,
    //    to wymazywane są jego wielokrotności.
    // ****************************************************************************
    for (long p = 2; p <= limit; p++) {
        if (is_prime[p]) {
            // ****************************************************************************
            // 3) Parallel region OpenMP:
            //    wszystkie wielokrotności p (od p*p do N, co p)
            //    są wymazywane równolegle w wątkach.
            //    schedule(dynamic) – każde zadanie to usunięcie bloku liczb,
            //                       może lepiej rozłożyć pracę jeśli p jest małe.
            // ****************************************************************************
            #pragma omp parallel for schedule(dynamic)
            for (long m = p * p; m <= N; m += p) {
                is_prime[m] = 0;
            }
        }
    }

    // ****************************************************************************
    // 4) Liczymy, ile liczb pierwszych znalazło się w tablicy
    // ****************************************************************************
    long count = 0;
    for (long i = 2; i <= N; i++) {
        if (is_prime[i]) {
            count++;
        }
    }

    // End full execution timing
    double t1 = omp_get_wtime();

    // Wypisujemy czas i – na stderr – liczbę znalezionych pierwszych
    printf("%f\n", t1 - t0);
    fprintf(stderr, "Primes up to %ld: %ld\n", N, count);

    free(is_prime);
    return EXIT_SUCCESS;
}
