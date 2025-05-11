#!/usr/bin/env python3
import os
import subprocess
import csv
import re

# Script runs OpenMP, MPI, and hybrid binaries for various N, processes, threads
# Collects results into results.csv, handles oversubscription and errors

BIN_DIR = 'bin'
MAX_THREADS = int(subprocess.getoutput('nproc --all'))

# Function to run OpenMP variant

def run_openmp(N, threads):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    cmd = [os.path.join(BIN_DIR, 'sieve_omp'), str(N)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        print(f"[ERROR OpenMP] N={N}, threads={threads}\n", proc.stderr.strip())
        return None, None
    # parse stdout: time in seconds
    try:
        time_s = float(proc.stdout.strip())
    except ValueError:
        print(f"[WARN OpenMP] Unexpected stdout: {proc.stdout}")
        time_s = None
    # parse stderr: "Primes up to N: count"
    m = re.search(r'Primes up to \d+: (\d+)', proc.stderr)
    primes = int(m.group(1)) if m else None
    return time_s, primes

# Function to run MPI variant

def run_mpi(N, procs):
    cmd = ['mpirun', '--oversubscribe', '-np', str(procs), os.path.join(BIN_DIR, 'sieve_mpi'), str(N)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[ERROR MPI] N={N}, procs={procs}\n", proc.stderr.strip())
        return None, None
    # parse stdout: "MPI: ... time=X s primes=Y"
    m = re.search(r'time=([\d\.]+) s primes=(\d+)', proc.stdout)
    if not m:
        print(f"[WARN MPI] Unexpected stdout: {proc.stdout}")
        return None, None
    time_s, primes = float(m.group(1)), int(m.group(2))
    return time_s, primes

# Function to run hybrid MPI+OpenMP

def run_hybrid(N, procs, threads):
    # skip oversubscription beyond hardware threads
    # Dodano --oversubscribe w wywołaniach MPI i hybrydzie, aby uniknąć blokady przy większej liczbie procesów.
    if procs * threads > MAX_THREADS:
        print(f"[SKIP Hybrid] N={N}, procs={procs}, threads={threads} (oversubscribe)")
        return None, None
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    cmd = ['mpirun', '--oversubscribe', '-np', str(procs), os.path.join(BIN_DIR, 'sieve_hybrid'), str(N)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        print(f"[ERROR Hybrid] N={N}, procs={procs}, threads={threads}\n", proc.stderr.strip())
        return None, None
    # parse stdout: "HYBRID: ... time=X s primes=Y"
    m = re.search(r'time=([\d\.]+) s primes=(\d+)', proc.stdout)
    if not m:
        print(f"[WARN Hybrid] Unexpected stdout: {proc.stdout}")
        return None, None
    time_s, primes = float(m.group(1)), int(m.group(2))
    return time_s, primes

# Main: iterate configurations, write CSV

def main():
    Ns = [10**6, 5*10**6, 10**7]
    threads_list = [1, 2, 4, 6, 8, 12]
    procs_list = [1, 2, 4, 6, 8, 12]

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['variant', 'N', 'procs', 'threads', 'time_s', 'primes'])

        for N in Ns:
            # OpenMP
            for t in threads_list:
                time_s, primes = run_openmp(N, t)
                writer.writerow(['openmp', N, 1, t, time_s, primes])
            # MPI
            for p in procs_list:
                time_s, primes = run_mpi(N, p)
                writer.writerow(['mpi', N, p, 1, time_s, primes])
            # Hybrid
            for p in procs_list:
                for t in threads_list:
                    time_s, primes = run_hybrid(N, p, t)
                    writer.writerow(['hybrid', N, p, t, time_s, primes])

if __name__ == '__main__':
    main()
