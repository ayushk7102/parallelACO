#!/bin/bash
#SBATCH -p RM
#SBATCH -N 2  # Adjust based on process count
#SBATCH -t 2:00:00
#SBATCH -C PERF  # Important flag for profiling
module load openmpi/4.0.2-gcc8.3.1
module load intel

# Memory consumption analysis
mpirun -np 64 amplxe-cl -result-dir mpi_mem_consumption -quiet -collect memory-consumption ./mpi_aco 300 500

# Memory access patterns (important for pheromone matrix access)
mpirun -np 64 amplxe-cl -result-dir mpi_mem_access -quiet -collect memory-access ./mpi_aco 300 500

# Memory bandwidth analysis
mpirun -np 64 amplxe-cl -result-dir mpi_bandw -quiet -collect uarch-exploration -knob collect-memory-bandwidth=true ./mpi_aco 300 500
