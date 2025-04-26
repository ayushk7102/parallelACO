#!/bin/bash
#SBATCH --job-name=mpi_aco_procs_
#SBATCH --output=mpi_aco_procs_%j.out
#SBATCH --error=mpi_aco_procs_%j.err
#SBATCH --time=12:00:00

# Change to working directory
# cd /jet/home/akumar27/expts/project/parallelACO


# Load OpenMPI module
# module load openmpi

# Compile the code
# make


# Run with different numbers of processes
echo "Starting MPI ACO scaling tests with 50 ants, 500 iterations"
echo "=================================================="

for procs in 1 2 4 8 16 32 64 128
do
    echo "Running with $procs processes..."
    # srun --ntasks=$procs ./mpi_aco 50 500
    # mpirun -n $procs ./mpi_aco $(($procs * 10)) 500
    mpirun -n $procs ./mpi_aco 200 500

    echo "------------------------------------------------"
done

echo "All tests complete!"

# 1 
# Iteration 499: 100 communities, modularity = 0.100589
# Final result: 100 communities, modularity = 0.100589

# MPI Node-Community ACO completed in 10.6707 seconds
# Detected 100 communities:


# Iteration 499: 89 communities, modularity = 0.13665
# Final result: 89 communities, modularity = 0.13665

# MPI Node-Community ACO completed in 12.0544 seconds
# Detected 89 communities:

# 2
# Community 185 (size: 117): 12 30 33 35 37 51 55 56 57 60 ...
# Community 146 (size: 52): 2 32 36 52 62 72 91 121 125 132 ...
# Community 226 (size: 26): 6 24 28 29 40 41 45 48 63 64 ...
# Community 173 (size: 17): 39 77 117 171 172 285 286 287 288 289 ...
# Community 407 (size: 15): 9 10 73 96 143 298 299 339 341 382 ...
# Community 5 (size: 14): 3 5 107 156 157 158 166 168 169 240 ...
# Community 227 (size: 13): 23 27 43 46 49 104 105 106 190 214 ...
# Community 228 (size: 12): 97 159 185 203 212 257 258 261 282 295 ...
# Community 351 (size: 10): 14 18 19 20 21 22 351 352 353 355 
# Community 284 (size: 10): 54 93 149 150 151 284 307 372 373 374 
# ... (79 more communities)
# Graph with communities saved to: mpi_communities.txt
# ------------------------------------------------
