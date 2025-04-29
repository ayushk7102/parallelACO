# 15-418/618 Spring 25 - Project Milestone Report

**By:** Naveen Shenoy (naveensh) and Ayush Kumar (ayushkum)

## TITLE:
Parallelized Community Detection in Graphs via Ant Colony Optimization

### Instructions to run:

MPI:
'''
make
mpirun -n <num_threads> ./mpi_aco <num_ants> <num_iterations> <alpha> <beta> <rho> <evaporation_ratio>
'''

OpenMP:
'''
g++ parallelACO.cpp -o parallelACO -fopenmp
./parallelACO <num_ants> <num_iterations> <alpha> <beta> <rho> <evaporation_ratio>
'''

Sequential:
'''
g++ sequential.cpp -o sequential
./sequential
'''
