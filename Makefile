EXECS=mpi_aco
MPICXX?=mpicxx

# Compiler flags
CXXFLAGS=-std=c++11 -O3 -Wall

all: ${EXECS}

mpi_aco: mpi_aco.cpp
	${MPICXX} ${CXXFLAGS} -o mpi_aco mpi_aco.cpp

clean:
	rm -f ${EXECS}