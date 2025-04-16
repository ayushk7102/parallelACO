# 15-418/618 Spring 25 - Project Milestone Report

**By:** Naveen Shenoy (naveensh) and Ayush Kumar (ayushkum)

## TITLE:
Parallelized Community Detection in Graphs via Ant Colony Optimization

**URL:** [https://www.andrew.cmu.edu/user/ayushkum/parallelACO/](https://www.andrew.cmu.edu/user/ayushkum/parallelACO/)

## Work Completed:
We have a working sequential and OpenMP parallel version of community detection in graphs using Ant Colony Optimization (ACO). We have found large community-based graph datasets using which we can benchmark our community finding algorithm. We have devised approaches to evaluate the formed communities using the modularity metric as well as based on the number of communities identified. Implementing the sequential version of ACO for graph community detection was a non-trivial component of our completed work and consumed the initial week along with testing out implementation strategies and datasets. Currently, our OpenMP version does provide a speedup as the number of threads are increased while maintaining solution modularity, though we would like to further optimize this and make our algorithm more scalable for increased number of cores, i.e. more parallelizable.

## Current Parallelization Approach:
Currently, we have parallelized graph community selection across ants as well as pheromone updation across the graph nodes after all ants complete their walks over the solution space. We see additional scope to parallelize the algorithm in the solution construction phase where an ant has to select the best community for a node based on neighboring nodes and their communities. Due to the varying density of edges in a graph, this can lead to workload imbalance if work is distributed across multiple nodes for an ant. We plan to utilize OpenMP tasks to achieve a more balanced workload distribution across threads.

## Preliminary Results:
The following is a visualization of the resulting communities detected by our OpenMP based version of ACO graph community detection. The graph depicts a network of American football games between Division IA colleges during regular season Fall 2000, as compiled by M. Girvan and M. Newman. Current results are on a personal computer (Apple M3, 8 cores and 16 GB memory).

The graph has 115 nodes and 306 edges. 

### Hyperparameters:
- 20 ants
- 200 iterations
- Alpha (pheromone influence): 1
- Beta (heuristic influence): 2
- Rho (evaporation rate): 0.05
- q0 (exploitation probability): 0.7

Time taken by Sequential Implementation: 3.4318 secs

| Number of Threads | 1 | 2 | 4 | 8 |
|-------------------|---|---|---|---|
| Modularity | 0.5735 | 0.5654 | 0.5651 | 0.5666 |
| Time taken (in sec) | 7.113 | 3.978 | 2.280 | 2.208 |

## Goals and Deliverables:
- Finalize the OpenMP parallel version.
- Thoroughly benchmark OpenMP ACO-based parallel graph community detection
- Analyse performance, bottlenecks, speedup, workload imbalance and other factors affecting algorithm performance
- Get started and implement an MPI version for community detection using ACO
- Benchmark MPI solution and analyse performance results
- Experiment with alternative algorithm (betweenness centrality using Girvan-Newman method) for community selection 
- Create additional plots and visualizations for poster

## Revised Project Schedule:

| Week | Plan |
|------|------|
| 4/15 - 4/18 | Optimize OpenMP Parallel Code (Naveen)<br>Benchmark OpenMP Implementation Performance on GHC and PSC Machines (Ayush)<br>Start MPI Implementation (Naveen, Ayush) |
| 4/18 - 4/21 | Finish MPI Parallelization (Naveen, Ayush)<br>Optimize MPI Community Detection (Ayush)<br>Benchmark MPI Implementation Performance on GHC and PSC Machines (Naveen) |
| 4/21 - 4/24 | Final Evaluation and Result Collection (Ayush, Naveen)<br>Drafting Final Report (Ayush, Naveen) |
| 4/24 - 4/27 | Finalizing Report (Ayush, Naveen)<br>Create Poster (Ayush, Naveen) |

## Poster Session:

### Demonstrations:
At the poster session, we plan to show a few visualizations of community detection on various networks. For the purpose of demonstrating the functionality of our algorithms, we plan to use smaller graphs such as the 115-node college football network dataset (M. Girvan et. al) and the 1005-node EU research institute emails dataset (J. Kleinberg et al). As an additional demonstration, we plan to tweak our ant colony optimization parameters (such as ant counts, iterations) and display the impact of each parameter on the distribution of the communities detected by the algorithm. Also, we may tweak our algorithm hyperparameters (alpha, beta, rho) to demonstrate how these perturb the community detection. We also plan to check how scaling graph community detection with a large number of parallel threads affects the runtime of the algorithm and quality of detected communities.

### Graphs:
We also plan to show graphs analyzing the performance and locality of our OpenMP version of the algorithm, by plotting the number of threads vs computation speedup, number of threads vs per-thread cache misses. Also, for a fixed number of threads, we plan to show the computation speedup as we increase the number of ants, and the number of iterations. 

## Challenges:

### Complexity of community detection on large graphs: 
We noticed that the runtime of the sequential algorithm is impractical on large networks (such as the DBLP dataset, 300,000+ nodes). As a result, it is not easy to benchmark our parallel algorithms on such large graphs, and we have constrained our benchmarking for graphs with under 10,000 nodes for now. This is an issue with respect to the nature of the problem itself, which has ~ O (N^2) complexity as the size of the graphs increases.

### Parameter tuning complexity: 
The result of the community detection algorithm in terms of modularity as well as in comparison to ground truth communities (when available) depends highly on the algorithm's parameters, such as alpha, beta, rho. In addition, the effect of number of ants as well as number of iterations can be counter-intuitive with respect to the other parameters, which makes tuning the parameters quite a labor-intensive and time consuming task. We plan to run more experiments and review prior literature to develop more intuition behind the parameters, which will allow us to develop a better algorithm.

### Workload imbalance: 
In our current OpenMP implementation, we find that there is potential for workload imbalance to impede the performance of our algorithm, particularly when there are graphs which have heterogeneous clusters of varying edge density. To circumvent this, we plan on using the OpenMP task and taskloop constructs.
