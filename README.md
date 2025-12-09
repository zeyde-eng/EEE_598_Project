# Vehicle Routing Problem with Stochastic Demands (VRPSD)

Branch-and-Price solution algorithm for the Vehicle Routing Problem with Stochastic Demands using Dantzig-Wolfe decomposition and column generation.

## Overview

This repository contains a Julia implementation of a Branch-and-Price algorithm for solving the VRPSD, based on the methodology from Christiansen & Lysgaard (2007). The algorithm uses:

- **Dantzig-Wolfe decomposition** with column generation
- **Forward labeling algorithm** with dominance rules for pricing subproblem
- **Customer-pair branching** strategy for set-covering formulation
- **Detour-to-Depot (DTD) recourse policy** for handling stochastic demands
- **Sample Average Approximation (SAA)** for computing expected recourse costs

## Author

**Zach Eyde** (Student ID: 1223877512)  
EEE 598 Final Project

## Key Features

### Problem Formulation

The VRPSD extends the classical Capacitated Vehicle Routing Problem (CVRP) by incorporating stochastic customer demands:

- Customers have expected demands with Poisson-distributed stochastic realizations
- Vehicle capacity constraints must be satisfied
- Fleet size is limited to K vehicles
- Objective: Minimize total expected cost (routing + recourse)

### Algorithm Components

#### 1. Master Problem (Restricted Master Problem - RMP)

**Set Covering Formulation:**
```
minimize: Σ c_r * y_r
subject to:
  Σ a_{i,r} * y_r ≥ 1    ∀ customers i   (each customer visited at least once)
  Σ y_r ≤ K                               (fleet size constraint)
  y_r ≥ 0                 ∀ routes r      (route selection variables)
```

**Note:** Current implementation uses set covering (≥1) which may allow duplicate customer visits in integer solutions. Changing to set partitioning (=1) would enforce each customer visited exactly once.

#### 2. Pricing Subproblem (Forward Labeling)

Dynamic programming with label-based state-space exploration:

- **Label Structure:** (node, load, visited_set, path, cost, recourse)
- **Dominance Rules:** Label L1 dominates L2 if:
  - Same current node
  - L1.visited ⊇ L2.visited (visits at least as many customers)
  - L1.load ≤ L2.load (uses less capacity)
  - L1.cost ≤ L2.cost (has lower cost)
- **Beam Search:** Maintains up to U_limit=1500 labels in queue for efficiency
- **Reduced Cost:** c̄ = cost - Σπᵢ - μ (dual values from master problem)

#### 3. Branching Strategy

**Customer-Pair Branching** on fractional "togetherness" values:

For customers i,j in fractional LP solution, compute togetherness:
```
T(i,j) = Σ_{r: i,j ∈ r} y_r
```

Select pair with T(i,j) closest to 0.5 and create two branches:
- **Left child:** Customers i,j must be on SAME route
- **Right child:** Customers i,j must be on DIFFERENT routes

#### 4. Recourse Cost Calculation

**Detour-to-Depot (DTD) Policy:**
- When vehicle exceeds capacity at customer i, returns to depot and resumes
- Expected recourse cost computed via Sample Average Approximation (SAA)
- 1000 demand scenarios sampled from Poisson distributions

**Expected Recourse Formula:**
```
E[Q(λ, q_remaining)] = (1/M) * Σ_{samples} DTD_cost(sample, q_remaining)
```

Where:
- λ: Poisson rate parameter
- q_remaining: Available vehicle capacity
- M: Number of samples (1000)

## Data Structure

### Input Files

Instance data stored in CSV format with columns:
- `node_id`: Node identifier (0-indexed in data, depot=0)
- `node_type`: "depot" or "customer"
- `x_coord`, `y_coord`: Euclidean coordinates
- `demand`: Expected demand value
- `parameters`: JSON-like string containing `{'lambda': rate}` for stochastic demands

Fleet specifications in separate `*_fleet.csv` files:
- Number of vehicles (K)
- Vehicle capacity (Q)

### Node Indexing Convention

**Internal Representation:**
- Node 1: Initial depot (route start)
- Nodes 2 to n+1: Customers (where n = number of customers)
- Node N: Duplicate depot (route end point)

**Output Files (0-indexed):**
- Node 0: Depot
- Nodes 1 to n: Customers

## Installation & Requirements

### Required Packages

```julia
using JuMP
using Gurobi  # Requires Gurobi license
using DataFrames
using CSV
using Distributions
using Random
using LinearAlgebra
using SparseArrays
using Distances
using Plots
using PlotlyJS
```

### Installing Packages

Uncomment the package installation line in the code:
```julia
import Pkg; Pkg.add("Distributions"); Pkg.add("DataFrames"); 
Pkg.add("CSV"); Pkg.add("Distances"); Pkg.add("PlotlyJS"); 
Pkg.add("Gurobi"); Pkg.add("DelimitedFiles"); Pkg.add("JuMP"); 
Pkg.add("LinearAlgebra"); Pkg.add("SparseArrays"); Pkg.add("Plots")
```

### Gurobi License

This implementation requires a Gurobi license. Academic licenses are available free at [gurobi.com/academia](https://www.gurobi.com/academia/).

## Usage

### Basic Execution

```julia
# Set instance parameters
input_list = ["A-n32-k5.csv", 5, 100.0]  # [filename, num_vehicles, capacity]

# Load and preprocess data
data, df, img = PreProcess("../data/new_dataset/" * input_list[1], 
                          input_list[2], input_list[3])

# Run Branch-and-Price
tic = time()
best_solution, best_ub, best_lb, final_gap, nodes_explored = branch_and_price(data)
toc = time()

# Generate results
solution_time = toc - tic
results = Results(best_solution, solution_time, best_ub, best_lb, 
                 final_gap, nodes_explored)
print_results_to_file(results, df)
```

### Key Parameters

- **U_limit**: Beam search limit for forward labeling (default: 1500)
- **max_cols_per_iter**: Maximum columns added per CG iteration (default: 300)
- **max_iter**: Maximum column generation iterations (default: 500)
- **SAA samples**: Number of demand scenarios for recourse estimation (default: 1000)

### Gurobi Parameters

```julia
set_optimizer_attribute(model, "OutputFlag", 0)      # Suppress output
set_optimizer_attribute(model, "TimeLimit", 300)     # 5-minute time limit
set_optimizer_attribute(model, "MIPGap", 0.0)        # 0% optimality gap
set_optimizer_attribute(model, "IntFeasTol", 1e-6)   # Integer feasibility tolerance
```

## Output Files

Results are automatically saved to `Results_BnP/<instance_name>/`:

### results.txt

Contains comprehensive solution information:
- **Summary Statistics:**
  - Objective value (total expected cost)
  - Upper and lower bounds
  - Optimality gap
  - Nodes explored in branch-and-bound tree
  - Computation time
  - Number of vehicles used
  - Column generation iterations

- **Route Details:**
  - Node sequences for each route (1-indexed internally)
  - Customer lists (0-indexed for output)
  - Route-specific information

- **Customer Coverage Analysis:**
  - Total customer visits
  - Unique customers served
  - Duplicate customer identification (if using set covering)

### Visualization Files

- `Customer_loc_w_nodes.png`: PlotlyJS plot of customer locations
- `Customer_routes_cg.png`: Route visualization with color-coded paths
- `fig_rc.png`: Reduced cost evolution during column generation
- `fig_obj.png`: Objective value convergence during column generation

## Benchmark Instances

The repository includes 18 VRPSD instances adapted from Christiansen & Lysgaard (2007):

### A-series (Augerat instances)
- A-n32-k5 through A-n60-k9
- 32-60 customers, 5-9 vehicles
- Capacity: 100
- Moderate demand variability

### E-series (Eilon instances)
- E-n22-k4, E-n33-k4, E-n51-k5
- 22-51 customers, 4-5 vehicles
- Large capacities (160-8000)
- Low recourse costs due to high capacity buffers

### P-series (Christofides & Eilon instances)
- P-n16-k8 through P-n60-k15
- 16-60 customers, 2-15 vehicles
- Variable capacities

## Algorithm Performance

### Typical Results (from Results_BnP/)

| Instance | Benchmark | Our Cost | Gap | Vehicles | Time (s) | Nodes |
|----------|-----------|----------|-----|----------|----------|-------|
| A-n32-k5 | 784.00 | 922.90 | 17.7% | 5 | 3,628.64 | 97 |
| A-n48-k7 | 1073.00 | 1335.45 | 24.5% | 7 | 246.54 | 5 |
| E-n51-k5 | 521.00 | 593.73 | 14.0% | 5 | 9,979.51 | 500 |

**Note:** Current implementation shows ~20% gaps above benchmarks for A-series instances, primarily due to:
1. Set covering formulation allowing duplicate customers
2. SAA approximation of recourse costs
3. Beam search pruning in pricing subproblem

E-series instances perform better due to large vehicle capacities minimizing recourse costs.

## Known Issues & Future Improvements

### Set Covering vs Set Partitioning

**Current Issue:** The RMP uses set covering (≥1) constraints, which allow customers to appear in multiple routes in integer solutions.

**Fix:** Change line 532 in the code from:
```julia
@constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) >= 1)
```
to:
```julia
@constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) == 1)
```

This enforces set partitioning where each customer is visited exactly once.

### Potential Enhancements

1. **Exact Recourse Calculation:** Replace SAA with closed-form expressions or dynamic programming
2. **Advanced Branching:** Implement strong branching or hybrid branching strategies
3. **Route Enumeration:** Add complete enumeration for small instances as verification
4. **Parallel Pricing:** Solve multiple pricing subproblems in parallel
5. **Column Pool Management:** Implement column deletion strategies for memory efficiency
6. **Warm Start Heuristics:** Use advanced construction heuristics (Clarke-Wright, sweep algorithm)

## References

- Christiansen, C. H., & Lysgaard, J. (2007). A branch-and-price algorithm for the capacitated vehicle routing problem with stochastic demands. *Operations Research Letters*, 35(6), 773-781.

## File Structure

```
EEE_598_Project/
├── code/
│   ├── EEE 598 CG Branch and Price rev3.jl   # Main implementation
│   └── VRP-REP to CSV.py                      # Data preprocessing script
├── data/
│   ├── christiansen-and-lysgaard-2007/        # Original XML instances
│   └── new_dataset/                           # Converted CSV instances
├── Results_BnP/                               # Solution outputs
├── Papers/                                    # Reference papers
├── pseudocode.tex                             # LaTeX algorithm documentation
├── instance_table.tex                         # LaTeX instance specifications
├── results_table.tex                          # LaTeX results comparison
├── A-n32-k5_routes.tex                        # LaTeX route visualizations
└── README.md                                  # This file
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please contact Zach Eyde.

---

**Note:** This is an academic implementation for EEE 598 course project. For production use, consider the improvements listed above and thorough validation against benchmark instances.
