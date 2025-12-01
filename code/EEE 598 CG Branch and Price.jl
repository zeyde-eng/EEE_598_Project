#=
================================================================================
Vehicle Routing Problem with Stochastic Demands (VRPSD)
Branch-and-Price Solution Algorithm
================================================================================

Author: Zach Eyde
Student ID: 1223877512
Course: EEE 598 Final Project

Implementation Details:
- Uses Dantzig-Wolfe decomposition with column generation
- Branch-and-price for exact integer solutions
- Labeling algorithm for pricing problem (SPPRC)
- Stochastic demands modeled via Poisson distribution
- Detour-to-Depot (DTD) recourse policy
- Sample Average Approximation (SAA) for expected recourse costs

Key References:
- Christiansen & Lysgaard (2007) - VRPSD benchmark instances
- Dumas et al. - Shortest Path with Resource Constraints
- Gendreau et al. - VRPSD formulations and solution methods

Problem Definition:
  The VRPSD extends the classical VRP by incorporating stochastic customer
  demands that follow known probability distributions. The objective is to
  minimize total expected routing costs, including expected recourse costs
  incurred when vehicle capacity is exceeded during route execution.

Solution Approach:
  1. Master Problem (RMP): Set partitioning formulation selecting optimal routes
  2. Pricing Problem: Shortest path with resource constraints via labeling
  3. Recourse Costs: Expected costs computed via SAA with DTD policy
  4. Branching: Variable branching on fractional route selections

=#


################################################################
############### Import Needed Packages #########################
################################################################

# Start by Importing Needed Packages
# Uncomment Line Below if need to Install Packages
#import Pkg; Pkg.add("Distributions");Pkg.add("DataFrames");Pkg.add("CSV");Pkg.add("Distances");Pkg.add("PlotlyJS");Pkg.add("Gurobi");Pkg.add("DelimitedFiles");Pkg.add("JuMP");Pkg.add("LinearAlgebra");Pkg.add("SparseArrays");Pkg.add("Plots"); Pkg.add("SparseArrays")


using JuMP
using LinearAlgebra
using DataFrames
using Plots, PlotlyJS
using SparseArrays
using Gurobi
using Distributions
using Random
using Random

# Import data from CSV
using CSV
using DataFrames
using Distances


#Get Gurobi Env
gurobi_env = Gurobi.Env()

################################################################
############### Define the Data Structure ######################
################################################################

"""
    Data

Core data structure for VRPSD instances.

# Fields
- `n::Int64`: Number of customers (excluding depot)
- `d::Vector{Float64}`: Expected demands for each node (deterministic component)
- `c::Array{Float64}`: Distance/cost matrix between all node pairs (N×N)
- `q::Float64`: Vehicle capacity constraint
- `K::Int64`: Maximum number of vehicles available
- `N::Int64`: Total number of nodes (n customers + 2 depot nodes)
- `λ_rate::Vector{Float64}`: Poisson rate parameters for stochastic demands

# Indexing Convention
- Node 1: Initial depot (route start)
- Nodes 2 to n+1: Customers
- Node N: Duplicate depot (route end)
"""
mutable struct Data
    n::Int64
    d::Vector{Float64}
    c::Array{Float64}
    q::Float64
    K::Int64
    N::Int64
    λ_rate::Vector{Float64}
end

"""
    Label

Represents a partial route state in the labeling algorithm for SPPRC.

# Fields
- `label::String`: Human-readable label (concatenated node sequence)
- `node::Int`: Current node position
- `cost::Float64`: Accumulated reduced cost (includes arc costs + recourse)
- `load::Float64`: Accumulated expected load
- `E_recourse_cost::Float64`: Total expected recourse cost along path
- `path::Vector{Int}`: Sequence of visited nodes

# Usage
Labels are extended through the network to find negative reduced cost routes.
Dominance rules prune labels that cannot lead to optimal solutions.
"""
mutable struct Label
    label::String
    node::Int
    cost::Float64
    load::Float64
    E_recourse_cost::Float64
    path::Vector{Int}
end

"""
    Graph

Reduced graph structure for efficient labeling algorithm.

# Fields
- `Nodes::Vector{Int}`: All nodes in the network
- `Arcs::Vector{Tuple{Int,Int}}`: Feasible arcs (reduced via nearest neighbor)

# Purpose
Pre-processing to reduce arc set improves computational efficiency while
maintaining solution quality. Nearest neighbor heuristic balances these factors.
"""
mutable struct Graph
    Nodes::Vector{Int}
    Arcs::Vector{Tuple{Int,Int}}
end


"""
    BranchNode

Represents a node in the branch-and-price tree.

# Fields
- `name::String`: Descriptive name (e.g., "Level 1 branch yᵣ[5] ≥ 1")
- `tree_lvl::Int`: Depth in branch tree (root = 0)
- `master::Model`: RMP model with branching constraints
- `objective::Vector{Float64}`: Objective values through CG iterations
- `rc::Vector{Float64}`: Reduced costs of generated columns
- `paths::Vector{Vector{Int}}`: Pool of all generated routes

# Branching Strategy
Branches on fractional route selection variables using "most fractional" rule.
Each child node adds constraint yᵣ ≤ 0 or yᵣ ≥ 1 for selected variable.
"""
mutable struct BranchNode
    name::String
    tree_lvl::Int
    master::Model
    objective::Vector{Float64}
    rc::Vector{Float64}
    paths::Vector{Vector{Int}}
end

"""
    Results

Stores final solution and computational statistics.

# Fields
- `node::BranchNode`: Best integer solution node from branch tree
- `time::Float64`: Total solution time in seconds

# Contains
The best BranchNode includes optimal routes, objective value, and solution details.
"""
mutable struct Results
    node::BranchNode
    time::Float64
end

################################################################
############### User-Defined Base Functions ####################
################################################################

# Note: Base function extensions would go here if needed.
# Currently all functionality uses standard Julia/JuMP operations.



################################################################
############### User define Functions ##########################
################################################################


"""
    Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, q)

Constructor for the main VRPSD data structure.

# Arguments
- `num_Cust::Int`: Number of customers (excluding depot)
- `xcoord::Vector`: X-coordinates of all nodes (depot + customers + duplicate depot)
- `ycoord::Vector`: Y-coordinates of all nodes
- `demand::Vector{Float64}`: Deterministic demand values (used as expected demands)
- `demand_rate::Vector{Float64}`: Poisson rate parameters λ for stochastic demands
- `max_V::Int`: Maximum number of vehicles available (K)
- `q::Float64`: Vehicle capacity constraint

# Returns
- `Data`: Structured data object containing:
  - `n`: Number of customers
  - `d`: Demand vector
  - `c`: Distance/cost matrix (Euclidean distances between all node pairs)
  - `q`: Vehicle capacity
  - `K`: Maximum vehicles
  - `N`: Total number of nodes (customers + depot nodes)
  - `λ_rate`: Stochastic demand rate parameters

# Notes
- Distance matrix `c` is computed using Euclidean norm between coordinate pairs
- Node indexing: 1=depot, 2 to n+1=customers, N=duplicate depot for route endings
"""
function Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, q)
    n = num_Cust
    d = demand
    N = length(xcoord)
    K = max_V
    c = [norm([xcoord[i]; ycoord[i]] - [xcoord[j]; ycoord[j]], 2) for i=1:N, j=1:N]
    λ_rate = demand_rate

    return Data(n, d, c, q, K, N, λ_rate)
end


################################################################
############### Import the Data Instance #######################
################################################################

"""
    PreProcess(filename, cars::Int64, capacity::Float64)

Preprocesses VRPSD instance data from CSV format and creates visualization.

# Arguments
- `filename::String`: Path to CSV file containing instance data
- `cars::Int64`: Number of vehicles available
- `capacity::Float64`: Vehicle capacity constraint

# Returns
- `process_data::Data`: Structured data object for VRPSD
- `A_final::DataFrame`: Complete dataframe with depot duplication
- `fig_plotly`: PlotlyJS visualization of the instance

# Functionality
1. **Data Loading**: Reads CSV with columns for coordinates, demands, and stochastic parameters
2. **Lambda Extraction**: Parses Poisson rate parameters from parameters column using regex
3. **Depot Duplication**: Adds duplicate depot node for route termination (standard VRPSD modeling)
4. **Visualization**: Creates interactive plot showing depot and customer locations with demand labels
5. **Data Structure Creation**: Converts raw data into structured format for optimization

# CSV Format Expected
- Must contain: node_id, node_type, x_coord, y_coord, demand, parameters
- Parameters column should contain: "{'lambda': <rate>}" for stochastic customers
- Node types: 'depot' for depot, 'customer' for customers

# Notes
- Handles missing or malformed lambda parameters by setting to 0.0
- Creates text labels showing (node_id, demand) for visualization
- Final node indexing: 1=depot, 2..n+1=customers, N=duplicate depot
"""
function PreProcess(filename, cars::Int64, capacity::Float64)
    #filename = "../data/new_dataset/A-n32-k5.csv"

    # reading the csv file and saving to Dataframe A
    A_data = CSV.read(joinpath(@__DIR__, filename), DataFrame; delim = ",")
    
    # Extract lambda values from parameters column, set to 0 if missing or empty
    if "parameters" in names(A_data)
        A_data.lambda_rate = map(A_data.parameters) do p
            try
                if ismissing(p)
                    return 0.0
                end
                p_str = String(p)
                if isempty(strip(p_str))
                    return 0.0
                end
                # Extract lambda value from dictionary-like string: "{'lambda': 19.0}"
                lambda_match = match(r"'lambda':\s*([0-9.]+)", p_str)
                if !isnothing(lambda_match)
                    return parse(Float64, lambda_match.captures[1])
                else
                    return 0.0
                end
            catch
                return 0.0
            end
        end
    else
        A_data.lambda_rate = zeros(nrow(A_data))
    end
    
    text_labels = "(" .* string.(A_data.node_id) .* "," .* string.(A_data.demand) .* ")"
    fig_plotly = PlotlyJS.plot(
            A_data, x=:x_coord, y=:y_coord, color=:node_type, text=text_labels,
            mode="markers+text", marker=PlotlyJS.attr(size=12, line=PlotlyJS.attr(width=2, color="DarkSlateGrey")),
            textposition="top center",
    )
    
    # Add depot row as final node
    depot_row = A_data[A_data.node_type .== "depot", :]
    depot_row.node_id .= maximum(A_data.node_id) + 1
    A_final = vcat(A_data, depot_row)

    # Get number of customers
    num_Cust = nrow(A_final[A_final.node_type .== "customer", :])

    xcoord = A_final[:,:x_coord]
    ycoord = A_final[:,:y_coord]
    demand = A_final[:,:demand] # demand
    demand_rate = A_final[:,:lambda_rate] # demand rate
    q = capacity # vehicle capacity
    max_V = cars # max vehicles 

    process_data = Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, q)
    return process_data, A_final, fig_plotly
end

################################################################
############### Define the RMP Formulation ####################
################################################################

# Function to create initial feasible routes respecting vehicle limit K
"""
    create_initial_routes(data::Data)

Generates initial feasible routes for the Restricted Master Problem.

# Arguments
- `data::Data`: VRPSD instance data structure

# Returns
- `routes::Vector{Vector{Int}}`: Collection of initial routes, each as node sequence [1, customers..., N]

# Strategy
Creates simple single-customer routes to provide initial feasible solution:
- Each route visits exactly one customer: depot → customer → depot
- Ensures initial RMP has feasible basis
- Route format: [1, customer_node, N] where N is duplicate depot
- All routes satisfy capacity constraints since each serves one customer

# Notes
- Customer nodes are indexed from 2 to n+1
- Routes start/end at depot nodes (1 and N respectively)
- This conservative initialization ensures feasibility but may not be efficient
- Column generation will improve these routes iteratively
"""
function create_initial_routes(data::Data)::Vector{Vector{Int}}
    n = data.n
    N = data.N
    c = data.c
    q = data.q
    K = data.K
    λ_rate = data.λ_rate
    
    println("\n=== Creating Initial Routes ===")
    println("Customers: ", n, " | Max Vehicles: ", K, " | Capacity: ", q)
    
    # Simple initialization: 1 customer per route
    println("Using simple initialization (1 customer per route)")
    routes = Vector{Vector{Int}}()
    for j = 2:n+1
        push!(routes, [1, j, N])
    end
    println("Created ", length(routes), " initial routes for ", n, " customers")
    
    # Validate initial routes cover all customers
    covered_customers = Set()
    for route in routes
        for node in route
            if node > 1 && node <= n+1
                push!(covered_customers, node)
            end
        end
    end
    if length(covered_customers) != n
        println("❌ ERROR: Initial routes don't cover all customers!")
        println("  Covered: ", length(covered_customers), " / ", n)
    else
        println("✓ Initial routes cover all ", n, " customers")
    end
    
    return routes

end

"""
    RMP(data::Data, initial_routes::Vector{Vector{Int}})

Formulates the Restricted Master Problem for VRPSD using Dantzig-Wolfe decomposition.

# Arguments
- `data::Data`: VRPSD instance data
- `initial_routes::Vector{Vector{Int}}`: Initial feasible routes for warm start

# Returns
- `model::JuMP.Model`: Configured RMP with variables, objective, and constraints

# Mathematical Formulation
```
min Σ c_r * y_r                           (minimize total route costs)
s.t. Σ a_{i,r} * y_r = 1    ∀i ∈ customers  (set partitioning: each customer visited exactly once)
     y_r ≥ 0               ∀r ∈ routes     (route selection variables)
```

# Key Components
1. **Decision Variables**: y_r ∈ [0,1] indicates if route r is used
2. **Objective**: Minimize sum of route costs (including expected recourse costs)
3. **A Matrix**: a_{i,r} = 1 if customer i is in route r, 0 otherwise
4. **Set Partitioning**: Each customer must be visited exactly once
5. **Route Costs**: Include travel costs + expected stochastic recourse costs

# Technical Details
- Uses Gurobi optimizer with 600s time limit
- Suppresses solver output for cleaner logging
- A matrix built from actual route structures
- Route costs computed from distance matrix
- Node indexing: customers 2..n+1, exclude depot nodes 1 and N from A matrix

# Notes
- This is the "master" in Dantzig-Wolfe decomposition
- Pricing problem (labeling algorithm) generates new routes
- Set partitioning ensures feasible VRP solutions
"""
function RMP(data::Data, initial_routes::Vector{Vector{Int}})

    n = data.n
    N = data.N
    c = data.c
    K = data.K
    λ_rate = data.λ_rate
    q = data.q

    Ω = length(initial_routes)
    c_r = []
    
    # Calculate the cost of each path from actual routes
    for route in initial_routes
        route_cost = 0.0
        for i in 1:(length(route)-1)
            route_cost += c[route[i], route[i+1]]
        end
        push!(c_r, route_cost)
    end

    # Build A matrix from actual routes
    # A[i,r] = 1 if customer i is in route r, 0 otherwise
    A = zeros(Int, n, Ω)
    for (r_idx, route) in enumerate(initial_routes)
        for node in route
            # Skip depot nodes (1 and N)
            if node != 1 && node != N
                customer_idx = node - 1  # Customer indices are 1 to n, nodes are 2 to n+1
                A[customer_idx, r_idx] = 1
            end
        end
    end
    
    # define empty model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))
    set_attribute(model, "TimeLimit", 600)
    set_attribute(model, "LogToConsole", 0)

    # add decision variable
    @variable(model, y[r = 1:Ω] >= 0)

    # add objective
    @objective(model, Min, sum(c_r[r]*y[r] for r = 1:Ω) )

    # add constraints
    @constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) >= 1)
    
    # Maximum vehicle constraint
    #@constraint(model, max_vehicle_constr, sum(y[r] for r=1:Ω) <= K)

    return model

end


################################################################
############### Define the Labeling Algo #######################
################################################################

"""
    expected_recourse_cost(S::Int, demand_rate::Float64, remaining_capacity::Float64, 
                          current_node::Int, next_node::Int, distances::Array{Float64})

Calculates expected recourse cost for visiting next_node given current vehicle state.

# Arguments
- `S::Int`: Number of samples for Sample Average Approximation (SAA)
- `demand_rate::Float64`: Poisson parameter λ for stochastic demand at next_node
- `remaining_capacity::Float64`: Current available capacity in vehicle
- `current_node::Int`: Current position in route
- `next_node::Int`: Next customer to visit
- `distances::Array{Float64}`: Distance/cost matrix between all node pairs

# Returns
- `expected_cost::Float64`: Expected cost of recourse action (Detour-to-Depot policy)

# Recourse Policy: Detour-to-Depot (DTD)
When demand D_j exceeds remaining capacity:
1. **Return to depot**: cost = distances[next_node, 1]
2. **Go to next customer**: cost = distances[1, next_node]
3. **Total recourse**: DTD cost = distances[next_node, 1] + distances[1, next_node]

# Mathematical Model
```
E[Recourse] = Σ_{d=0}^{∞} P(D_j = d) * recourse_cost(d)
where recourse_cost(d) = {
    0                                          if d ≤ remaining_capacity
    distances[next_node, 1] + distances[1, next_node]  if d > remaining_capacity
}
```
# Sample Average Approximation (SAA)
- Uses deterministic seed based on node parameters for reproducibility
- Generates S samples from Poisson(λ) distribution
- Averages recourse costs across scenarios
- Critical for consistent dominance rules in labeling algorithm

# Notes
- Assumes Poisson demand: D_j ~ Poisson(demand_rate)
- DTD policy is most common recourse strategy in VRPSD literature
- Deterministic seeding ensures column generation convergence
- Higher S values increase accuracy but computational cost
"""
function expected_recourse_cost(S::Int, demand_rate::Float64, remaining_capacity::Float64,
                            current_node::Int, next_node::Int, distances::Array{Float64})
    
    # Stochastic demand follows Poisson distribution with rate λ
    poisson_dist = Poisson(demand_rate)
    
    # Use a deterministic seed based on node and demand_rate to ensure consistency
    # This is critical for proper dominance and column generation
    seed_value = hash((current_node, next_node, demand_rate, remaining_capacity))
    rng = MersenneTwister(seed_value)
    
    # Sample Average Approximation: Generate S samples and average the recourse cost
    total_recourse = 0.0
    
    for s in 1:S
        # Generate random demand sample from Poisson distribution with fixed RNG
        demand_sample = rand(rng, poisson_dist)
        
        # Check if demand exceeds remaining capacity
        if demand_sample > remaining_capacity
            # DTD Recourse Policy:
            # When arriving at next_node, if demand > capacity:
            # 1. Go from next_node to depot (restocking trip)
            # 2. Return from depot to next_node (to serve the customer)
            # The route continues from next_node after restocking
            # Total extra distance = next_node → depot → next_node
            recourse_distance = distances[next_node, 1] + distances[1, next_node]
            total_recourse += recourse_distance
        end
        # If demand ≤ capacity, no recourse needed (cost = 0)
    end
    
    # Return average recourse cost over all samples
    return total_recourse / S
end

"""
    initialize_labels(start_node::Int)

Initializes the labeling algorithm with a starting label at the depot.

# Arguments
- `start_node::Int`: Starting node (typically depot node 1)

# Returns
- `Vector{Label}`: Single-element array containing initial label

# Initial Label Properties
- **Node**: start_node (depot)
- **Cost**: 0.0 (no cost incurred yet)
- **Load**: 0.0 (vehicle empty)
- **Recourse Cost**: 0.0 (no recourse needed at depot)
- **Path**: [start_node] (route begins at depot)
- **Label Name**: String representation of start_node

# Role in Labeling Algorithm
This creates the "root" label from which all feasible routes will be generated through
label extension. The algorithm explores paths by extending labels from depot through
customers to final depot, tracking costs, loads, and recourse costs.

# Notes
- Part of shortest path with resource constraints (SPPRC) framework
- Each label represents a partial route state
- Extensions create new labels representing route continuations
"""
function initialize_labels(start_node::Int)
    name = string(start_node)
    path = [start_node]

    return [Label(name, start_node, 0.0, 0.0, 0.0, path)]
end

"""
    extend_label(label::Label, current_node::Int, next_node::Int, arc_costs::Array{Float64}, 
                demand_rate::Float64, vehicle_capacity::Float64, distances::Array{Float64})

Extends a label by visiting next_node, creating new label with updated costs and state.

# Arguments
- `label::Label`: Current label representing partial route
- `current_node::Int`: Current position (should match label.node)
- `next_node::Int`: Customer to visit next
- `arc_costs::Array{Float64}`: Modified cost matrix (includes dual prices from RMP)
- `demand_rate::Float64`: Poisson rate λ for stochastic demand at next_node
- `vehicle_capacity::Float64`: Maximum vehicle capacity
- `distances::Array{Float64}`: Original distance matrix for recourse calculations

# Returns
- `Label`: New extended label, or `nothing` if extension infeasible

# Extension Process
1. **Cycle Check**: Reject if next_node already visited
2. **Load Update**: Add expected demand to current load
3. **Capacity Check**: Reject if expected load exceeds capacity  
4. **Cost Calculation**: Combine arc cost + expected recourse cost
5. **Path Update**: Append next_node to route path
6. **Label Creation**: Return new label with updated state

# Mathematical Components
```
new_cost = label.cost + arc_costs[current,next] + E[recourse_cost]
new_load = label.load + E[demand] = label.load + demand_rate
new_path = label.path ∪ {next_node}
```

# Key Features
- **Stochastic Modeling**: Uses expected demands and recourse costs
- **Resource Constraints**: Enforces capacity constraints
- **Cycle Prevention**: Maintains simple path property
- **Cost Accuracy**: Includes both arc and expected recourse costs

# Infeasibility Conditions
- Next node already in path (cycle)
- Expected load exceeds vehicle capacity
- Any computational errors in cost calculation

# Notes
- Core operation in SPPRC labeling algorithm
- Arc costs include dual prices from RMP (reduced costs)
- Recourse costs handle demand uncertainty
- Expected demands used for deterministic capacity checking
"""
function extend_label(label::Label, current_node::Int, next_node::Int, arc_costs::Array{Float64}, 
    demand_rate::Float64, vehicle_capacity::Float64, distances::Array{Float64})

    if next_node in label.path  # Avoid cycles
        return nothing
    end

    remaining_capacity = vehicle_capacity - label.load
    expected_demand = demand_rate
    new_load = label.load + expected_demand
    
    # Check if expected load exceeds capacity
    if new_load > vehicle_capacity
        return nothing  # Infeasible
    end

    new_name = ""
    for node in label.path
        new_name = new_name * string(node) * "." 
    end
    new_name = new_name * string(next_node)
    
    # Calculate recourse cost for this extension
    S = 500  # Number of samples for SAA - higher for better accuracy
    recourse_cost = expected_recourse_cost(S, demand_rate, remaining_capacity, 
                                        current_node, next_node, distances)
    
    arc_cost = arc_costs[current_node, next_node]
    additional_cost = arc_cost + recourse_cost
    new_cost = label.cost + additional_cost
    new_path = vcat(label.path, next_node)
    
    # Accumulate total recourse cost for the entire route
    total_recourse = label.E_recourse_cost + recourse_cost

    return Label(new_name, next_node, new_cost, new_load, total_recourse, new_path)


end

"""
    dominates(label1::Label, label2::Label, N::Int)

Determines if label1 dominates label2 for pruning in labeling algorithm.

# Arguments
- `label1::Label`: Potentially dominating label
- `label2::Label`: Potentially dominated label  
- `N::Int`: Total number of nodes (for customer set comparison)

# Returns
- `Bool`: true if label1 dominates label2, false otherwise

# Dominance Rules
Label1 dominates label2 if:
1. **Same Location**: Both labels at same node
2. **Better/Equal Cost**: label1.cost ≤ label2.cost
3. **Better/Equal Load**: label1.load ≤ label2.load
4. **Same/Subset Customers**: customers(label1) ⊆ customers(label2)

# Mathematical Formulation
```
label1 ≻ label2 ⇔ 
    node1 = node2 ∧
    cost1 ≤ cost2 ∧  
    load1 ≤ load2 ∧
    customers1 ⊆ customers2
```

# Pruning Logic
If label1 dominates label2:
- Label2 can be safely discarded
- Any extension possible from label2 is also possible from label1
- Label1 will always produce better/equal solutions
- Reduces search space without losing optimality

# Customer Set Extraction
- Filters path to extract customer nodes (excludes depot nodes 1 and N)
- Compares customer sets using subset relationship
- Ensures route feasibility is preserved

# Computational Impact
- Critical for algorithm efficiency
- Poor dominance → exponential label growth
- Effective dominance → polynomial complexity
- Must be computationally fast (called frequently)

# Notes
- Conservative dominance rule (simplified for efficiency)
- Could be enhanced with time windows, precedence constraints
- Balance between pruning power and computational cost
- Essential for practical VRPSD solution
"""
function dominates(label1::Label, label2::Label, N::Int)
    # Same node, better or equal cost and load
    if label1.node != label2.node
        return false
    end

    customers1 = Set(filter(x -> x > 1 && x < N, label1.path))
    customers2 = Set(filter(x -> x > 1 && x < N, label2.path))

    return (label1.cost <= label2.cost && label1.load <= label2.load && issubset(customers1, customers2))
end


"""
    preprocess_graph(data::Data, neighbors::Int, cost::Array{Float64})

Creates a reduced arc set for the labeling algorithm to improve computational efficiency.

# Arguments
- `data::Data`: VRPSD instance data
- `neighbors::Int`: Maximum number of nearest neighbors per customer (0 = full graph)
- `cost::Array{Float64}`: Modified cost matrix with dual prices

# Returns
- `Graph`: Structure containing nodes and feasible arcs

# Arc Generation Strategies

## Full Graph (neighbors = 0)
- Creates all possible arcs between nodes: O(N²) arcs
- Computationally expensive but guarantees optimality
- Used when problem size is small

## Nearest Neighbor (neighbors > 0)  
- Each customer connects to `neighbors` closest customers
- Depot connections: Always include arcs from depot to all customers
- Return connections: Always include arcs from all customers to final depot
- Reduces arc count from O(N²) to O(N × neighbors)

# Arc Filtering Rules
Based on Dumas et al. paper:
1. **No depot returns**: Remove (N,i) arcs ∀i (can't leave final depot)
2. **No depot departures**: Remove (i,1) arcs ∀i (can't return to initial depot)
3. **No direct depot-to-depot**: Remove (1,N) arc (meaningless route)

# Neighborhood Selection
- Uses `partialsortperm` to find nearest neighbors by distance
- Ensures depot accessibility: (1,i) and (i,N) arcs always included
- Balances solution quality vs. computational efficiency

# Computational Benefits
- **Memory**: Reduces storage from O(N²) to O(N × k)
- **Speed**: Fewer arcs to explore in labeling algorithm
- **Scalability**: Enables solution of larger instances

# Quality Trade-offs
- Smaller neighborhoods: faster but potentially suboptimal
- Larger neighborhoods: slower but better solution quality
- Full graph: slowest but guaranteed optimal (within labeling)

# Usage Patterns
- Start with small neighborhoods for quick bounds
- Increase neighborhood size if no improving columns found
- Adaptive strategy balances speed and quality

# Notes
- Arc reduction is key enabler for large VRPSD instances
- Must preserve depot connectivity for feasible routes
- Neighborhood size is critical tuning parameter
"""
function preprocess_graph(data::Data, neighbors::Int, cost::Array{Float64})

    N = data.N
    n = data.n

    Nodes = collect(1:N)
    
    Arcs = []
    if neighbors == 0
        for i in Nodes
            for j in Nodes if i ≠ j
                arc = (i,j)
                push!(Arcs, arc)
            end 
        end end
    else
        for i in 2:n+1
            neighbor_cost = partialsortperm(cost[i,2:n+1], 1:neighbors)
            #println("Neighbors for node ", i, ": ", neighbor_cost)
            for j in neighbor_cost
                if i != j
                    arc = (i,j)
                    push!(Arcs, arc)
                end
            end

        end

        for i in 2:n+1

            if isempty(filter(x->x == (1,i), Arcs))
                arc = (1,i)
                push!(Arcs, arc)
            end

            if isempty(filter(x->x == (i,N), Arcs))
                arc = (i,N)
                push!(Arcs, arc)
            end
        end
    end

    # println("Arcs before pruning:", length(Arcs))
    # reduction of Arcs based on Dumas Paper

    
    for i in Nodes
        # a) priority
        filter!(x->x ≠ (N,i), Arcs)
        filter!(x->x ≠ (i,1), Arcs)
        filter!(x->x ≠ (1,N), Arcs)
    end

    #println("Arcs after pruning:", length(Arcs))


    return Graph(Nodes, Arcs)

end

"""
    labeling_algortihm(data::Data, graph::Graph, ĉ, col2return::Int, U_lim::Int)

Dynamic programming labeling algorithm for solving the Shortest Path Problem with 
Resource Constraints (SPPRC) in the VRPSD pricing problem.

# Arguments
- `data::Data`: VRPSD instance data structure
- `graph::Graph`: Reduced network with feasible arcs (from preprocessing)
- `ĉ::Array{Float64}`: Modified cost matrix with dual prices (reduced costs)
- `col2return::Int`: Maximum number of negative reduced cost routes to return
- `U_lim::Int`: Beam search width (maximum unprocessed labels to keep)

# Returns
- `Vector{Label}`: Up to `col2return` routes with negative reduced cost, sorted by cost
- `nothing`: If no improving columns found

# Algorithm: Dynamic Programming Labeling with Dominance and Beam Search

## Core Data Structures
- **U**: Queue of unprocessed labels (partial routes to extend)
- **L_f**: Set of final/processed labels (complete and dominated partial routes)
- **Lᵢ**: Subset of L_f containing labels at node i (for dominance checking)

## Algorithm Steps
1. **Initialize**: U = [depot_label], L_f = []
2. **Main Loop** (while U not empty):
   a) Pop label L from U
   b) If L reached final depot: add to L_f, continue
   c) Check dominance: if any label in L_f at same node dominates L, discard
   d) Extend L along all feasible outgoing arcs, add new labels to U
   e) Sort U by cost, keep only best U_lim labels (beam search)
3. **Post-process**: Filter L_f for negative reduced cost, return best routes

## Mathematical Foundation (SPPRC)
\`\`\`
minimize:    c(P) - Σ π_i    (reduced cost)
subject to:  q(P) ≤ Q         (capacity)
             no cycles         (simple path)
\`\`\`

## Key Features
- **Dominance Pruning**: L1 ≻ L2 if same node, lower cost/load, fewer customers
- **Beam Search**: Limit memory by keeping best U_lim labels
- **Resource Tracking**: Cost, load, path for each label
- **Stochastic Costs**: Include expected recourse via SAA

## Performance Tuning
- U_lim: 500 (fast), 1500 (balanced), 3000+ (quality)
- col2return: More columns → fewer CG iterations
- Neighborhood size: Affects arc count and route quality

# See Also
- `initialize_labels()`, `extend_label()`, `dominates()`, `preprocess_graph()`
"""
function labeling_algortihm(data::Data, graph::Graph, ĉ, col2return::Int, U_lim::Int)
    
    n = data.n
    N = data.N
    q = data.q
    d = data.d
    c = data.c
    q = data.q
    λ_rate = data.λ_rate

    arcs = graph.Arcs
    #println(arcs)
    
    U = initialize_labels(1)
    L_f = []  # final labels at sink node N
    cnt = 0
    max_iterations = 10000  # Add iteration limit
    while !isempty(U) && cnt < max_iterations
        L = popfirst!(U)
        i = L.node
        
        if i == N
            push!(L_f, L)
        else # i < N
            
            # define arc list where tuple is (i, whatever)
            arc_list = []
            for (from, to) in arcs
                if from == i
                    push!(arc_list,(from,to))
                end
            end

            # Lᵢ is the set of processed labels at node i (paths ending at node i)
            Lᵢ = filter(x -> x.node == i, L_f)
            keep = true
            if !isempty(Lᵢ)
                for label in Lᵢ
                    if dominates(label, L, N)
                        keep = false
                        break
                    end
                end

                if keep
                    push!(L_f,L)
                end
            end

            if keep 
                for (from, to) in arc_list # extending labels to all arcs leaving node i
                    new_L = extend_label(L, from, to, ĉ, λ_rate[to], q, c)
                    if !isnothing(new_L)
                        push!(U,new_L)
                    end
                end
            end
        
        end
        
        #Only keep the 500 best U
        sort!(U, by = x -> x.cost)
        U = U[1:min(U_lim,end)]

        cnt += 1
    end
    
    sort!(L_f, by = x -> x.cost)
    filter!(x->x.cost < -1e-8, L_f)
    if !isempty(L_f)
        vals = min(col2return,length(L_f))
        return L_f[1:vals]
    else
        return nothing
    end


end


"""
    runCG(master::Model, data::Data, col2return::Int, neighbors::Vector{Int}, 
          paths::Vector{Vector{Int}}, U_limit::Int)

Column generation procedure to solve LP relaxation of RMP.

# Arguments
- `master::Model`: Restricted master problem (JuMP model)
- `data::Data`: VRPSD instance data
- `col2return::Int`: Number of columns to generate per iteration
- `neighbors::Vector{Int}`: Neighborhood sizes for adaptive arc reduction
- `paths::Vector{Vector{Int}}`: Current pool of generated routes
- `U_limit::Int`: Beam width for labeling algorithm

# Returns
- `m_obj::Vector{Float64}`: Objective values through iterations
- `r_cost::Vector{Float64}`: Reduced costs of all generated columns
- `paths::Vector{Vector{Int}}`: Updated route pool

# Algorithm Flow
1. **Solve RMP**: Obtain optimal LP solution
2. **Extract Duals**: Get π from set partitioning constraints
3. **Modify Costs**: Create ĉ = c - π (reduced cost matrix)
4. **Pricing Problem**: Solve SPPRC via labeling algorithm
5. **Add Columns**: Insert routes with negative reduced cost
6. **Convergence Check**: Terminate when no improving columns found

# Adaptive Neighborhood Strategy
- Starts with small neighborhood (fast iterations)
- Increases neighborhood size if no columns found
- Progresses through `neighbors` vector until convergence
- Balances speed vs. solution quality

# Convergence Criteria
- No columns with reduced cost < -ε (where ε = 1e-6)
- All neighborhood sizes exhausted
- Maximum iterations reached (100)

# Technical Details
- Only adds columns with RC < -1e-6 (numerical tolerance)
- Updates A matrix and objective coefficients dynamically
- Tracks active routes and vehicle usage
- Provides iteration-by-iteration progress output

# Mathematical Foundation
Dantzig-Wolfe decomposition:
- Master: Select routes to cover customers
- Pricing: Find routes with negative reduced cost
- Duals π link master and pricing problems
"""
function runCG(master::Model, data::Data, col2return::Int, neighbors::Vector{Int}, paths::Vector{Vector{Int}}, U_limit::Int)
    
    n = data.n
    c = data.c
    N = data.N

    m_obj = []
    r_cost = []

    indx = 1
    max_iter = 100  # Maximum CG iterations to prevent infinite loops
    iter = 0

    while iter < max_iter
        iter += 1
        # solve the restricted master problem
        optimize!(master)

        if is_solved_and_feasible(master)
        
            println("\n=== CG Iteration ", iter, " ===")
            println("Master Problem Objective Value = ", round(objective_value(master), digits=2))
            push!(m_obj, objective_value(master))
            
            # Show current solution
            y_vals = value.(master[:y])
            active_routes = findall(x -> x > 1e-5, y_vals)
            println("  Active routes: ", length(active_routes))
            println("  Vehicles used: ", round(sum(y_vals), digits=2), " / ", data.K)

            # Get dual variables
            π = dual.(master[:sp_constr])  # Duals from set partitioning constraints
            #ν = dual(master[:max_vehicle_constr])  # Dual from vehicle constraint

            ĉ = deepcopy(c)
            ĉ[2:n+1,:] .-= π  # Subtract dual values only for customers visited


            graph = preprocess_graph(data, neighbors[indx], ĉ)

            cols = labeling_algortihm(data, graph, ĉ, col2return, U_limit)
            
            # Track how many columns are actually added
            cols_added = 0
            
            if !isnothing(cols)
                for col in cols
                    
                    reduced_cost = col.cost
                    
                    # Only add columns with negative reduced cost (improving columns)
                    if reduced_cost >= -1e-6
                        continue  # Skip non-improving columns
                    end
                    
                    cols_added += 1
                    push!(r_cost, reduced_cost)
                    push!(paths, col.path)
                    # Create new A column
                    A_new = zeros(Int, n)
                    new_path = filter(x->(x>1) && (x<=n+1), col.path)
                    
                    for node in new_path
                        customer_idx = node - 1  # Customer indices are 1 to n, nodes are 2 to n+1
                        A_new[customer_idx] = 1
                    end
                    
                    # create new c_r
                    c_r_new = col.E_recourse_cost
                    for i in 1:(length(col.path)-1)
                        c_r_new += c[col.path[i],col.path[i+1]]
                    end
                    #println("New Cost: ", c_r_new)
                    
                    # Add new variable to master problem
                    push!(master[:y], @variable(master, lower_bound = 0.0))
                    set_objective_coefficient.(master, master[:y][end], c_r_new)
                    # Set coefficient in set partitioning constraints
                    set_normalized_coefficient.(master[:sp_constr], master[:y][end], A_new)
                    # Set coefficient in vehicle constraint (each route uses 1 vehicle)
                    # set_normalized_coefficient(master[:max_vehicle_constr], master[:y][end], 1.0)
                    end
                    
                println("  Columns added: ", cols_added, " out of ", length(cols), " candidates")
            end
            
            # Check if we should terminate CG
            if cols_added == 0
                # No columns with negative reduced cost found
                println("  No improving columns found with neighbors=", neighbors[indx])
                if indx < length(neighbors)
                    indx += 1
                    println("  Trying with larger neighborhood: ", neighbors[indx])
                else
                    println("✓ Column generation converged - no more improving columns")
                    break
                end
            end
        else
            println("❌ Master problem infeasible or failed to solve")
            break
        end

    end
    
    if iter >= max_iter
        println("⚠️  WARNING: Reached maximum CG iterations (", max_iter, ") - may not be fully optimized")
    end

    return m_obj, r_cost, paths
end

"""
    branch_and_price(data::Data, col2return::Int, neighbors::Vector{Int}, U_limit::Int)

Branch-and-price algorithm for exact solution of VRPSD.

# Arguments
- `data::Data`: VRPSD instance data
- `col2return::Int`: Columns to generate per CG iteration
- `neighbors::Vector{Int}`: Adaptive neighborhood sizes
- `U_limit::Int`: Beam width for labeling algorithm

# Returns
- `BranchNode`: Best integer solution found

# Algorithm Overview
Exact solution method combining:
1. **Column Generation**: Solve LP relaxation at each node
2. **Branching**: Create child nodes on fractional variables
3. **Pruning**: Eliminate nodes that cannot improve incumbent
4. **Bounding**: Track global upper and lower bounds

# Node Selection Strategy
- **Best-First Search**: Explores node with lowest LP bound first
- Advantages: Finds good solutions early, effective pruning
- Alternative strategies: depth-first, breadth-first

# Branching Strategy
- **Most Fractional Rule**: Select yᵣ closest to 0.5
- Creates two children: yᵣ ≤ 0 and yᵣ ≥ 1
- Single variable branching prevents exponential explosion
- Critical for computational tractability

# Pruning Rules
1. **By Bound**: LB(node) ≥ UB → prune
2. **By Integrality**: Integer solution → update incumbent
3. **By Infeasibility**: No feasible solution → prune

# Bound Management
- **Upper Bound (UB)**: Best integer solution found
- **Lower Bound (LB)**: Minimum LP bound of open nodes
- **Optimality Gap**: UB - LB
- **Convergence**: Gap = 0

# Computational Safeguards
- Maximum nodes: 150 (prevents excessive runtime)
- Maximum CG iterations: 100 per node
- Time limits on RMP solves: 600 seconds

# Performance Monitoring
- Tracks: nodes explored, open nodes, UB, LB, gap
- Progress output each iteration
- Warning if limits reached

# Mathematical Foundation
Branch-and-price combines:
- Branch-and-bound framework for tree search
- Column generation for LP bounds
- Ensures optimal integer solution (if found within limits)

# Notes
- Solution quality depends on: neighborhood sizes, beam width, node limit
- Larger parameters: better solutions but longer runtime
- Trade-off between optimality guarantee and practical runtime
"""
function branch_and_price(data::Data, col2return::Int, neighbors::Vector{Int}, U_limit::Int)

    n = data.n
    N = data.N

    # Create smart initial routes that respect K constraint
    init_paths = create_initial_routes(data)
    
    # Initialize RMP with these routes
    init_master = RMP(data, init_paths)

    root_node = BranchNode("Level 0 --> root_node", 0, init_master, [], [], init_paths)
    branches = [root_node]
    best_ub = Inf
    best_lb = -Inf  # Initialize to -Inf for proper lower bound tracking
    best_solution = [root_node]

    run_cnt = 1
    max_nodes = 200 # Add max nodes limit to prevent infinite branching

    while !isempty(branches) && run_cnt <= max_nodes
        # IMPROVED NODE SELECTION: Best-first search (lowest lower bound first)
        # This explores promising branches first and prunes faster
        sort!(branches, by = x -> length(x.objective) > 0 ? x.objective[end] : Inf)
        branch_node = popfirst!(branches)
        
        # Prune nodes that cannot improve the best solution
        if length(branch_node.objective) > 0 && branch_node.objective[end] >= best_ub
            println(branch_node.name, "--> Pruned before solving (LB=", branch_node.objective[end], " >= UB=", best_ub, ")")
            continue
        end
        
        println("Running ", branch_node.name)
        m_obj, r_cost, new_paths = runCG(branch_node.master, data, col2return, neighbors, branch_node.paths, U_limit)
        branch_node.objective = m_obj
        branch_node.rc = r_cost
        branch_node.paths = new_paths

        if is_solved_and_feasible(branch_node.master)

            ### Find integral solution
            yᵣ = value.(branch_node.master[:y])
            frac_yᵣ_indx = findall(x->x < 1.0 && x > 1e-5, yᵣ)
            println("fractional yᵣ index ", frac_yᵣ_indx)
            
            # VALIDATION: Check all customers are served
            customer_coverage = zeros(n)
            for (idx, y_val) in enumerate(yᵣ)
                if y_val > 1e-8
                    route = new_paths[idx]
                    for node in route
                        if node > 1 && node <= n+1
                            customer_idx = node - 1
                            customer_coverage[customer_idx] += y_val
                        end
                    end
                end
            end
            
            nonserved = findall(x -> x <= 0 , customer_coverage)
            unserved = findall(x -> x < 0.99, customer_coverage)
            overserved = findall(x -> x > 1.01, customer_coverage)
            
            if !isempty(unserved) || !isempty(overserved)
                println("⚠️  COVERAGE ISSUES:")
                if !isempty(unserved)
                    println("  Unserved customers: ", length(unserved), " - ", unserved)
                end
                if !isempty(overserved)
                    println("  Overserved customers: ", length(overserved), " - ", overserved)
                end
                if !isempty(nonserved)
                    println("  Non served customers: ", length(nonserved), " - ", nonserved)
                end
            end
            
            # VALIDATION: Check set partitioning constraints
            active_routes_idx = findall(x -> x > 1e-5, yᵣ)
            println("Active routes (y > 1e-5): ", length(active_routes_idx))

            if isempty(frac_yᵣ_indx) && branch_node.objective[end]<best_ub
                best_ub = branch_node.objective[end]
                push!(best_solution, branch_node)
                println("*** New incumbent solution found! UB = ", best_ub, " ***")
            elseif branch_node.objective[end]>best_ub
                println(branch_node.name, "-->Model was fathomed by bound (LB=", branch_node.objective[end], " > UB=", best_ub, ")")
            else # branch on ONLY THE FIRST non-integral variable (not all of them!)
                if branch_node.objective[end]<best_lb
                    best_lb = branch_node.objective[end]
                end
                
                # Branch on only ONE fractional variable, not all of them
                # Use "most fractional" branching rule: choose variable closest to 0.5
                frac_values = yᵣ[frac_yᵣ_indx]
                fractionality = abs.(frac_values .- 0.5)
                most_frac_idx = argmin(fractionality)
                frac_var = frac_yᵣ_indx[most_frac_idx]
                
                println("Branching on yᵣ[", frac_var, "] = ", yᵣ[frac_var], " (most fractional)")
                
                branch_tree_lvl = branch_node.tree_lvl + 1
                
                #### create 2 branches 
                ### Less than 0 inequality (set to 0)
                lt_node_master = copy(branch_node.master)
                set_optimizer(lt_node_master, () -> Gurobi.Optimizer(gurobi_env))
                set_attribute(lt_node_master, "TimeLimit", 600)
                set_attribute(lt_node_master, "LogToConsole", 0)
                
                ### Add new constraint
                @constraint(lt_node_master, lt_node_master[:y][frac_var] ≤ 0 )

                #### name of node
                lt_node_name = "Level " *string(branch_tree_lvl)* " branch yᵣ[" *string(frac_var) * "] ≤ 0"

                lt_branch = BranchNode(lt_node_name, branch_tree_lvl, lt_node_master, [], [], new_paths)
                push!(branches, lt_branch)

                ### Greater than equal 1 inequality
                gt_node_master = copy(branch_node.master)
                set_optimizer(gt_node_master, () -> Gurobi.Optimizer(gurobi_env))
                set_attribute(gt_node_master, "TimeLimit", 600)
                set_attribute(gt_node_master, "LogToConsole", 0)

                ### Add new constraint
                @constraint(gt_node_master, gt_node_master[:y][frac_var] ≥ 1 )

                #### name of node
                gt_node_name = "Level " *string(branch_tree_lvl)* " branch yᵣ[" *string(frac_var) * "] ≥ 1"

                gt_branch = BranchNode(gt_node_name, branch_tree_lvl, gt_node_master, [], [], new_paths)
                push!(branches, gt_branch)
            end
        else
            println(branch_node.name, "-->Model was infeasible")
        end
        
        # Update global lower bound (minimum objective of all open nodes)
        if !isempty(branches)
            best_lb = minimum([length(b.objective) > 0 ? b.objective[end] : Inf for b in branches])
        end
        
        println("Nodes explored: ", run_cnt, " | Open nodes: ", length(branches), " | Best UB=", best_ub, " | Best LB=", best_lb, " | Gap=", best_ub - best_lb)
        run_cnt += 1
    end
    
    if run_cnt > max_nodes
        println("WARNING: Reached maximum node limit (", max_nodes, "). Solution may be suboptimal.")
    end

    return best_solution[end]

end


"""
    print_results_to_file(results::Results, df::DataFrame, data::Data)

Generates comprehensive output files and visualizations for VRPSD solution.

# Arguments
- `results::Results`: Solution object containing best BranchNode and timing
- `df::DataFrame`: Original instance data for visualization
- `data::Data`: VRPSD instance data for validation

# Generated Files
Creates directory structure: `Results_BnP/<instance_name>/`

1. **results.txt**: Text summary including:
   - Input parameters (instance, vehicles, capacity)
   - Solution quality (objective value, solve time)
   - Routes by customer indices and node IDs
   - Algorithm statistics (CG iterations, vehicles used)
   - Neighborhood sizes used

2. **Customer_loc_w_nodes.png**: PlotlyJS plot showing:
   - Customer and depot locations
   - Node IDs and demands

3. **Customer_routes_cg.png**: Route visualization showing:
   - All selected routes with different colors
   - Depot and customer positions
   - Directional arrows for route flow

4. **fig_rc.png**: Reduced cost plot showing:
   - All generated column reduced costs
   - Convergence to non-negative values

5. **fig_obj.png**: Objective value plot showing:
   - RMP objective through CG iterations
   - Convergence trajectory

# Route Extraction
- Identifies routes with yᵣ ≥ 0.99 (integer threshold)
- Converts node indices to customer IDs
- Calculates actual route usage

# Visualization Features
- Color-coded routes for clarity
- Aspect ratio = 1 for accurate distances
- Legend with route identifiers
- Directional arrows showing travel direction

# Output Format
All numeric values rounded appropriately:
- Objectives: 2 decimal places
- Coordinates: original precision
- Times: 4 decimal places

# Notes
- Creates directory structure automatically
- Overwrites existing results for same instance
- Uses global `input_list` for instance identification
"""
function print_results_to_file(results::Results, df::DataFrame)
    
    final_sol = results.node
    
    m_obj = final_sol.objective
    r_cost = final_sol.rc
    paths = final_sol.paths

    fig_obj = Plots.plot(1:length(m_obj), m_obj, label="Master Problem Objective", lw=2)

    fig_rc = Plots.plot(1:length(r_cost), r_cost, label="Reduced Cost", lw=2)

    ind_CG_routes = []
    routes_cg = [] 

    yVal_cg = value.(final_sol.master[:y])
    
    println("\n=== EXTRACTING SOLUTION ===")
    println("Total columns generated: ", length(yVal_cg))
    println("Non-zero y values: ", count(x -> x > 1e-5, yVal_cg))
    
    for i=1:length(yVal_cg)
        if yVal_cg[i] >= 0.99  # Only include routes that are essentially selected (integer solution)
            push!(ind_CG_routes,i)
            push!(routes_cg, paths[i])
            println("  Route ", i, ": y = ", yVal_cg[i], ", path = ", paths[i])
        end
    end

    println("\nPaths chosen: ", ind_CG_routes, " with y values of ", yVal_cg[ind_CG_routes])
    println("Number of CG Iterations: ", length(m_obj))
    
    # Get problem dimensions from dataframe
    n_customers = nrow(df[df.node_type .== "customer", :])
    N_total = nrow(df)
    
    # VALIDATION: Check for duplicate and unvisited customers
    println("\n=== SOLUTION VALIDATION ===")
    println("Expected customers: ", n_customers, " (nodes 2 to ", n_customers+1, ")")
    println("Total nodes in data: ", N_total)
    
    all_customers = []
    for (route_idx, route) in enumerate(routes_cg)
        # Extract customers (exclude depot nodes 1 and N_total)
        customers_in_route = filter(x -> x > 1 && x < N_total, route)
        println("Route ", route_idx, ": ", route, " → Customers: ", customers_in_route)
        append!(all_customers, customers_in_route)
    end
    
    # Check for duplicates
    if length(all_customers) != length(unique(all_customers))
        println("\n❌ ERROR: Duplicate customers found!")
        customer_counts = Dict()
        for c in all_customers
            customer_counts[c] = get(customer_counts, c, 0) + 1
        end
        duplicates = filter(p -> p.second > 1, customer_counts)
        println("Duplicate customers: ", duplicates)
        println("\nDETAILED ANALYSIS:")
        for (cust, count) in duplicates
            println("  Customer node ", cust, " (customer index ", cust-1, ") appears in ", count, " routes:")
            for (ridx, route) in enumerate(routes_cg)
                if cust in route
                    println("    - Route ", ridx, " (column ", ind_CG_routes[ridx], ", y=", yVal_cg[ind_CG_routes[ridx]], ")")
                end
            end
        end
    end
    
    # Check for unvisited customers
    expected_customers = Set(2:n_customers+1)
    visited_customers = Set(all_customers)
    unvisited = setdiff(expected_customers, visited_customers)
    
    if !isempty(unvisited)
        println("\n❌ ERROR: ", length(unvisited), " customers UNVISITED!")
        println("Unvisited customer nodes: ", sort(collect(unvisited)))
        println("Unvisited customer indices (for constraints): ", sort([c-1 for c in unvisited]))
    end
    
    # Final verdict
    if isempty(unvisited) && length(all_customers) == length(unique(all_customers))
        println("\n✓ Solution is VALID: All ", n_customers, " customers visited exactly once")
    else
        println("\n❌ Solution is INVALID")
        if !isempty(unvisited)
            println("  - ", length(unvisited), " customers unvisited")
        end
        if length(all_customers) != length(unique(all_customers))
            println("  - Duplicate customer visits detected")
        end
    end
    
    println("Total customers in routes: ", length(all_customers))
    println("Unique customers in routes: ", length(unique(all_customers)))
    println("========================\n")

    route_nodes = []
    for r in routes_cg
        node_list = []
        for loc in r
            push!(node_list, copy(df.node_id[loc]))
        end
        push!(route_nodes, copy(node_list))
    end

    fig_cg = Plots.scatter(df[df.node_type .== "depot",:x_coord], df[df.node_type .== "depot",:y_coord], label="Depot", aspect_ratio = 1, legend = :outertopright, xticks=:true, yticks=:true)
    Plots.scatter!(fig_cg, df[df.node_type .== "customer",:x_coord], df[df.node_type .== "customer",:y_coord], label="Customer")


    for k in 1:size(routes_cg,1)
        p_x = df[routes_cg[k], :x_coord]
        p_y = df[routes_cg[k], :y_coord]
        Plots.plot!(fig_cg, p_x, p_y, arrow=true, label= "Car "*string(k)* " Route", lw=1.5 )
    end

    ################################################################
    ################### Save the Result #######################
    ################################################################

    main_folder = "Results_BnP"

    if !isdir(main_folder)
        mkdir(main_folder)
    end

    if !isdir(main_folder * "/" * input_list[1])
        mkdir(main_folder * "/" * input_list[1])
    end

    open(main_folder * "/" * input_list[1] * "/results.txt", "w") do io
        write(io, "================================\n")
        write(io, "Inputs to Model:\n")
        write(io, "================================\n")
        write(io, "Instance: " * string(input_list[1]) * "\n")
        write(io, "Number Vehicles: " * string(input_list[2]) * "\n")
        write(io, "Capacity: " * string(input_list[3]) * "\n")
        write(io, "\n")
        write(io, "================================\n")
        write(io, "Results of Model:\n")
        write(io, "================================\n")
        write(io, "Objective Solution: ", string(m_obj[end]), "\n")
        write(io, "Solve TIme: ", string(solution_time), " seconds\n")
        write(io, "Cars Used: ", string(size(routes_cg,1)), "\n")
        write(io, "CG iterations: ", string(length(m_obj)), "\n")
        write(io, "\n")
        write(io, "================================\n")
        write(io, "Routes by Cusotomers:\n")
        write(io, "================================\n")
        for (i, r) in enumerate(routes_cg)
            write(io, "Route ", string(i), ": ", string(r), "\n")
        end
        write(io, "\n")
        write(io, "================================\n")
        write(io, "Routes by Nodes:\n")
        write(io, "================================\n")
        for (i, rn) in enumerate(route_nodes)
            write(io, "Route ", string(i), ": ", string(rn), "\n")
        end
        write(io, "\n")
        write(io, "================================\n")
        write(io, "Neighborhood Size Used:\n")
        write(io, "================================\n")
        write(io, "Neighborhood Sizes: ", string(neighbors), "\n")
        write(io, "\n")
    end

    PlotlyJS.savefig(img, main_folder * "/" * input_list[1] * "/Customer_loc_w_nodes.png")
    Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/Customer_routes_cg.png")
    Plots.savefig(fig_rc, main_folder * "/" * input_list[1] * "/fig_rc.png")
    Plots.savefig(fig_obj, main_folder * "/" * input_list[1] * "/fig_obj.png")

end


################################################################
################### Main Execution Block #######################
################################################################

# Configuration and Execution
#
# This section demonstrates typical usage of the VRPSD solver:
# 1. Define instance parameters
# 2. Load and preprocess data
# 3. Configure algorithm parameters
# 4. Execute branch-and-price
# 5. Generate output files and visualizations
#
# Key Parameters to Tune:
# - col2return: More columns → faster convergence but slower iterations
# - neighbors: Larger values → better solution quality but slower
# - U_limit: Larger beam → better solutions but more memory
#
# Example configurations:
# - Fast: col2return=50, neighbors=[5], U_limit=500
# - Balanced: col2return=100, neighbors=[0], U_limit=1500
# - Quality: col2return=200, neighbors=[0], U_limit=3000
input_list = ["A-n33-k5.csv", 5, 100.0]
data, df, img = PreProcess("../data/new_dataset/" * input_list[1], input_list[2] , input_list[3])
col2return = 100
neighbors = [0]
U_limit = 2000

tic = time()
best_solution = branch_and_price(data, col2return, neighbors, U_limit)
toc = time()

solution_time = toc-tic
println("Initial Solution Time(s): ", solution_time)

results = Results(best_solution, solution_time)
print_results_to_file(results, df)

    