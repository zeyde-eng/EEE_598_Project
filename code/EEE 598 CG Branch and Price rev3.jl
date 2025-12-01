#=
================================================================================
Vehicle Routing Problem with Stochastic Demands (VRPSD)
Branch-and-Price Solution Algorithm - Revision 3
================================================================================

Author: Zach Eyde
Student ID: 1223877512
Course: EEE 598 Final Project

Key References:
- Christiansen & Lysgaard (2007) - A branch-and-price algorithm for the VRPSD

Revision 3 Changes:
- Added fleet size constraint: Σy_r ≤ K (maximum K vehicles)
- Updated initial routes to respect K constraint (multi-customer routes)
- Modified column generation to incorporate fleet constraint dual value
- Set partitioning formulation with vehicle limit

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

struct Label
    current_node::Int         # Current node
    load::Int                 # Total load on vehicle so far
    visited::Vector{Bool}     # Which customers have been visited
    path::Vector{Int}         # Path of visited nodes
    cost::Float64             # Cumulative distance traveled + Recourse
    total_recourse_cost::Float64  # Total recourse cost
end


"""
    BranchNode

Represents a node in the branch-and-price tree.

# Fields
- `name::String`: Descriptive name (e.g., "Level 1 branch customers (i,j) same vehicle")
- `tree_lvl::Int`: Depth in branch tree (root = 0)
- `master::Model`: RMP model with branching constraints
- `objective::Vector{Float64}`: Objective values through CG iterations
- `rc::Vector{Float64}`: Reduced costs of generated columns
- `paths::Vector{Vector{Int}}`: Pool of all generated routes
- `must_together::Vector{Tuple{Int,Int}}`: Customer pairs that must be on same route
- `must_separate::Vector{Tuple{Int,Int}}`: Customer pairs that must be on different routes

# Branching Strategy (Set-Partitioning on Customer Pairs)
Branches on pairs of customers based on same-vehicle relationship.
For customers i,j with fractional "togetherness", creates two branches:
- Left child: Customers i,j must be on SAME route (together in at least one selected route)
- Right child: Customers i,j must be on DIFFERENT routes (never together in any selected route)

Note: With set covering, customers may appear in multiple routes. The branching constraint
applies to the selected routes (those with y=1 in integer solution).
"""
mutable struct BranchNode
    name::String
    tree_lvl::Int
    master::Model
    objective::Vector{Float64}
    rc::Vector{Float64}
    paths::Vector{Vector{Int}}
    must_together::Vector{Tuple{Int,Int}}
    must_separate::Vector{Tuple{Int,Int}}
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
    best_ub::Float64
    best_lb::Float64
    gap::Float64
    nodes_explored::Int
end

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

"""
    create_initial_routes(data::Data)

Generates initial feasible routes for the Restricted Master Problem.

# Arguments
- `data::Data`: VRPSD instance data structure

# Returns
- `routes::Vector{Vector{Int}}`: Collection of initial routes, each as node sequence [1, customers..., N]

"""
function create_initial_routes(data::Data)::Vector{Vector{Int}}
    n = data.n
    N = data.N
    c = data.c
    q = data.q
    K = data.K
    λ_rate = data.λ_rate

    println("\n=== Creating Initial Routes (Fleet Constraint K=", K, ") ===")
    println("Customers: ", n, " | Max Vehicles: ", K, " | Capacity: ", q)
    
    # Build routes respecting K constraint using nearest neighbor heuristic
    routes = Vector{Vector{Int}}()
    unvisited = Set(2:(n+1))  # All customer nodes
    
    vehicle_count = 0
    
    while !isempty(unvisited) && vehicle_count < K
        vehicle_count += 1
        route = [1]  # Start at depot
        current_node = 1
        current_load = 0.0
        
        # Build route using nearest neighbor with capacity constraint
        while !isempty(unvisited)
            # Find nearest unvisited customer that fits in vehicle
            best_customer = nothing
            best_distance = Inf
            
            for customer in unvisited
                customer_demand = λ_rate[customer]  # Expected demand
                
                # Check if customer fits in remaining capacity
                if current_load + customer_demand <= q
                    dist = c[current_node, customer]
                    if dist < best_distance
                        best_distance = dist
                        best_customer = customer
                    end
                end
            end
            
            if best_customer === nothing
                # No more customers fit in this vehicle
                break
            end
            
            # Add customer to route
            push!(route, best_customer)
            delete!(unvisited, best_customer)
            current_load += λ_rate[best_customer]
            current_node = best_customer
        end
        
        # Return to depot
        push!(route, N)
        push!(routes, route)
        
        println("Route ", vehicle_count, ": ", route, " (load=", round(current_load, digits=2), "/", q, ")")
    end
    
    # If still have unvisited customers and no more vehicles, force them into routes
    if !isempty(unvisited)
        println("\n⚠ WARNING: ", length(unvisited), " customers remaining but K=", K, " limit reached!")
        println("Adding individual routes for remaining customers (will need column generation to fix)")
        for customer in unvisited
            route = [1, customer, N]
            push!(routes, route)
            println("Extra route: ", route)
        end
    end
    
    # Validate coverage
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
        println("✓ Initial routes cover all ", n, " customers using ", length(routes), " routes")
    end
    
    return routes

end

"""
    RMP(data::Data, initial_routes::Vector{Vector{Int}})

Formulates the Restricted Master Problem for VRPSD using Dantzig-Wolfe decomposition.
Includes fleet size constraint limiting number of vehicles to K.

# Arguments
- `data::Data`: VRPSD instance data
- `initial_routes::Vector{Vector{Int}}`: Initial feasible routes for warm start

# Returns
- `model::JuMP.Model`: Configured RMP with variables, objective, and constraints

# Mathematical Formulation
```
min Σ c_r * y_r                           (minimize total route costs)
s.t. Σ a_{i,r} * y_r >= 1    ∀i ∈ customers  (set covering: each customer visited at least once)
     Σ y_r ≤ K                           (fleet size: use at most K vehicles)
     y_r ≥ 0                  ∀r ∈ routes     (route selection variables)
```

# Key Components
1. **Decision Variables**: y_r ∈ [0,1] indicates if route r is used
2. **Objective**: Minimize sum of route costs (including expected recourse costs)
3. **A Matrix**: a_{i,r} = 1 if customer i is in route r, 0 otherwise
4. **Set COVERING**: Each customer must be visited at least once
5. **Fleet Constraint**: Total routes used cannot exceed K vehicles
6. **Route Costs**: Include travel costs + expected stochastic recourse costs

"""
function RMP(data::Data, initial_routes::Vector{Vector{Int}})

    n = data.n
    N = data.N
    c = data.c
    K = data.K


    Ω = length(initial_routes)
    c_r = []
    
    # Calculate the cost of each path from actual routes (including recourse costs)
    for route in initial_routes
        route_cost = 0.0
        recourse_cost = 0.0
        current_load = 0
        
        for i in 1:(length(route)-1)
            # Add arc travel cost
            route_cost += c[route[i], route[i+1]]
            
            # Add expected recourse cost for visiting customers
            current_node = route[i]
            next_node = route[i+1]
            
            # Only calculate recourse for customer nodes (not depot)
            if next_node > 1 && next_node <= n+1
                customer_demand = data.λ_rate[next_node]
                remaining_capacity = data.q - current_load
                recourse = expected_recourse_cost(500, customer_demand, remaining_capacity, current_node, next_node, c)
                recourse_cost += recourse
                current_load += customer_demand
            end
        end
        
        total_cost = route_cost + recourse_cost
        push!(c_r, total_cost)
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
    set_attribute(model, "Threads", 32)
    set_attribute(model, "Method", -1)

    # add decision variable
    @variable(model, y[r = 1:Ω] >= 0)

    # add objective
    @objective(model, Min, sum(c_r[r]*y[r] for r = 1:Ω) )

    # add constraints - SET COVERING: each customer visited at least once
    @constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) >= 1)
    
    # Fleet size constraint: use at most K vehicles
    @constraint(model, fleet_constr, sum(y[r] for r=1:Ω) <= K)

    return model

end

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
1. **Return to depot from next customer**: cost = distances[next_node, 1]
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

"""
function expected_recourse_cost(S::Int, demand_rate::Float64, remaining_capacity::Float64, current_node::Int, next_node::Int, distances::Array{Float64})
    
    # Stochastic demand follows Poisson distribution with rate λ
    poisson_dist = Poisson(demand_rate)
    
    # Sample Average Approximation: Generate S samples and average the recourse cost
    total_recourse = 0.0
    Random.seed!((current_node + next_node) * 12345)  # Deterministic per node
    demand_sample = rand(poisson_dist, S)
    
    for s in 1:S
        
        # Check if demand exceeds remaining capacity
        if demand_sample[s] > remaining_capacity
            recourse_distance = distances[next_node, 1] + distances[1, next_node]
            total_recourse += recourse_distance
        end
        # If demand ≤ capacity, no recourse needed (cost = 0)
    end
    
    # Return average recourse cost over all samples
    return total_recourse / S
end


################################################################
############### Create Feasible Routes #########################
############### Forward labeling Algo  #########################
################################################################

"""
    dominates(label1::Label, label2::Label)

Checks if label1 dominates label2 according to Christiansen & Lysgaard (2007).

Label1 dominates label2 if:
1. Same current node
2. label1.cost ≤ label2.cost (better or equal cost)
3. label1.load ≤ label2.load (better or equal load)
4. label1.visited ⊇ label2.visited (visited at least the same customers)

If label1 dominates label2, then label2 can be pruned as it cannot lead to better solutions.
"""
function dominates(label1::Label, label2::Label)
    # Must be at same node
    if label1.current_node != label2.current_node
        return false
    end
    
    # label1 must have better or equal cost
    if label1.cost > label2.cost
        return false
    end
    
    # label1 must have better or equal load
    if label1.load > label2.load
        return false
    end
    
    # label1 must have visited at least the same customers as label2
    # (if label2 visited a customer that label1 didn't, then label1 doesn't dominate)
    for i in 1:length(label1.visited)
        if label2.visited[i] && !label1.visited[i]
            return false
        end
    end
    
    return true
end

"""
    forward_labeling(cost_matrix, data::Data, U_limit::Int=1500, must_together=Tuple{Int,Int}[], must_separate=Tuple{Int,Int}[])

Dynamic programming labeling algorithm for solving SPPRC in pricing problem.
Implements dominance checking and beam search pruning as per Christiansen & Lysgaard (2007).

# Arguments
- `cost_matrix`: Modified cost matrix with dual values subtracted
- `data::Data`: VRPSD instance data
- `U_limit::Int`: Beam width for pruning (default 1500)
- `must_together`: Customer pairs that must be on same route (branching constraints)
- `must_separate`: Customer pairs that must be on different routes (branching constraints)

# Returns
- `Vector{Label}`: Feasible routes (labels at final depot)
"""
function forward_labeling(cost_matrix, data::Data, U_limit::Int=1500, must_together::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[], must_separate::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[])
    n = data.n
    N = data.N
    d = data.d
    capacity = data.q
    demand_rate = data.λ_rate
    
    # All customer indices (not including depot, which is 1)
    cust_indices = 2:(n+1)
    
    # U: unprocessed labels (queue)
    # L_f: processed labels at each node (for dominance checking)
    U = Label[]
    L_f = Dict{Int, Vector{Label}}()  # node_id => labels at that node
    feasible_solutions = Label[]
    
    # Initialize at depot
    start_label = Label(1, 0, falses(n), [1], 0.0, 0.0)
    push!(U, start_label)
    
    iteration = 0
    max_iterations = 500000

    while !isempty(U) && iteration < max_iterations
        iteration += 1
        
        # Pop first label from queue (FIFO)
        label = popfirst!(U)
        
        # If at final depot, store as feasible solution
        if label.current_node == N
            push!(feasible_solutions, label)
            continue
        end
        
        # Check dominance against processed labels at this node
        if haskey(L_f, label.current_node)
            is_dominated = false
            for existing_label in L_f[label.current_node]
                if dominates(existing_label, label)
                    is_dominated = true
                    break
                end
            end
            
            # If dominated, discard this label
            if is_dominated
                continue
            end
        end
        
        # Add label to processed set
        if !haskey(L_f, label.current_node)
            L_f[label.current_node] = Label[]
        end
        push!(L_f[label.current_node], label)
        
        # Try extending to unvisited customers
        extended_any = false
        for cust_idx in cust_indices
            if !label.visited[cust_idx-1]
                expected_demand = d[cust_idx]
                
                # Check branching constraints before extending
                # Check must_together: if any customer in path requires togetherness with cust_idx
                violates_constraints = false
                for (ci, cj) in must_together
                    # If ci is in path but cj is not being added (and vice versa), we must skip
                    # We need to ensure if one is visited, the other will be too
                    has_ci = label.visited[ci-1]
                    has_cj = label.visited[cj-1]
                    will_visit_ci = (cust_idx == ci)
                    will_visit_cj = (cust_idx == cj)
                    
                    # If we're about to visit ci but cj is not in path and not being visited
                    if will_visit_ci && !has_cj && !will_visit_cj
                        # We can still visit ci, but we MUST visit cj later
                        # This is allowed - constraint checked at route completion
                    end
                    # If ci is already visited but we're trying to close without visiting cj
                    if has_ci && !has_cj && !will_visit_cj && cust_idx != cj
                        # Can't extend to other customers if ci visited but cj not yet visited
                        # (would need to visit cj first)
                    end
                end
                
                # Check must_separate: if cust_idx must be separated from any visited customer
                for (ci, cj) in must_separate
                    cust_pair = Set([ci, cj])
                    if cust_idx in cust_pair
                        # Find the other customer in the pair
                        other = (cust_idx == ci) ? cj : ci
                        # If other customer already visited, cannot add cust_idx
                        if label.visited[other-1]
                            violates_constraints = true
                            break
                        end
                    end
                end
                
                if violates_constraints
                    continue
                end
                
                # Check capacity constraint
                if label.load + expected_demand <= capacity
                    # Create extended label
                    next_visited = copy(label.visited)
                    next_visited[cust_idx-1] = true
                    next_path = [label.path; cust_idx]
                    
                    # Calculate recourse cost for this arc
                    recourse_cost = expected_recourse_cost(1000, demand_rate[cust_idx], 
                                                        capacity - label.load, label.current_node, cust_idx, cost_matrix)
                    
                    # Update costs and load
                    arc_cost = cost_matrix[label.current_node, cust_idx]
                    next_cost = label.cost + arc_cost + recourse_cost
                    next_recourse = label.total_recourse_cost + recourse_cost
                    next_load = label.load + expected_demand
                    
                    new_label = Label(cust_idx, next_load, next_visited, next_path, 
                                    next_cost, next_recourse)
                    push!(U, new_label)
                    extended_any = true
                end
            end
        end
        
        # If no extensions possible or all customers visited, close route to depot
        if !extended_any || all(label.visited)
            if label.current_node != N  # Don't add if already at final depot
                # Check must_together constraints before closing route
                valid_route = true
                for (ci, cj) in must_together
                    has_ci = label.visited[ci-1]
                    has_cj = label.visited[cj-1]
                    # Both must be in route or both must be out
                    if (has_ci && !has_cj) || (!has_ci && has_cj)
                        valid_route = false
                        break
                    end
                end
                
                if valid_route
                    final_cost = label.cost + cost_matrix[label.current_node, N]
                    final_label = Label(N, label.load, label.visited, [label.path; N], 
                                    final_cost, label.total_recourse_cost)
                    push!(U, final_label)
                end
            end
        end
        
        # Beam search pruning: keep only best U_limit labels in queue
        if length(U) > U_limit
            sort!(U, by = l -> l.cost)
            U = U[1:U_limit]
        end
    end
    
    if iteration >= max_iterations
        println("WARNING: Forward labeling reached max iterations ($(max_iterations))")
    end
    
    return feasible_solutions
end


################################################################
################### CG Algorithm ###############################
################################################################

function runCG(master::Model, data::Data, paths::Vector{Vector{Int}}, U_limit::Int, 
            must_together::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[], 
            must_separate::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
            max_cols_per_iter::Int=300)  # Maximum columns to add per iteration
    
    n = data.n
    c = data.c
    N = data.N

    m_obj = []
    r_cost = []

    max_iter = 500 # Maximum CG iterations to prevent infinite loops
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
            println("Active routes: ", length(active_routes))
            println("Vehicles used: ", round(sum(y_vals), digits=2), " / ", data.K)

            # Get dual variables
            π = dual.(master[:sp_constr])  # Duals from set partitioning constraints
            μ = dual(master[:fleet_constr])  # Dual from fleet size constraint

            ĉ = deepcopy(c)
            ĉ[2:n+1,:] .-= π  # Subtract dual values only for customers visited
            
            # Track how many columns are actually added
            cols_added = 0

            # Forward labeling algorithm to generate columns
            cols = forward_labeling(ĉ, data, U_limit, must_together, must_separate)
            
            if !isnothing(cols) && !isempty(cols)
                # Sort columns by reduced cost (best first) and limit to max_cols_per_iter
                sort!(cols, by = x -> x.cost)
                
                # Filter only columns with negative reduced cost
                # Adjusted reduced cost = route_cost - Σπᵢ - μ (fleet constraint dual)
                improving_cols = filter(x -> (x.cost - μ) < -1e-6, cols)
                cols_to_add = improving_cols[1:min(max_cols_per_iter, length(improving_cols))]
                
                println("  Labeling found ", length(cols), " columns, ", length(improving_cols), 
                        " with negative reduced cost (μ=", round(μ, digits=4), "), adding best ", length(cols_to_add))
                
                for col in cols_to_add
                    cols_added += 1
                    
                    # Calculate actual route cost with original costs (for master objective)
                    c_r_new = col.total_recourse_cost
                    for i in 1:(length(col.path)-1)
                        c_r_new += c[col.path[i],col.path[i+1]]
                    end
                    
                    # col.cost is the reduced cost from labeling algorithm
                    # Store adjusted reduced cost including fleet constraint
                    push!(r_cost, col.cost - μ)
                    push!(paths, col.path)
                    
                    # Create new A column
                    A_new = zeros(Int, n)
                    new_path = filter(x->(x>1) && (x<=n+1), col.path)
                    
                    for node in new_path
                        customer_idx = node - 1  # Customer indices are 1 to n, nodes are 2 to n+1
                        A_new[customer_idx] = 1
                    end
                    
                    # Add new variable to master problem
                    push!(master[:y], @variable(master, lower_bound = 0.0))
                    set_objective_coefficient.(master, master[:y][end], c_r_new)
                    # Set coefficient in set partitioning constraints
                    set_normalized_coefficient.(master[:sp_constr], master[:y][end], A_new)
                    # Set coefficient in fleet constraint (each route uses 1 vehicle)
                    set_normalized_coefficient(master[:fleet_constr], master[:y][end], 1.0)
                end
                    
                #println("Columns added: ", cols_added, " out of ", length(cols), " candidates")
                
                # If no improving columns found, terminate
                if cols_added == 0
                    println("No more improving columns found. CG converged.")
                    break
                end
            else
                println("No columns generated. CG converged.")
                break
            end

        else
            println("Master problem infeasible or failed to solve")
            break
        end

    end
    
    if iter >= max_iter
        println("WARNING: Reached maximum CG iterations (", max_iter, ") - may not be fully optimized")
    end

    return m_obj, r_cost, paths
end



################################################################
################### Integer Master Problem ####################
################################################################

"""
    solve_integer_master(master::Model, data::Data)

Solves the integer version of the master problem by setting y variables to binary.
Used to find integer solutions when LP relaxation is fractional.

# Returns
- `objective_value`: Objective value of integer solution (or Inf if infeasible)
- `is_feasible`: Whether integer solution was found
"""
function solve_integer_master(master::Model, data::Data)
    # Create a copy to avoid modifying the LP master
    int_master = copy(master)
    
    # Set all y variables to binary
    y_vars = int_master[:y]
    for i in 1:length(y_vars)
        set_binary(y_vars[i])
    end
    
    # Solve the integer problem
    set_optimizer(int_master, () -> Gurobi.Optimizer(gurobi_env))
    set_attribute(int_master, "TimeLimit", 720)  # 12 minutes
    set_attribute(int_master, "LogToConsole", 0)  # Enable logging to see progress
    set_attribute(int_master, "MIPGap", 0.01)  # 1% optimality gap
    set_attribute(int_master, "MIPFocus", 0)  # 1: Focus on finding feasible solutions quickly; 2: Focus on proving optimality; 0: Balance between the two
    set_attribute(int_master, "Threads", 32)  # Use 32 threads for parallel MIP solving
    println("Solving integer master with ", length(y_vars), " routes...")
    optimize!(int_master)
    
    # Get termination status
    status = termination_status(int_master)
    prim_status = primal_status(int_master)
    
    if is_solved_and_feasible(int_master)
        obj_val = objective_value(int_master)
        println("✓ Integer master solution found: ", round(obj_val, digits=2))
        return obj_val, true, int_master
    else
        # Diagnose why it failed
        if status == MOI.INFEASIBLE
            println("Integer master problem is INFEASIBLE")
        elseif status == MOI.TIME_LIMIT
            if prim_status == MOI.FEASIBLE_POINT
                # Time limit but found a feasible solution
                obj_val = objective_value(int_master)
                println("Time limit reached, but found feasible solution: ", round(obj_val, digits=2))
                return obj_val, true, int_master
            else
                println("Time limit reached with NO feasible solution found")
            end
        elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
            println("Integer master problem is INFEASIBLE or UNBOUNDED")
        elseif status == MOI.NUMERICAL_ERROR
            println("Numerical error in integer master problem")
        else
            println("Integer master failed with status: ", status)
        end
        return Inf, false, nothing
    end
end

################################################################
################### B-P Algorithm ##############################
################################################################

function branch_and_price(data::Data)

    n = data.n
    N = data.N

    # Create smart initial routes that respect K constraint
    init_paths = create_initial_routes(data)
    
    # Initialize RMP with these routes
    init_master = RMP(data, init_paths)

    root_node = BranchNode("Level 0 --> root_node", 0, init_master, [], [], init_paths, Tuple{Int,Int}[], Tuple{Int,Int}[])
    branches = [root_node]
    best_ub = Inf
    best_lb = -Inf  # Initialize to -Inf for proper lower bound tracking
    best_solution = [root_node]

    run_cnt = 1
    max_nodes = 150 # Add max nodes limit to prevent infinite branching

    while !isempty(branches) && run_cnt <= max_nodes
        # This explores promising branches first and prunes faster
        sort!(branches, by = x -> length(x.objective) > 0 ? x.objective[end] : Inf)
        branch_node = popfirst!(branches)
        
        # Prune nodes that cannot improve the best solution
        if length(branch_node.objective) > 0 && branch_node.objective[end] >= best_ub
            println(branch_node.name, "--> Pruned before solving (LB=", branch_node.objective[end], " >= UB=", best_ub, ")")
            continue
        end
        
        println("Running ", branch_node.name)
        # U_limit=1500: beam width for labeling algorithm
        # max_cols_per_iter=50: limit columns added per iteration (higher = better LP bound, slower)
        m_obj, r_cost, new_paths = runCG(branch_node.master, data, branch_node.paths, 1500, 
                                        branch_node.must_together, branch_node.must_separate)
        
        branch_node.objective = m_obj
        branch_node.rc = r_cost
        branch_node.paths = new_paths

        if is_solved_and_feasible(branch_node.master)

            ### Find integral solution
            yᵣ = value.(branch_node.master[:y])
            # Check for fractional values: not close to 0 and not close to 1
            # A value is fractional if it's in the range (tol, 1-tol) where tol = 1e-4
            integrality_tol = 1e-4
            frac_yᵣ_indx = findall(x -> x > integrality_tol && x < (1.0 - integrality_tol), yᵣ)
            println("fractional yᵣ index ", frac_yᵣ_indx)
            
            # Check set partitioning constraints
            active_routes_idx = findall(x -> x > 1e-5, yᵣ)
            println("Active routes (y > 1e-5): ", length(active_routes_idx))
            
            # Try solving integer master more aggressively
            # Column generation only gives LP relaxation - we MUST solve integer problem to get actual solutions
            # Try integer master if:
            # 1. At root node (level 0) - always try to get initial integer solution
            # 2. Any fractional variables exist and we have enough routes (> n)
            # 3. No incumbent yet (best_ub == Inf)
            # 4. Deep in tree (level >= 2)
            should_try_integer = false
            if !isempty(frac_yᵣ_indx)
                num_fractional = length(frac_yᵣ_indx)
                num_routes = length(new_paths)
                
                if branch_node.tree_lvl == 0
                    println("Root node - trying integer master to get initial solution...")
                    should_try_integer = true
                elseif best_ub == Inf
                    println("No incumbent yet - trying integer master...")
                    should_try_integer = true
                elseif num_fractional <= 10 && num_routes >= n
                    println("Few fractional variables (", num_fractional, "), trying integer master...")
                    should_try_integer = true
                elseif branch_node.tree_lvl >= 2
                    println("Deep in tree (level ", branch_node.tree_lvl, "), trying integer master...")
                    should_try_integer = true
                elseif best_ub < Inf && branch_node.objective[end] >= 0.90 * best_ub
                    println("Close to best UB, trying integer master...")
                    should_try_integer = true
                end
            end
            
            if should_try_integer
                int_obj, int_feasible, int_master = solve_integer_master(branch_node.master, data)
                if int_feasible && int_obj < best_ub
                    # Found better integer solution
                    best_ub = int_obj
                    # Update the branch node with integer solution
                    branch_node.master = int_master
                    branch_node.objective[end] = int_obj
                    push!(best_solution, branch_node)
                    println("*** New incumbent from integer master! UB = ", best_ub, " ***")
                    if int_obj > best_lb
                        best_lb = int_obj
                    end
                    # Mark as no fractional variables since we have integer solution
                    frac_yᵣ_indx = Int[]
                end
            end

            if isempty(frac_yᵣ_indx) && branch_node.objective[end]<best_ub
                best_ub = branch_node.objective[end]
                push!(best_solution, branch_node)
                println("*** New incumbent solution found! UB = ", best_ub, " ***")
                # Update lower bound since this node is now closed with integer solution
                if branch_node.objective[end] > best_lb
                    best_lb = branch_node.objective[end]
                end
            elseif branch_node.objective[end]>best_ub
                println(branch_node.name, "-->Model was fathomed by bound (LB=", branch_node.objective[end], " > UB=", best_ub, ")")
            else 
                if branch_node.objective[end]<best_lb
                    best_lb = branch_node.objective[end]
                end
                
                # Customer-pair branching: find most fractional pair
                # For SET COVERING: Calculate how often customer pairs appear together
                # A pair is "together" if they appear in the same route that is selected (y > 0)
                # Note: With set covering, a customer may appear in multiple selected routes
                pair_togetherness = Dict{Tuple{Int,Int}, Float64}()
                
                for route_idx in 1:length(new_paths)
                    y_val = yᵣ[route_idx]
                    if y_val > 1e-5
                        route = new_paths[route_idx]
                        # Get customers in this route (excluding depot)
                        customers_in_route = filter(x -> x > 1 && x <= n+1, route)
                        # For each pair of customers in this route
                        for i in 1:length(customers_in_route)
                            for j in (i+1):length(customers_in_route)
                                cust_i = customers_in_route[i]
                                cust_j = customers_in_route[j]
                                pair = (min(cust_i, cust_j), max(cust_i, cust_j))
                                if !haskey(pair_togetherness, pair)
                                    pair_togetherness[pair] = 0.0
                                end
                                # For set covering: accumulate y values where both customers appear together
                                pair_togetherness[pair] += y_val
                            end
                        end
                    end
                end
                
                # Find most fractional pair (closest to 0.5)
                best_pair = nothing
                best_fractionality = Inf
                integrality_tol = 1e-4
                for (pair, together_val) in pair_togetherness
                    # Only branch on truly fractional values (not 0 or 1)
                    if together_val > integrality_tol && together_val < (1.0 - integrality_tol)
                        frac = abs(together_val - 0.5)
                        if frac < best_fractionality
                            best_fractionality = frac
                            best_pair = pair
                        end
                    end
                end
                
                if !isnothing(best_pair)
                    i, j = best_pair
                    together_val = pair_togetherness[best_pair]
                    println("Branching on customer pair (", i-1, ",", j-1, ") with togetherness = ", round(together_val, digits=3))
                    
                    branch_tree_lvl = branch_node.tree_lvl + 1
                    
                    # Left branch: Customers MUST be together on same route
                    together_pairs = copy(branch_node.must_together)
                    push!(together_pairs, (i, j))
                    
                    # Filter routes: keep only those with BOTH i and j, or NEITHER
                    valid_routes_together = Vector{Vector{Int64}}()
                    for route in new_paths
                        has_i = i in route
                        has_j = j in route
                        # Keep route if both present or both absent
                        if (has_i && has_j) || (!has_i && !has_j)
                            # Check existing together constraints
                            satisfies_all = true
                            for (req_i, req_j) in together_pairs
                                has_req_i = req_i in route
                                has_req_j = req_j in route
                                if (has_req_i && !has_req_j) || (!has_req_i && has_req_j)
                                    satisfies_all = false
                                    break
                                end
                            end
                            # Check existing separate constraints
                            for (sep_i, sep_j) in branch_node.must_separate
                                has_sep_i = sep_i in route
                                has_sep_j = sep_j in route
                                if has_sep_i && has_sep_j
                                    satisfies_all = false
                                    break
                                end
                            end
                            if satisfies_all
                                push!(valid_routes_together, route)
                            end
                        end
                    end
                    
                    if !isempty(valid_routes_together)
                        together_master = RMP(data, valid_routes_together)
                        together_name = "Level " *string(branch_tree_lvl)* " customers (" *string(i-1)* "," *string(j-1)* ") TOGETHER"
                        together_branch = BranchNode(together_name, branch_tree_lvl, together_master, [], [], 
                                                    valid_routes_together, together_pairs, copy(branch_node.must_separate))
                        push!(branches, together_branch)
                    end
                    
                    # Right branch: Customers MUST be separated on different routes
                    separate_pairs = copy(branch_node.must_separate)
                    push!(separate_pairs, (i, j))
                    
                    # Filter routes: keep only those without BOTH i and j
                    valid_routes_separate = Vector{Vector{Int64}}()
                    for route in new_paths
                        has_i = i in route
                        has_j = j in route
                        # Keep route if not both present
                        if !(has_i && has_j)
                            # Check existing together constraints
                            satisfies_all = true
                            for (req_i, req_j) in branch_node.must_together
                                has_req_i = req_i in route
                                has_req_j = req_j in route
                                if (has_req_i && !has_req_j) || (!has_req_i && has_req_j)
                                    satisfies_all = false
                                    break
                                end
                            end
                            # Check existing separate constraints
                            for (sep_i, sep_j) in separate_pairs
                                has_sep_i = sep_i in route
                                has_sep_j = sep_j in route
                                if has_sep_i && has_sep_j
                                    satisfies_all = false
                                    break
                                end
                            end
                            if satisfies_all
                                push!(valid_routes_separate, route)
                            end
                        end
                    end
                    
                    if !isempty(valid_routes_separate)
                        separate_master = RMP(data, valid_routes_separate)
                        separate_name = "Level " *string(branch_tree_lvl)* " customers (" *string(i-1)* "," *string(j-1)* ") SEPARATE"
                        separate_branch = BranchNode(separate_name, branch_tree_lvl, separate_master, [], [], 
                                                    valid_routes_separate, copy(branch_node.must_together), separate_pairs)
                        push!(branches, separate_branch)
                    end
                else
                    println("WARNING: No fractional customer pairs found for branching")
                end
            end
        else
            println(branch_node.name, "-->Model was infeasible")
        end
        
        # Update global lower bound (minimum objective of all open nodes)
        if !isempty(branches)
            open_lb = minimum([length(b.objective) > 0 ? b.objective[end] : Inf for b in branches])
            # Keep the best lower bound (could be from a closed node)
            if open_lb < Inf && open_lb > best_lb
                best_lb = open_lb
            end
        end
        
        println("Nodes explored: ", run_cnt, " | Open nodes: ", length(branches), " | Best UB=", best_ub, " | Best LB=", best_lb, " | Gap=", best_ub - best_lb)
        run_cnt += 1
    end
    
    if run_cnt > max_nodes
        println("WARNING: Reached maximum node limit (", max_nodes, "). Solution may be suboptimal.")
    end
    
    # Calculate final gap
    final_gap = best_ub - best_lb
    nodes_explored = run_cnt - 1  # Subtract 1 because run_cnt increments after last iteration
    
    return best_solution[end], best_ub, best_lb, final_gap, nodes_explored

end


################################################################
################### Display Results ############################
################################################################

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
        println("\n ERROR: Duplicate customers found!")
        customer_counts = Dict()
        for c in all_customers
            customer_counts[c] = get(customer_counts, c, 0) + 1
        end
        duplicates = filter(p -> p.second > 1, customer_counts)
        println("Duplicate customers: ", duplicates)
        println("\n DETAILED ANALYSIS:")
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
        println("\n ERROR: ", length(unvisited), " customers UNVISITED!")
        println("Unvisited customer nodes: ", sort(collect(unvisited)))
        println("Unvisited customer indices (for constraints): ", sort([c-1 for c in unvisited]))
    end
    
    # Final verdict
    if isempty(unvisited) && length(all_customers) == length(unique(all_customers))
        println("\n✓ Solution is VALID: All ", n_customers, " customers visited exactly once")
    else
        println("\n Solution is INVALID")
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

    main_folder = "Results_BnP_VehicleConstraint"

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
        write(io, "Upper Bound (UB): ", string(round(results.best_ub, digits=2)), "\n")
        write(io, "Lower Bound (LB): ", string(round(results.best_lb, digits=2)), "\n")
        write(io, "Gap: ", string(round(results.gap, digits=2)), " (", string(round(100*results.gap/max(results.best_ub, 1e-10), digits=2)), "%)\n")
        write(io, "Nodes Explored: ", string(results.nodes_explored), "\n")
        write(io, "Solve Time: ", string(solution_time), " seconds\n")
        write(io, "Cars Used: ", string(size(routes_cg,1)), "\n")
        write(io, "CG iterations: ", string(length(m_obj)), "\n")
        write(io, "\n")
        write(io, "================================\n")
        write(io, "Routes by Customers:\n")
        write(io, "================================\n")
        for (i, r) in enumerate(routes_cg)
            # Extract customers (exclude depot nodes 1 and N_total)
            n_customers = nrow(df[df.node_type .== "customer", :])
            N_total = nrow(df)
            customers_in_route = filter(x -> x > 1 && x < N_total, r)
            customer_indices = [c - 1 for c in customers_in_route]  # Convert to 0-indexed customer IDs
            write(io, "Route ", string(i), ": Nodes ", string(r), "\n")
            write(io, "  Customers served (0-indexed): ", string(customer_indices), "\n")
        end
        write(io, "\n")
        
        # Check for duplicate customers
        all_customers = []
        for route in routes_cg
            n_customers = nrow(df[df.node_type .== "customer", :])
            N_total = nrow(df)
            customers_in_route = filter(x -> x > 1 && x < N_total, route)
            append!(all_customers, customers_in_route)
        end
        
        write(io, "================================\n")
        write(io, "Customer Coverage Analysis:\n")
        write(io, "================================\n")
        write(io, "Total customer visits: ", string(length(all_customers)), "\n")
        write(io, "Unique customers served: ", string(length(unique(all_customers))), "\n")
        
        # Identify duplicates
        if length(all_customers) != length(unique(all_customers))
            customer_counts = Dict()
            for c in all_customers
                customer_counts[c] = get(customer_counts, c, 0) + 1
            end
            duplicates = filter(p -> p.second > 1, customer_counts)
            write(io, "\nDUPLICATE CUSTOMERS FOUND:\n")
            for (cust, count) in sort(collect(duplicates))
                customer_idx = cust - 1
                write(io, "  Customer ", string(customer_idx), " (node ", string(cust), ") appears ", string(count), " times\n")
                for (ridx, route) in enumerate(routes_cg)
                    if cust in route
                        write(io, "    - Route ", string(ridx), "\n")
                    end
                end
            end
        else
            write(io, "No duplicate customers - all customers served exactly once\n")
        end
        write(io, "\n")
        
        write(io, "================================\n")
        write(io, "Routes by Nodes:\n")
        write(io, "================================\n")
        for (i, rn) in enumerate(route_nodes)
            write(io, "Route ", string(i), ": ", string(rn), "\n")
        end
        write(io, "\n")
    end

    PlotlyJS.savefig(img, main_folder * "/" * input_list[1] * "/Customer_loc_w_nodes.png")
    Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/Customer_routes_cg.png")
    Plots.savefig(fig_rc, main_folder * "/" * input_list[1] * "/fig_rc.png")
    Plots.savefig(fig_obj, main_folder * "/" * input_list[1] * "/fig_obj.png")

end



################################################################
################### Run the B-P ################################
################################################################
input_list = ["A-n32-k5.csv", 5, 100.0]
data, df, img = PreProcess("../data/new_dataset/" * input_list[1], input_list[2] , input_list[3])

tic = time()
best_solution, best_ub, best_lb, final_gap, nodes_explored = branch_and_price(data)
toc = time()

solution_time = toc-tic
println("Initial Solution Time(s): ", solution_time)

results = Results(best_solution, solution_time, best_ub, best_lb, final_gap, nodes_explored)
print_results_to_file(results, df)

# Print summary results
println("\n" * "="^60)
println("FINAL SOLUTION SUMMARY")
println("="^60)
println("Objective Value: ", round(best_solution.objective[end], digits=2))
println("Solution Time: ", round(solution_time, digits=2), " seconds")
println("CG Iterations: ", length(best_solution.objective))
println("Total Routes Generated: ", length(best_solution.paths))

# Get final solution routes
if is_solved_and_feasible(best_solution.master)
    y_vals = value.(best_solution.master[:y])
    active_routes = findall(x -> x > 1e-5, y_vals)
    println("\nActive Routes (y > 1e-5): ", length(active_routes))
    println("Vehicles Used: ", round(sum(y_vals), digits=2))
    
    println("\nRoutes:")
    for (idx, route_idx) in enumerate(active_routes)
        route = best_solution.paths[route_idx]
        y_val = y_vals[route_idx]
        println("  Route ", idx, " (y=", round(y_val, digits=3), "): ", route)
    end
end
println("="^60)