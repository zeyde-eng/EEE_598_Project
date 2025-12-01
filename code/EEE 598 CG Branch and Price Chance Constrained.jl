#=
================================================================================
Vehicle Routing Problem with Stochastic Demands (VRPSD)
Chance-Constrained Programming Formulation
Branch-and-Price Solution Algorithm
================================================================================

Author: Zach Eyde
Student ID: 1223877512
Course: EEE 598 Final Project

Key References:
- Christiansen & Lysgaard (2007) - A branch-and-price algorithm for the VRPSD
- Stewart & Golden (1983) - Stochastic vehicle routing with chance constraints

Approach:
- Chance constraints: P(total demand ≤ Q) ≥ 1-α where α is risk tolerance
- For Poisson demands: Total demand ~ Poisson(Σλᵢ) for customers in route
- Uses normal approximation or exact Poisson CDF for constraint checking
- No recourse costs - routes either succeed or fail based on probability threshold

=#

################################################################
############### Import Needed Packages #########################
################################################################

using JuMP
using LinearAlgebra
using DataFrames
using Plots, PlotlyJS
using SparseArrays
using Gurobi
using Distributions
using Random

using CSV
using Distances

# Get Gurobi Env
gurobi_env = Gurobi.Env()

################################################################
############### Define the Data Structure ######################
################################################################

"""
    Data

Core data structure for VRPSD instances with chance constraints.

# Fields
- `n::Int64`: Number of customers (excluding depot)
- `d::Vector{Float64}`: Expected demands for each node (λ for Poisson)
- `c::Array{Float64}`: Distance/cost matrix between all node pairs (N×N)
- `q::Float64`: Vehicle capacity constraint
- `K::Int64`: Maximum number of vehicles available
- `N::Int64`: Total number of nodes (n customers + 2 depot nodes)
- `λ_rate::Vector{Float64}`: Poisson rate parameters for stochastic demands
- `α::Float64`: Risk tolerance for chance constraints (default 0.05 = 95% reliability)

# Chance Constraint
For a route visiting customers S ⊆ {1,...,n}:
P(Σᵢ∈S Dᵢ ≤ Q) ≥ 1-α

where Dᵢ ~ Poisson(λᵢ) and total demand Σᵢ∈S Dᵢ ~ Poisson(Σᵢ∈S λᵢ)
"""
mutable struct Data
    n::Int64
    d::Vector{Float64}
    c::Array{Float64}
    q::Float64
    K::Int64
    N::Int64
    λ_rate::Vector{Float64}
    α::Float64  # Risk tolerance (probability of failure)
end

struct Label
    current_node::Int
    load::Float64            # Expected load (sum of λ values)
    visited::Vector{Bool}
    path::Vector{Int}
    cost::Float64           # Only travel cost, no recourse
    total_lambda::Float64   # Total λ for Poisson feasibility check
end

"""
    BranchNode

Represents a node in the branch-and-price tree.
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

mutable struct Results
    node::BranchNode
    time::Float64
    best_ub::Float64
    best_lb::Float64
    gap::Float64
    nodes_explored::Int
end

"""
    Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, q, α=0.05)

Constructor for VRPSD data structure with chance constraints.

# Arguments
- `α::Float64`: Risk tolerance (default 0.05 means 95% reliability)
"""
function Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, q, α=0.05)
    n = num_Cust
    d = demand
    N = length(xcoord)
    K = max_V
    c = [norm([xcoord[i]; ycoord[i]] - [xcoord[j]; ycoord[j]], 2) for i=1:N, j=1:N]
    λ_rate = demand_rate

    return Data(n, d, c, q, K, N, λ_rate, α)
end

################################################################
############### Import the Data Instance #######################
################################################################

"""
    PreProcess(filename, cars::Int64, capacity::Float64, α::Float64=0.05)

Preprocesses VRPSD instance data from CSV format.

# Arguments
- `α::Float64`: Risk tolerance for chance constraints (default 0.05)
"""
function PreProcess(filename, cars::Int64, capacity::Float64, α::Float64=0.05)

    # Get input csv file
    file_name = filename
    A_raw = DataFrame(CSV.File(file_name))

    # Add Depot to the end with same info as first node which is the depot
    A_end = A_raw[1:1, :]
    A_final = vcat(A_raw, A_end)

    num_Cust = nrow(A_raw) - 1

    xcoord = A_final[:, 2]
    ycoord = A_final[:, 3]
    demand = A_final[:, 4]
    
    # Use demand as λ parameter for Poisson distribution
    demand_rate = A_final[:, 4]

    max_V = cars

    process_data = Data(num_Cust, xcoord, ycoord, demand, demand_rate, max_V, capacity, α)

    # Create visualization
    trace = PlotlyJS.scatter(
        x=xcoord, y=ycoord,
        mode="markers+text",
        text=string.(0:num_Cust),
        textposition="top center",
        marker=attr(size=10, color="blue")
    )
    
    layout = PlotlyJS.Layout(
        title="VRPSD Instance (Chance Constrained - α=$(α))",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate"
    )
    
    fig_plotly = PlotlyJS.plot(trace, layout)

    return process_data, A_final, fig_plotly
end

################################################################
############### Chance Constraint Functions ####################
################################################################

"""
    poisson_quantile(λ::Float64, α::Float64)

Computes the maximum capacity Q_α such that P(Poisson(λ) ≤ Q_α) ≥ 1-α.

For a route with total demand ~ Poisson(λ), this returns the minimum capacity
needed to ensure the route is feasible with probability at least 1-α.

# Arguments
- `λ::Float64`: Total λ for route (sum of customer λ values)
- `α::Float64`: Risk tolerance (probability of failure)

# Returns
- `Q_α::Float64`: Capacity quantile (route needs capacity ≥ Q_α)
"""
function poisson_quantile(λ::Float64, α::Float64)
    if λ < 0
        return 0.0
    end
    
    # For large λ, use normal approximation: X ~ N(λ, λ)
    if λ > 30
        return λ + sqrt(λ) * quantile(Normal(0, 1), 1 - α)
    end
    
    # For small λ, use exact Poisson quantile
    dist = Poisson(λ)
    return quantile(dist, 1 - α)
end

"""
    is_route_feasible(route::Vector{Int}, data::Data)

Checks if a route satisfies the chance constraint.

Returns true if P(total demand ≤ Q) ≥ 1-α
"""
function is_route_feasible(route::Vector{Int}, data::Data)
    # Calculate total λ for customers in route
    total_lambda = 0.0
    for node in route
        if node > 1 && node <= data.n + 1  # Customer nodes only
            total_lambda += data.λ_rate[node]
        end
    end
    
    # Check if capacity quantile is within vehicle capacity
    required_capacity = poisson_quantile(total_lambda, data.α)
    
    return required_capacity <= data.q
end

"""
    route_cost(route::Vector{Int}, data::Data)

Computes the travel cost of a route (no recourse, just distance).
"""
function route_cost(route::Vector{Int}, data::Data)
    cost = 0.0
    for i in 1:(length(route)-1)
        cost += data.c[route[i], route[i+1]]
    end
    return cost
end

################################################################
############### Create Initial Routes ##########################
################################################################

"""
    create_initial_routes(data::Data)

Creates initial feasible routes for the RMP that satisfy chance constraints.
Uses nearest neighbor heuristic, adding customers only if route remains feasible.
"""
function create_initial_routes(data::Data)
    n = data.n
    N = data.N
    K = data.K
    
    routes = Vector{Vector{Int}}()
    unvisited = Set(2:(n+1))  # Customer nodes
    
    # Create routes using nearest neighbor with chance constraint checking
    vehicle_count = 0
    
    while !isempty(unvisited) && vehicle_count < K
        route = [1]  # Start at depot
        current = 1
        current_lambda = 0.0
        
        while !isempty(unvisited)
            # Find nearest unvisited customer
            best_customer = nothing
            best_distance = Inf
            
            for customer in unvisited
                dist = data.c[current, customer]
                if dist < best_distance
                    # Check if adding this customer keeps route feasible
                    test_lambda = current_lambda + data.λ_rate[customer]
                    required_cap = poisson_quantile(test_lambda, data.α)
                    
                    if required_cap <= data.q
                        best_distance = dist
                        best_customer = customer
                    end
                end
            end
            
            if best_customer === nothing
                break  # No feasible customer to add
            end
            
            # Add customer to route
            push!(route, best_customer)
            delete!(unvisited, best_customer)
            current = best_customer
            current_lambda += data.λ_rate[best_customer]
        end
        
        push!(route, N)  # Return to depot
        push!(routes, route)
        vehicle_count += 1
        
        println("Initial route $vehicle_count: ", route, " (λ_total=$(round(current_lambda, digits=2)), " *
                "Q_α=$(round(poisson_quantile(current_lambda, data.α), digits=2)))")
    end
    
    # If not all customers covered, add individual customer routes
    for customer in unvisited
        route = [1, customer, N]
        push!(routes, route)
        println("Individual route for customer $(customer-1)")
    end
    
    println("\nCreated $(length(routes)) initial routes covering $n customers")
    
    return routes
end

################################################################
############### Restricted Master Problem (RMP) ################
################################################################

"""
    RMP(data::Data, initial_routes::Vector{Vector{Int}})

Creates the Restricted Master Problem for chance-constrained VRPSD.
Uses set partitioning formulation with pure travel costs (no recourse).
"""
function RMP(data::Data, initial_routes::Vector{Vector{Int}})
    n = data.n
    N = data.N
    c = data.c
    K = data.K

    Ω = length(initial_routes)
    c_r = Float64[]
    
    # Calculate travel cost for each route (no recourse cost)
    for route in initial_routes
        cost = route_cost(route, data)
        push!(c_r, cost)
    end

    # Build A matrix: A[i,r] = 1 if customer i is in route r
    A = zeros(Int, n, Ω)
    for (r_idx, route) in enumerate(initial_routes)
        for node in route
            if node != 1 && node != N
                customer_idx = node - 1
                A[customer_idx, r_idx] = 1
            end
        end
    end
    
    # Define model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))
    set_attribute(model, "TimeLimit", 600)
    set_attribute(model, "LogToConsole", 0)
    set_attribute(model, "Threads", 4)
    set_attribute(model, "Method", 2)

    # Decision variables
    @variable(model, y[r = 1:Ω] >= 0)

    # Objective: minimize total travel cost
    @objective(model, Min, sum(c_r[r]*y[r] for r = 1:Ω))

    # Set partitioning: each customer visited exactly once
    @constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) == 1)
    
    # Fleet size constraint: use at most K vehicles
    @constraint(model, fleet_constr, sum(y[r] for r=1:Ω) <= K)

    return model
end

################################################################
############### Labeling Algorithm (Pricing Problem) ###########
################################################################

"""
    forward_labeling(c::Array{Float64}, data::Data, U_limit::Int, 
                    must_together::Vector{Tuple{Int,Int}}, 
                    must_separate::Vector{Tuple{Int,Int}})

Labeling algorithm for generating routes with chance constraints.
Routes are only extended if they remain feasible under the chance constraint.
"""
function forward_labeling(c::Array{Float64}, data::Data, U_limit::Int,
                         must_together::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                         must_separate::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[])
    
    n = data.n
    N = data.N
    q = data.q
    α = data.α
    
    # Initialize labels at depot
    initial_visited = fill(false, N)
    initial_label = Label(1, 0.0, initial_visited, [1], 0.0, 0.0)
    
    # Labels by node: node_id => Vector of labels
    labels = Dict{Int, Vector{Label}}()
    labels[1] = [initial_label]
    
    # Final labels (routes that return to depot)
    final_labels = Label[]
    
    # Process each stage
    for stage in 1:n+1
        new_labels = Dict{Int, Vector{Label}}()
        
        for (node, node_labels) in labels
            # Limit labels per node (beam search)
            if length(node_labels) > U_limit
                sort!(node_labels, by = l -> l.cost)
                node_labels = node_labels[1:U_limit]
            end
            
            for label in node_labels
                # Try extending to each customer
                for next_node in 2:(n+1)
                    if label.visited[next_node]
                        continue
                    end
                    
                    # Calculate new load
                    new_lambda = label.total_lambda + data.λ_rate[next_node]
                    
                    # Check chance constraint feasibility
                    required_capacity = poisson_quantile(new_lambda, α)
                    if required_capacity > q
                        continue  # Route would violate chance constraint
                    end
                    
                    # Check branching constraints
                    if !check_branching_constraints(label.path, next_node, 
                                                   must_together, must_separate)
                        continue
                    end
                    
                    # Create new label
                    new_visited = copy(label.visited)
                    new_visited[next_node] = true
                    new_path = vcat(label.path, next_node)
                    new_cost = label.cost + c[label.current_node, next_node]
                    
                    new_label = Label(next_node, new_lambda, new_visited, 
                                    new_path, new_cost, new_lambda)
                    
                    # Check dominance before adding
                    if !is_dominated(new_label, get(new_labels, next_node, Label[]))
                        if !haskey(new_labels, next_node)
                            new_labels[next_node] = Label[]
                        end
                        push!(new_labels[next_node], new_label)
                    end
                end
                
                # Try returning to depot
                if !isempty(label.path) && label.current_node != 1
                    return_cost = label.cost + c[label.current_node, N]
                    final_path = vcat(label.path, N)
                    
                    final_label = Label(N, label.load, label.visited, 
                                      final_path, return_cost, label.total_lambda)
                    
                    # Only add if route has negative reduced cost
                    if final_label.cost < -1e-6
                        push!(final_labels, final_label)
                    end
                end
            end
        end
        
        labels = new_labels
        
        if isempty(labels)
            break
        end
    end
    
    return final_labels
end

"""
    is_dominated(label1::Label, labels::Vector{Label})

Checks if label1 is dominated by any label in labels.
Label dominates if: same or better node, less/equal load, less cost, subset visited.
"""
function is_dominated(label1::Label, labels::Vector{Label})
    for label2 in labels
        if label2.current_node == label1.current_node &&
           label2.load <= label1.load &&
           label2.cost <= label1.cost &&
           all(label1.visited .| .!label2.visited)
            return true
        end
    end
    return false
end

"""
    check_branching_constraints(path::Vector{Int}, next_node::Int,
                                must_together::Vector{Tuple{Int,Int}},
                                must_separate::Vector{Tuple{Int,Int}})

Checks if adding next_node to path violates branching constraints.
"""
function check_branching_constraints(path::Vector{Int}, next_node::Int,
                                    must_together::Vector{Tuple{Int,Int}},
                                    must_separate::Vector{Tuple{Int,Int}})
    path_customers = filter(x -> x > 1 && x <= length(path), path)
    
    # Check must_together constraints
    for (i, j) in must_together
        node_i = i + 1
        node_j = j + 1
        
        if next_node == node_i && !(node_j in path_customers)
            # Would need to add j later
            continue
        end
        if next_node == node_j && !(node_i in path_customers)
            # Would need to add i later
            continue
        end
    end
    
    # Check must_separate constraints
    for (i, j) in must_separate
        node_i = i + 1
        node_j = j + 1
        
        if next_node == node_i && node_j in path_customers
            return false
        end
        if next_node == node_j && node_i in path_customers
            return false
        end
    end
    
    return true
end

################################################################
############### Column Generation ##############################
################################################################

"""
    is_solved_and_feasible(model::Model)

Checks if optimization was successful and solution is feasible.
"""
function is_solved_and_feasible(model::Model)
    status = termination_status(model)
    return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
end

"""
    runCG(master::Model, data::Data, paths::Vector{Vector{Int}}, U_limit::Int,
          must_together, must_separate, max_cols_per_iter::Int=300)

Column generation for chance-constrained VRPSD.
"""
function runCG(master::Model, data::Data, paths::Vector{Vector{Int}}, U_limit::Int, 
               must_together::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[], 
               must_separate::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
               max_cols_per_iter::Int=300)
    
    n = data.n
    c = data.c
    N = data.N

    m_obj = Float64[]
    r_cost = Float64[]

    max_iter = 100
    iter = 0

    while iter < max_iter
        iter += 1
        optimize!(master)

        if is_solved_and_feasible(master)
            println("\n=== CG Iteration ", iter, " ===")
            println("Master Problem Objective Value = ", round(objective_value(master), digits=2))
            push!(m_obj, objective_value(master))
            
            y_vals = value.(master[:y])
            active_routes = findall(x -> x > 1e-5, y_vals)
            println("Active routes: ", length(active_routes))
            println("Vehicles used: ", round(sum(y_vals), digits=2), " / ", data.K)

            # Get dual variables
            π = dual.(master[:sp_constr])
            μ = dual(master[:fleet_constr])  # Fleet size dual
            
            # Create modified cost matrix
            ĉ = deepcopy(c)
            ĉ[2:n+1,:] .-= π
            
            # Adjust for fleet constraint dual (applies to all routes)
            # Each route contributes 1 to fleet constraint
            fleet_cost_adjustment = μ
            
            cols_added = 0
            cols = forward_labeling(ĉ, data, U_limit, must_together, must_separate)
            
            if !isnothing(cols) && !isempty(cols)
                sort!(cols, by = x -> x.cost)
                
                # Filter for improving columns (considering fleet dual)
                improving_cols = filter(x -> (x.cost + fleet_cost_adjustment) < -1e-6, cols)
                cols_to_add = improving_cols[1:min(max_cols_per_iter, length(improving_cols))]
                
                println("  Labeling found ", length(cols), " columns, ", 
                       length(improving_cols), " improving, adding ", length(cols_to_add))
                
                for col in cols_to_add
                    cols_added += 1
                    
                    # Calculate actual route cost
                    c_r_new = route_cost(col.path, data)
                    
                    push!(r_cost, col.cost + fleet_cost_adjustment)
                    push!(paths, col.path)
                    
                    # Create new A column
                    A_new = zeros(Int, n)
                    for node in col.path
                        if node > 1 && node <= n+1
                            customer_idx = node - 1
                            A_new[customer_idx] = 1
                        end
                    end
                    
                    # Add new variable to master
                    push!(master[:y], @variable(master, lower_bound = 0.0))
                    set_objective_coefficient(master, master[:y][end], c_r_new)
                    
                    # Set coefficients in constraints
                    for i in 1:n
                        set_normalized_coefficient(master[:sp_constr][i], master[:y][end], A_new[i])
                    end
                    set_normalized_coefficient(master[:fleet_constr], master[:y][end], 1.0)
                end
                
                println("Columns added: ", cols_added)
                
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
        println("WARNING: Reached maximum CG iterations (", max_iter, ")")
    end

    return m_obj, r_cost, paths
end

################################################################
############### Integer Master Problem ########################
################################################################

"""
    solve_integer_master(master::Model, data::Data)

Solves the integer version of the master problem.
"""
function solve_integer_master(master::Model, data::Data)
    int_master = copy(master)
    
    y_vars = int_master[:y]
    for i in 1:length(y_vars)
        set_binary(y_vars[i])
    end
    
    set_optimizer(int_master, () -> Gurobi.Optimizer(gurobi_env))
    set_attribute(int_master, "TimeLimit", 600)
    set_attribute(int_master, "LogToConsole", 0)
    set_attribute(int_master, "MIPGap", 0.01)
    set_attribute(int_master, "MIPFocus", 2)
    
    println("Solving integer master with ", length(y_vars), " routes...")
    optimize!(int_master)
    
    if is_solved_and_feasible(int_master)
        obj_val = objective_value(int_master)
        println("Integer master solution found: ", round(obj_val, digits=2))
        return obj_val, true, int_master
    else
        println("Integer master problem infeasible or time limit reached")
        return Inf, false, nothing
    end
end

################################################################
############### Branch-and-Price Algorithm #####################
################################################################

"""
    branch_and_price(data::Data)

Main branch-and-price algorithm for chance-constrained VRPSD.
"""
function branch_and_price(data::Data)
    n = data.n
    N = data.N

    init_paths = create_initial_routes(data)
    init_master = RMP(data, init_paths)

    root_node = BranchNode("Level 0 --> root_node", 0, init_master, [], [], 
                          init_paths, Tuple{Int,Int}[], Tuple{Int,Int}[])
    branches = [root_node]
    best_ub = Inf
    best_lb = -Inf
    best_solution = [root_node]

    run_cnt = 1
    max_nodes = 150

    while !isempty(branches) && run_cnt <= max_nodes
        sort!(branches, by = x -> length(x.objective) > 0 ? x.objective[end] : Inf)
        branch_node = popfirst!(branches)
        
        if length(branch_node.objective) > 0 && branch_node.objective[end] >= best_ub
            println(branch_node.name, "--> Pruned (LB >= UB)")
            continue
        end
        
        println("\n" * "="^60)
        println("Running ", branch_node.name)
        println("="^60)
        
        m_obj, r_cost, new_paths = runCG(branch_node.master, data, branch_node.paths, 
                                        1500, branch_node.must_together, 
                                        branch_node.must_separate, 300)
        
        branch_node.objective = m_obj
        branch_node.rc = r_cost
        branch_node.paths = new_paths

        if is_solved_and_feasible(branch_node.master)
            yᵣ = value.(branch_node.master[:y])
            fractional_vars = findall(x -> x > 1e-4 && x < 0.9999, yᵣ)
            
            lb = length(m_obj) > 0 ? m_obj[end] : Inf
            best_lb = max(best_lb, lb)
            
            println("\nLP Solution: ", round(lb, digits=2))
            println("Fractional variables: ", length(fractional_vars))
            
            if isempty(fractional_vars)
                # Integer solution found
                println("✓ INTEGER SOLUTION FOUND")
                if lb < best_ub
                    best_ub = lb
                    best_solution = [branch_node]
                    println("New best UB: ", round(best_ub, digits=2))
                end
            else
                # Try solving integer master
                int_obj, int_feasible, int_master = solve_integer_master(branch_node.master, data)
                
                if int_feasible && int_obj < best_ub
                    best_ub = int_obj
                    branch_node_int = deepcopy(branch_node)
                    branch_node_int.master = int_master
                    best_solution = [branch_node_int]
                    println("New best UB from integer solve: ", round(best_ub, digits=2))
                end
                
                # Branch on fractional pair
                if run_cnt < max_nodes && best_ub - best_lb > 0.01
                    new_branches = branch_on_customer_pair(branch_node, yᵣ, data, run_cnt)
                    append!(branches, new_branches)
                end
            end
            
            gap = best_ub < Inf ? 100 * (best_ub - best_lb) / best_ub : Inf
            println("\nCurrent Gap: ", round(gap, digits=2), "%")
            println("Best LB: ", round(best_lb, digits=2), " | Best UB: ", round(best_ub, digits=2))
        end
        
        run_cnt += 1
    end
    
    println("\n" * "="^60)
    println("BRANCH-AND-PRICE COMPLETED")
    println("="^60)
    println("Nodes explored: ", run_cnt - 1)
    println("Best LB: ", round(best_lb, digits=2))
    println("Best UB: ", round(best_ub, digits=2))
    gap = best_ub < Inf ? 100 * (best_ub - best_lb) / best_ub : Inf
    println("Final Gap: ", round(gap, digits=2), "%")
    
    return best_solution, best_ub, best_lb, gap, run_cnt - 1
end

"""
    branch_on_customer_pair(node::BranchNode, y_vals::Vector{Float64}, 
                           data::Data, level::Int)

Creates two child nodes by branching on a customer pair.
"""
function branch_on_customer_pair(node::BranchNode, y_vals::Vector{Float64}, 
                                data::Data, level::Int)
    n = data.n
    
    # Find most fractional customer pair
    pair_scores = Dict{Tuple{Int,Int}, Float64}()
    
    for (r_idx, y_val) in enumerate(y_vals)
        if y_val < 1e-5
            continue
        end
        
        route = node.paths[r_idx]
        customers = filter(x -> x > 1 && x <= n+1, route)
        
        for i in 1:length(customers)
            for j in (i+1):length(customers)
                cust_i = customers[i] - 1
                cust_j = customers[j] - 1
                pair = (min(cust_i, cust_j), max(cust_i, cust_j))
                
                if !haskey(pair_scores, pair)
                    pair_scores[pair] = 0.0
                end
                pair_scores[pair] += y_val
            end
        end
    end
    
    # Find most fractional pair
    best_pair = nothing
    best_score = -Inf
    
    for (pair, score) in pair_scores
        fractionality = min(score, 1.0 - score)
        if fractionality > best_score
            best_score = fractionality
            best_pair = pair
        end
    end
    
    if best_pair === nothing
        return BranchNode[]
    end
    
    (i, j) = best_pair
    
    # Create left branch: i and j together
    left_master = copy(node.master)
    left_must_together = vcat(node.must_together, [best_pair])
    left_node = BranchNode(
        "Level $(level) --> customers ($i,$j) together",
        level,
        left_master,
        [],
        [],
        copy(node.paths),
        left_must_together,
        copy(node.must_separate)
    )
    
    # Create right branch: i and j separate
    right_master = copy(node.master)
    right_must_separate = vcat(node.must_separate, [best_pair])
    right_node = BranchNode(
        "Level $(level) --> customers ($i,$j) separate",
        level,
        right_master,
        [],
        [],
        copy(node.paths),
        copy(node.must_together),
        right_must_separate
    )
    
    return [left_node, right_node]
end

################################################################
############### Output and Visualization #######################
################################################################

"""
    print_solution(best_solution::Vector{BranchNode}, data::Data)

Prints the final solution with route details and feasibility checks.
"""
function print_solution(best_solution::Vector{BranchNode}, data::Data)
    if isempty(best_solution)
        println("No solution found")
        return
    end
    
    node = best_solution[1]
    y_vals = value.(node.master[:y])
    
    println("\n" * "="^60)
    println("FINAL SOLUTION (Chance Constrained α=$(data.α))")
    println("="^60)
    
    route_num = 1
    total_cost = 0.0
    
    for (idx, y_val) in enumerate(y_vals)
        if y_val > 0.5
            route = node.paths[idx]
            cost = route_cost(route, data)
            total_cost += cost
            
            # Calculate route statistics
            customers = filter(x -> x > 1 && x <= data.n+1, route)
            total_lambda = sum(data.λ_rate[c] for c in customers)
            required_cap = poisson_quantile(total_lambda, data.α)
            
            println("\nRoute $route_num:")
            println("  Path: ", [x == 1 ? 0 : (x == data.N ? 0 : x-1) for x in route])
            println("  Customers: ", [c-1 for c in customers])
            println("  Cost: ", round(cost, digits=2))
            println("  Expected demand (Σλ): ", round(total_lambda, digits=2))
            println("  Required capacity (Q_α): ", round(required_cap, digits=2))
            println("  Vehicle capacity: ", data.q)
            println("  Feasible: ", required_cap <= data.q ? "✓" : "✗")
            println("  Success probability: ", round(100*(1-data.α), digits=1), "%")
            
            route_num += 1
        end
    end
    
    println("\n" * "="^60)
    println("Total routes: ", route_num - 1)
    println("Total cost: ", round(total_cost, digits=2))
    println("Risk tolerance α: ", data.α)
    println("Service level: ", round(100*(1-data.α), digits=1), "%")
    println("="^60)
end

################################################################
############### Main Execution #################################
################################################################

"""
    solve_vrpsd_chance_constrained(filename::String, K::Int, Q::Float64, α::Float64=0.05)

Main function to solve VRPSD with chance constraints.

# Arguments
- `filename`: Path to instance CSV file
- `K`: Number of vehicles
- `Q`: Vehicle capacity
- `α`: Risk tolerance (default 0.05 = 95% service level)
"""
function solve_vrpsd_chance_constrained(filename::String, K::Int, Q::Float64, α::Float64=0.05)
    println("="^60)
    println("VRPSD CHANCE CONSTRAINED SOLVER")
    println("="^60)
    println("Instance: ", filename)
    println("Vehicles: ", K)
    println("Capacity: ", Q)
    println("Risk tolerance α: ", α)
    println("Service level: ", round(100*(1-α), digits=1), "%")
    println("="^60)
    
    # Load data
    data, df, fig = PreProcess(filename, K, Q, α)
    
    # Solve
    start_time = time()
    best_solution, best_ub, best_lb, gap, nodes = branch_and_price(data)
    solve_time = time() - start_time
    
    # Print results
    print_solution(best_solution, data)
    
    println("\nComputational Statistics:")
    println("  Solve time: ", round(solve_time, digits=2), " seconds")
    println("  Nodes explored: ", nodes)
    println("  Final gap: ", round(gap, digits=2), "%")
    
    return best_solution, data, solve_time
end


solution, data, time = solve_vrpsd_chance_constrained(
    "../data/new_dataset/A-n32-k5.csv",
    5, 100.0, 0.05
)
