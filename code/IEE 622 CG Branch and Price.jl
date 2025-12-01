#=
IEE 622 Final Project
Zach Eyde
ID: 1223877512


Based on Following Paper

=#


################################################################
############### Import Needed Packages #########################
################################################################

# Start by Importing Needed Packages
# Uncomment Line Below if need to Install Packages
# import Pkg; Pkg.add("DataFrames");Pkg.add("CSV");Pkg.add("Distances");Pkg.add("PlotlyJS");Pkg.add("Gurobi");Pkg.add("DelimitedFiles")


using JuMP
using LinearAlgebra
using DataFrames
using Plots, PlotlyJS
using SparseArrays
using Gurobi, GLPK

# Import data from CSV
using CSV
using DataFrames
using Distances


#Get Gurobi Env
gurobi_env = Gurobi.Env()

################################################################
############### Define the Data Structure ######################
################################################################

mutable struct Data
    n::Int64
    d::Vector{Float64}
    c::Array{Float64}
    t::Array{Float64}
    s::Vector{Float64}
    q::Float64
    K::Int64
    a::Vector{Float64}
    b::Vector{Float64}
    N::Int64
    bigM::Array{Float64}
    bigW::Array{Float64}
end

# Step 1: Define the necessary data structures
mutable struct Label
    label::String
    node::Int
    cost::Float64
    time::Float64
    load::Int
    path::Vector{Int}
    Open_Req::Vector{Int}
end

mutable struct Graph
    Nodes::Vector{Int}
    Arcs::Vector{Tuple{Int,Int}}
end


mutable struct BranchNode
    name::String
    tree_lvl::Int
    master::Model
    objective::Vector{Float64}
    rc::Vector{Float64}
    paths::Vector{Vector{Int}}
end

mutable struct Results
    node::BranchNode
    time::Float64
end

################################################################
############### User define Base Functions #####################
################################################################



################################################################
############### User define Functions ##########################
################################################################


# Initialize Data
function Data(num_Cust, xcoord, ycoord, demand, max_V, init_a, init_b, s, q; obj_func=1)
    n = num_Cust
    d = demand
    N = length(xcoord)
    K = max_V
    t = [norm([xcoord[i]; ycoord[i]] - [xcoord[j]; ycoord[j]], 2) for i=1:N, j=1:N]

    if obj_func == 1
        c = zeros(Float64,N,N)
        for j in 2:N-1
            c[1,j] = 1
        end
    elseif obj_func == 2
        c = deepcopy(t)
    end

    bigM = zeros(Float64,N,N)
    bigW = zeros(Float64,N,N)

    a = init_a
    b = init_b

    # Define BigM
    for i in 1:N
        for j in 1:N
            bigM[i,j] = max(s[i] + b[i] + t[i,j] - a[j],0)
            bigW[i,j] = min(q, q +  d[i])
        end
    end

    return Data(n, d, c, t, s, q, K, a, b, N, bigM, bigW)
end


################################################################
############### Import the Data Instance #######################
################################################################

function PreProcess(filename, num_Cust::Int64, cars::Int64, capacity::Float64, obj::Int64)
    # filename = "lrc101.csv"

    # reading the csv file and saving to Dataframe A
    A = CSV.read(joinpath(@__DIR__, filename), DataFrame; delim = ",")
    A[!, :TYPE] .= ""

    # Select number of Customers to service, n
    # num_Cust = 10

    # Sort dataframe A by first 10 pickups
    A_sort = sort(A, [:PICKUP])

    # Select the Depot and First n customer for Pickup
    depot = DataFrame(A[1,:])
    depot[:,:TYPE] .= "DEPOT"

    pickup = A_sort[2:num_Cust+1,:] 
    pickup[:,:TYPE] .= "PICKUP"

    # create empty dataframe 
    delivery = similar(A,0)

    # Select the corresponding delivery
    for i in 1:num_Cust
        # Select correspsonding delivery node
        row = A[(A.NODE .== pickup.DELIVERY[i]),:]
        append!(delivery, row)
    end

    delivery[:,:TYPE] .= "DELIVERY"

    # Convert in a final dataframe
    A_final = vcat(depot, pickup, delivery, depot, cols=:union)
    A_final.row_num = 0:nrow(A_final)-1

    # plot location of depot and customers
    fig = Plots.scatter(depot[:,:X], depot[:,:Y], label="Depot", aspect_ratio = 1, legend = :outertopright, xticks=:true, yticks=:true)
    Plots.scatter!(fig, pickup[:,:X], pickup[:,:Y], label="Pickup")
    Plots.scatter!(fig, delivery[:,:X], delivery[:,:Y], label="Delivery")
    


    fig_plotly = PlotlyJS.plot(
            A_final, x=:X, y=:Y, color=:TYPE, text=:row_num,
            mode="markers+text", marker=PlotlyJS.attr(size=12, line=PlotlyJS.attr(width=2, color="DarkSlateGrey")),
            textposition="top center",
    )

    #display(fig)
    #display(fig_plotly)

    xcoord = A_final[:,:X]
    ycoord = A_final[:,:Y]
    demand = A_final[:,:DEMAND] # demand
    a = A_final[:,:START] # start time of service window
    b = A_final[:,:END] # ending time of service window
    s = A_final[:,:SERVICE] # service time
    q = capacity # vehicle capacity
    max_V = cars # max vehicles 

    process_data = Data(num_Cust, xcoord, ycoord, demand, max_V, a, b, s, q, obj_func=obj)
    return process_data, A_final, [fig fig_plotly]
end

################################################################
############### Define the RMP Formulation ####################
################################################################

function RMP(data::Data)

    n = data.n
    N = data.N
    c = data.c

    c_r = []

    # Calculate the cost of each path
    for j = 2:n+1
        push!(c_r,c[1,j]+c[j,n+j]+c[n+j,N])
    end

    # Define initial A Matrix
    A = Matrix{Int}(I,n,n)


    # NUmber of Paths
    Ω = size(A,2)

    # define empty model
    model = Model(() -> Gurobi.Optimizer(gurobi_env))
    set_attribute(model, "TimeLimit", 600)
    set_attribute(model, "LogToConsole", 0)

    # mute the output
    # set_silent(model)

    # add decision variable
    @variable(model, y[r = 1:Ω] >= 0)

    # add objective
    @objective(model, Min, sum(c_r[r]*y[r] for r = 1:Ω) )

    # add constraints
    # set partitioning
    @constraint(model, sp_constr[i=1:n], sum(A[i,r]*y[r] for r=1:Ω) == 1)

    return model

end


################################################################
############### Define the Labeling Algo #######################
################################################################

# Step 2: Initialize the algorithm
function initialize_labels(start_node::Int)
    name = string(start_node)
    open_req = []
    path = [start_node]

    return [Label(name, start_node, 0.0, 0.0, 0.0, path, open_req)]
end

# Step 3: Define the extension function
function extend_label(P::Vector{Int}, D::Vector{Int}, n::Int, label::Label, arc::Tuple{Int,Int}, arc_cost::Float64, 
    service_time::Float64, travel_time::Float64, demand::Float64, time_window::Tuple{Float64, Float64}, vehicle_capacity::Float64)

    j = arc[2]
    
    new_cost = label.cost + arc_cost
    new_time = max(label.time + service_time + travel_time, time_window[1])
    new_load = label.load + demand
    new_path = vcat(label.path, j)

    new_name = ""
    for node in label.path
        new_name = new_name * string(node) * "." 
    end
    new_name = new_name * string(j)

    label_open_req = deepcopy(label.Open_Req)
    if (j in P && j ∉ label.path)
        new_open_req = vcat(label_open_req ,j)
    elseif (j in D && (j-n) in label.Open_Req)
        new_open_req = deleteat!(label_open_req, label_open_req .== (j-n))
    elseif (j == (2*n+2) && isempty(label.Open_Req) && new_load == 0)
        new_open_req = label_open_req
    else 
        return nothing
    end

    if new_time <= time_window[2] && new_load <= vehicle_capacity
        return Label(new_name, j, new_cost, new_time, new_load, new_path, new_open_req)
    else
        return nothing
    end
end

# Step 4: Implement dominance rules
function dominates(label1::Label, label2::Label)
    return (label1.node == label2.node && label1.cost <= label2.cost 
            && label1.time <= label2.time && label1.path ⊆ label2.path 
            && label1.Open_Req ⊆ label2.Open_Req)
end


function preprocess_graph(data::Data, neighbors::Int, cost::Array{Float64})

    N = data.N
    n = data.n
    a = data.a
    b = data.b
    d = data.d
    s = data.s
    t = data.t
    q = data.q

    new_a = deepcopy(a)
    new_b = deepcopy(b)
    P = collect(2:n+1)

    Nodes = collect(1:N)
    
    Arcs = []
    if neighbors == 0
        for i in Nodes
            for j in Nodes if i ≠ j
                arc = (i,j)
                push!(Arcs, arc)
            end end
        end
    else
        for i in 2:N-1
            pickup_cost = partialsortperm(cost[i,2:n+1], 1:neighbors)
            delivery_cost = partialsortperm(cost[i,n+2:N-1], 1:neighbors)

            for j in pickup_cost
                if i != j+1
                    arc = (i,j+1)
                    push!(Arcs, arc)
                end
            end

            for j in delivery_cost
                if i != j+n+1
                    arc = (i,j+n+1)
                    push!(Arcs, arc)
                end
            end

        end

        for i in 2:n+1
            if isempty(filter(x->x == (i,n+i), Arcs))
                arc = (i,n+i)
                push!(Arcs, arc)
            end

            if isempty(filter(x->x == (1,i), Arcs))
                arc = (1,i)
                push!(Arcs, arc)
            end

            if isempty(filter(x->x == (n+i,N), Arcs))
                arc = (n+i,N)
                push!(Arcs, arc)
            end
        end
    end

    # println("Arcs before pruning:", length(Arcs))
    # reduction of Arcs based on Dumas Paper
    # paper

    
    for i in P
        # a) priority
        filter!(x->x ≠ (1,n+i), Arcs)
        filter!(x->x ≠ (n+i,i), Arcs)
        filter!(x->x ≠ (N,1), Arcs)
        filter!(x->x ≠ (N,i), Arcs)
        filter!(x->x ≠ (N,n+i), Arcs)

        # b) pairing
        filter!(x->x ≠ (i,N), Arcs)

        # c) vehicle capacity
        for j in P if i ≠ j
            if d[i]+d[j]>q
                filter!(x->x ≠ (i,j), Arcs)
                filter!(x->x ≠ (j,i), Arcs)
                filter!(x->x ≠ (i,n+j), Arcs)
                filter!(x->x ≠ (j,n+i), Arcs)
                filter!(x->x ≠ (n+i,n+j), Arcs)
                filter!(x->x ≠ (n+j,n+i), Arcs)
            end
        end end
    end

    for i in 2:N-1
        for j in 2:N-1
            if a[i]+s[i]+t[i,j] > b[j]
                filter!(x->x ≠ (i,j), Arcs)
            end
        end
    end
    # println("Arcs after pruning:", length(Arcs))


    return Graph(Nodes, Arcs)

end


function labeling_algortihm(data::Data, graph::Graph, ĉ, col2return::Int, U_lim::Int)
    
    n = data.n
    N = data.N
    t = data.t
    s = data.s
    a = data.a
    b = data.b
    q = data.q
    d = data.d
    c = data.c
    q = data.q


    P = collect(2:n+1)
    D = collect(n+2:2*n+1)
    
    arcs = graph.Arcs
    

    U = initialize_labels(1)
    L_f = [] 
    cnt = 0
    while !isempty(U) 
        L = popfirst!(U)
        i = L.node
        if i == N
            push!(L_f, L)
        else # i < N
            
            # define arc list where tuple is (i, whatever)
            arc_list = []
            for (to, from) in arcs
                if to == i
                    push!(arc_list,(to,from))
                end
            end

            # Lᵢ is the set of processed labels at node i (paths ending at node i)
            Lᵢ = filter(x -> x.node == i, L_f)
            keep = true
            if !isempty(Lᵢ)
                for label in Lᵢ
                    if dominates(label, L)
                        keep = false
                        break
                    end
                end
                                
                if keep
                    push!(L_f,L)
                end
            end

            if keep 
                for (i,j) in arc_list # extending labels to all arcs leaving node i
                    new_L = extend_label(P, D, n, L, (i, j), ĉ[i, j], s[i], t[i, j], d[j], (a[j],b[j]), q)
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


function runCG(master::Model, data::Data, col2return::Int, neighbors::Vector{Int}, paths::Vector{Vector{Int}}, U_limit::Int)
    
    n = data.n
    c = data.c
    N = data.N

    m_obj = []
    r_cost = []

    indx = 1

    while true
        # solve the restricted master problem
        optimize!(master)

        if is_solved_and_feasible(master)
        
            println("Master Problem Objective Value = ", objective_value(master))
            push!(m_obj, objective_value(master))

            # get duals
            π = dual.(master[:sp_constr])

            ĉ = deepcopy(c)
            ĉ[2:n+1,:] .-= π

            graph = preprocess_graph(data, neighbors[indx], ĉ)

            cols = labeling_algortihm(data, graph, ĉ, col2return, U_limit)
            if !isnothing(cols) println("Number of Columns Added-->",length(cols)) end

            if !isnothing(cols)
                for col in cols
                    
                    reduced_cost = col.cost
                    push!(r_cost, reduced_cost)
                    push!(paths, col.path)
                    # Create new A column
                    A_new = zeros(Int, n)
                    new_path = filter(x->(x>1) && (x<=n+1), col.path)
                    for node in new_path
                        A_new[node-1] = 1
                    end
                    
                    # create new c_r
                    c_r_new = 0.0
                    for i in 1:(length(col.path)-1)
                        c_r_new += c[col.path[i],col.path[i+1]]
                    end
                    
                    push!(master[:y], @variable(master, lower_bound = 0.0))
                    set_objective_coefficient.(master, master[:y][end], c_r_new)
                    set_normalized_coefficient.(master[:sp_constr], master[:y][end], A_new)
                    end
            else
                if indx < length(neighbors)
                    indx += 1
                else
                    break
                end
            end
        else
            break
        end

    end

    return m_obj, r_cost, paths
end

function branch_and_price(data::Data, col2return::Int, neighbors::Vector{Int}, U_limit::Int)

    n = data.n
    N = data.N

    init_master = RMP(data)
    # Set up initial paths
    paths = []

    for j = 1:n
        push!(paths,[1, j, n+j, N])
    end

    root_node = BranchNode("Level 0 --> root_node", 0, init_master, [], [], paths)
    branches = [root_node]
    best_ub = Inf
    best_lb = Inf
    best_solution = [root_node]

    run_cnt = 1

    while !isempty(branches) #&& run_cnt < 15
        branch_node = popfirst!(branches)
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

            if isempty(frac_yᵣ_indx) && branch_node.objective[end]<best_ub
                best_ub = branch_node.objective[end]
                push!(best_solution, branch_node)
            elseif branch_node.objective[end]>best_ub
                println(branch_node.name, "-->Model was fathomed by bound")
            else # branch on first non-integral variable
                if branch_node.objective[end]<best_lb
                    best_lb = branch_node.objective[end]
                end
                
                branch_tree_lvl = branch_node.tree_lvl + 1
                for frac_var in frac_yᵣ_indx
                    #### create 2 branches 
                    ### Less than 0 inequality
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
            end
        else
            println(branch_node.name, "-->Model was infeasible")
        end
        println("Best Upper Bound=", best_ub, " and Best Lower Bound=", best_lb)
        run_cnt += 1
    end

    return best_solution[end]

end


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
    for i=1:length(yVal_cg)
        if yVal_cg[i] >= 1e-5
            push!(ind_CG_routes,i)
            push!(routes_cg, paths[i])
        end
    end

    println("Paths choosen ", ind_CG_routes, " with y values of ", yVal_cg[ind_CG_routes] )
    println("Numbe of CG Iterations: ", length(m_obj))

    route_nodes = []
    for r in routes_cg
        node_list = []
        for loc in r
            push!(node_list, copy(df.NODE[loc]))
        end
        push!(route_nodes, copy(node_list))
    end

    fig_cg = Plots.scatter(df[df.TYPE .== "DEPOT",:X], df[df.TYPE .== "DEPOT",:Y], label="Depot", aspect_ratio = 1, legend = :outertopright, xticks=:true, yticks=:true)
    Plots.scatter!(fig_cg, df[df.TYPE .== "PICKUP",:X], df[df.TYPE .== "PICKUP",:Y], label="Pickup")
    Plots.scatter!(fig_cg, df[df.TYPE .== "DELIVERY",:X], df[df.TYPE .== "DELIVERY",:Y], label="Delivery")

    for k in 1:size(routes_cg,1)
        p_x = df[routes_cg[k], :X]
        p_y = df[routes_cg[k], :Y]
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
        write(io, "Number Vehicles: " * string(input_list[3]) * "\n")
        write(io, "Capacity: " * string(input_list[4]) * "\n")
        write(io, "Number of Customers: " * string(input_list[2]) * "\n")
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

    Plots.savefig(img[1], main_folder * "/" * input_list[1] * "/Customer_loc.png")
    PlotlyJS.savefig(img[2], main_folder * "/" * input_list[1] * "/Customer_loc_w_nodes.png")
    Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/Customer_routes_cg.png")
    Plots.savefig(fig_rc, main_folder * "/" * input_list[1] * "/fig_rc.png")
    Plots.savefig(fig_obj, main_folder * "/" * input_list[1] * "/fig_obj.png")


end


################################################################
################### Run the MILP ###############################
################################################################
input_list = ["lc101", 53, 25, 200.0]
data, df, img = PreProcess(input_list[1] * ".csv", input_list[2] , input_list[3] , input_list[4] , 2)
col2return = 100
neighbors = [5,10,20]
U_limit = 1000

tic = time()
best_solution = branch_and_price(data, col2return, neighbors, U_limit)
toc = time()

solution_time = toc-tic
println("Initial Solution Time(s): ", solution_time)

results = Results(best_solution, solution_time)
print_results_to_file(results, df)

