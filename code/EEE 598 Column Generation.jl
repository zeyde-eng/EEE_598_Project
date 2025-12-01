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

# RMP structure
mutable struct RMP
    model::Model
    y::Vector{VariableRef}
    A::Array{Float64}
    c_r::Vector{Float64}
    sp_constr
    Ω::Int64
end

mutable struct Sub
    model::Model
    x::Matrix{VariableRef}
    B::Array{VariableRef}
    Q::Array{VariableRef}
end

# Initialize Data
function Data(num_Cust, xcoord, ycoord, demand, max_V, a, b, s, q; obj_func=1)
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
    display(fig)


    fig_plotly = PlotlyJS.plot(
            A_final, x=:X, y=:Y, color=:TYPE, text=:row_num,
            mode="markers+text", marker=PlotlyJS.attr(size=12, line=PlotlyJS.attr(width=2, color="DarkSlateGrey")),
            textposition="top center",
    )
    display(fig_plotly)

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
    model = Model(Gurobi.Optimizer)
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

    return RMP(model, y, A, c_r, sp_constr, Ω)

end


function Sub(data::Data)

    n = data.n
    d = data.d
    N = data.N
    c = data.c
    t = data.t
    q = data.q
    a = data.a
    b = data.b
    s = data.s
    bigM = data.bigM
    bigW = data.bigW
    timeout = 5.0 # timeout in minutes

    # define dummy dual variable 
    π̂ = zeros(Float64, n)

    ĉ = deepcopy(c)
    ĉ[2:n+1,:] .-= π̂

    # define empty model
    model = Model(Gurobi.Optimizer)
    set_attribute(model, "TimeLimit", timeout * 60)
    set_attribute(model, "LogToConsole", 0)

    # mute the output
    # set_silent(model)

    # add variables
    @variable(model, x[1:N, 1:N], Bin)
    @variable(model, B[1:N] >=0)
    @variable(model, Q[1:N] >=0)

    # add objective
    @objective(
        model, 
        Min, 
        sum(ĉ[i,j]*x[i,j] for i=1:N, j=1:N)
    )
    
    # add constraints
    @constraint(
        model, 
        c3[i=2:n+1], 
        sum(x[i,j] for j=1:N if j!=1 && i!=j) - sum(x[n+i,j] for j=1:N if j!=1 && (n+i)!=j) == 0
    )


    @constraint(
        model, 
        c4, 
        sum(x[1,j] for j=1:N if j!=1) == 1
    )


    @constraint(
        model, 
        c5[h=1:N; h!=1 && h!=N], 
        sum(x[i,h] for i=1:N if i!=h && i!=N ) - sum(x[h,j] for j=1:N if j!=h && j!=1) == 0
    )


    @constraint(
        model, 
        c6, 
        sum(x[i,N] for i=1:N if i!=N) == 1
    )


    @constraint(
        model, 
        c7[i=1:N, j=1:N; i!=j && i!=N && j!=1], 
        B[i] + s[i] + t[i,j] - B[j] - bigM[i,j]*(1 - x[i,j]) <= 0
    )



    @constraint(
        model, 
        c8[i=1:N, j=1:N; i!=j && i!=N && j!=1], 
        Q[i] + d[j] - Q[j] - bigW[i,j]*(1 - x[i,j]) <= 0
    )


    @constraint(
        model, 
        c9[i=2:n+1], 
        B[i] + t[i,n+i] - B[n+i] <= 0
    )


    @constraint(
        model, 
        c10a[i=1:N], 
        a[i] - B[i] <= 0
    )


    @constraint(
        model, 
        c10b[i=1:N],
        B[i] - b[i] <= 0
    )

    @constraint(
        model, 
        c11a[i=1:N],
        max(0,d[i]) - Q[i] <= 0
    )


    @constraint(
        model, 
        c11b[i=1:N],
        Q[i] - min(q, q+d[i]) <= 0
    )


    @constraint(
        model, 
        c12a, 
        B[1] == 0
    )

    @constraint(
        model, 
        c12b, 
        Q[1] == 0
    )

    @constraint(
        model, 
        c13[i=1:N, j=1:N; i==j], 
        x[i,j] == 0
    )

    @constraint(
        model, 
        c14[i=1:N], 
        x[i,1] == 0
    )

    return Sub(model, x, B, Q)
end


function runCG(master::RMP, sub::Sub, data::Data)
    
    n = data.n
    c = data.c
    N = data.N
    B̂ = []
    Q̂ = []

    for j = 1:n
        B_init = zeros(Float64,N)
        B_init[j+1] = 1
        B_init[n+j+1] = 2
        B_init[N] = 3
        push!(B̂,B_init)
    end


    m_obj = []
    r_cost = []


    while true 
        # solve the restricted master problem
        optimize!(master.model)
        #println(value.(master.y))
        println("Master Problem Objective Value = ", objective_value(master.model))
        push!(m_obj, objective_value(master.model))

        # get duals
        π = dual.(master.sp_constr)

        ĉ = deepcopy(c)
        ĉ[2:n+1,:] .-= π

        
        # update the subproblem objective function using the dual info
        set_objective_coefficient.(sub.model, sub.x, ĉ)
        
        # solve the subproblem
        optimize!(sub.model)

        # obtain the optimal value of the sub, which corresponds to the reduced cost of the column generated by the sub
        reduced_cost = objective_value(sub.model)
        println(reduced_cost)
        push!(r_cost, reduced_cost)

        push!(B̂,value.(sub.B))
        push!(Q̂,value.(sub.Q))

        if reduced_cost < -1e-8 
            # add the column with negative reduced cost
            x_val = value.(sub.x)
            A_new = sum(x_val, dims=2)[2:n+1]
            
            c_r_new = 0
            for m in 1:N
                for n in 1:N
                    c_r_new += c[m,n].*x_val[m,n]
                end
            end

            master.A = hcat(master.A,A_new)
            push!(master.y, @variable(master.model,  lower_bound = 0.0))
            set_objective_coefficient.(master.model, master.y[end], c_r_new)
            set_normalized_coefficient.(master.sp_constr, master.y[end], A_new)

        else
            break
        end
    end

    return m_obj, r_cost, B̂, Q̂
end



################################################################
################### Run the MILP ###############################
################################################################
input_list = ["lc104", 53, 25, 200.0]
method = "mip" #options are "mip" or "labeling"

data, df, img = PreProcess(input_list[1] * ".csv", input_list[2] , input_list[3] , input_list[4] , 2)

# generate master 
master = RMP(data)
sub = Sub(data)

tic = time()
m_obj, r_cost, B̂, Q̂ = runCG(master, sub, data)
toc = time()

solution_time = toc-tic

################################################################
################### Plot the GC Result #######################
################################################################


fig_obj = Plots.plot(1:length(m_obj), m_obj, label="Master Problem Objective", lw=2)
display(fig_obj)

fig_rc = Plots.plot(1:length(r_cost), r_cost, label="Reduced Cost", lw=2)
display(fig_rc)

ind_CG_routes = []

yVal_cg = value.(master.y)
for i=1:length(yVal_cg)
    if yVal_cg[i] >= 1e-5
        push!(ind_CG_routes,i)
    end
end

#println("Paths choosen ", ind_CG_routes, " with y values of ", yVal_cg[ind_CG_routes] )

routes_cg = []
B̂_new = []

Nverts = data.N

n = size(master.A,1)

for j in ind_CG_routes
    route_cg = [1]
    B̂_temp = [0.0]
    for i=1:n
        if master.A[i,j] != 0
            push!(route_cg,copy(i+1))
            push!(route_cg,copy(n+i+1))
            push!(B̂_temp, copy(B̂[j][i+1]))
            push!(B̂_temp, copy(B̂[j][n+i+1]))
        end
    end
    push!(route_cg, copy(Nverts))
    push!(B̂_temp, copy(B̂[j][Nverts]))

    # reorder route_cg based on ŝ_temp 
    perm = sortperm(B̂_temp)
    
    push!(routes_cg, copy(route_cg[perm]))
    push!(B̂_new, copy(B̂_temp[perm]))
end

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
display(fig_cg)


################################################################
################### Save the Result #######################
################################################################

main_folder = "Results_CG"

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
end

Plots.savefig(img[1], main_folder * "/" * input_list[1] * "/Customer_loc.png")
PlotlyJS.savefig(img[2], main_folder * "/" * input_list[1] * "/Customer_loc_w_nodes.png")
Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/Customer_routes_cg.png")
Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/fig_rc.png")
Plots.savefig(fig_cg, main_folder * "/" * input_list[1] * "/fig_obj.png")
