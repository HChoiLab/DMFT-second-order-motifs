using Distributed
using SlurmClusterManager
using JLD2, Dates
const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")
const DMFT_PATH = joinpath(@__DIR__, "DMFT.jl")
addprocs(SlurmManager(), exeflags=["--project=MotifNets", "-t", "16"])
println("Total workers: $(length(workers()))")
flush(stdout)
# Using 32 workers with 12 threads, this should take around 5 hours 

@everywhere begin
    include($UTILS_PATH)
    include($DMFT_PATH)
end

@everywhere begin
    using LinearAlgebra, Dates
    using .Utils
    using .DMFT

    BLAS.set_num_threads(1)

    const N = 2000
    const J0 = 4.0 / N
    const Nτchn_vec = range(-1.0, -3.0, 64)
    const g_vec = range(2 / sqrt(3), 2.7, 64)
    const T = 200.0
    const dt = 0.05
    const burn_in = 150.0
    const μ_init1 = 1e-4
    const σ_init1 = 0.5
    const μ_init2 = 0.5
    const σ_init2 = 0.5
end

const 📫 = RemoteChannel(() -> Channel{String}(1000))
@async begin
    while true
        ✉️ = take!(📫)
        println(✉️)
        flush(stdout)
    end
end

@everywhere function compute_Cϕ_lc_chaos(
    N::Int64,
    J0::Float64,
    g::Float64,
    τ::Tuple{Vararg{Float64}},
    μ_init::Float64,
    σ_init::Float64,
)
    global T, dt, burn_in

    if (N * J0)^2 + 4 * g^2 * N * τ[1] >= 0
        return nothing
    else
        _, _, _, Cϕ, _ = DMFT_Nonstationary_Solver_B0(
            N, J0, g, τ;
            dt=dt,
            Ttot=T,
            traj_init_μx=μ_init,
            traj_init_σx=σ_init,
        )
        start_idx = floor(Int, burn_in / dt) + 1
        Cϕ = Cϕ[start_idx, start_idx:end]
        return Cϕ
    end
end

@everywhere function build_lc_chaos_tasks()
    global N, Nτchn_vec, g_vec

    tasks = []
    for i in eachindex(Nτchn_vec)
        Nτchn = Float64(Nτchn_vec[i])
        for j in eachindex(g_vec)
            g = Float64(g_vec[j])
            push!(tasks, (i=i, j=j, Nτchn=Nτchn, g=g))
        end
    end

    return tasks
end

@everywhere function compute_Cϕ_lc_chaos_task(task, log_channel::RemoteChannel)
    global N, J0, μ_init1, σ_init1, μ_init2, σ_init2
    start = now()
    τchn = task.Nτchn / N
    τ = (τchn, 2 * τchn)

    Cϕ1 = compute_Cϕ_lc_chaos(N, J0, task.g, τ, μ_init1, σ_init1)
    Cϕ2 = compute_Cϕ_lc_chaos(N, J0, task.g, τ, μ_init2, σ_init2)

    put!(log_channel, "Worker $(myid()) finished N*τchn=$(round(task.Nτchn, digits=4)), " *
        "g=$(round(task.g, digits=6)) ($(task.i),$(task.j))/($(length(Nτchn_vec)),$(length(g_vec))). " *
        "Elapsed = $(format_elapsed(start))")

    return (i=task.i, j=task.j, Cϕ1=Cϕ1, Cϕ2=Cϕ2)
end

function compute_Cϕ_lc_chaos_grid()
    tasks = build_lc_chaos_tasks()

    results = pmap(tasks; batch_size=1) do task
        compute_Cϕ_lc_chaos_task(task, 📫)
    end

    Cϕ1_grid = Matrix{Union{Nothing,Vector{Float64}}}(nothing, length(Nτchn_vec), length(g_vec))
    Cϕ2_grid = Matrix{Union{Nothing,Vector{Float64}}}(nothing, length(Nτchn_vec), length(g_vec))
    for result in results
        Cϕ1_grid[result.i, result.j] = result.Cϕ1
        Cϕ2_grid[result.i, result.j] = result.Cϕ2
    end

    return Cϕ1_grid, Cϕ2_grid
end

Cϕ1_grid, Cϕ2_grid = compute_Cϕ_lc_chaos_grid()
close(📫)

file_name = "NegChnLCShadingN$(N)J0$(replace(string(round(N * J0, digits=1)), "." => "p"))B0.jld2"

output_file = joinpath(@__DIR__, "results", file_name)
jldsave(output_file;
    N, J0, Nτchn_vec, g_vec, T, dt,
    burn_in, μ_init1, σ_init1, μ_init2, σ_init2, Cϕ1_grid, Cϕ2_grid)
