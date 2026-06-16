using Distributed
using SlurmClusterManager
using JLD2
using LinearAlgebra
using Dates
using Random
using DSP
using Trapz
using DifferentialEquations
using QuadGK

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

# Set number of threads on each worker to --cpus-per-task
addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "24"])
println("Total workers: $(nworkers())")
flush(stdout)
@everywhere begin
    include($UTILS_PATH)
end
@everywhere begin
    using .Utils
    using LinearAlgebra
    BLAS.set_num_threads(1)
end

N = 2000
J0 = 4.0/N
num_points = 96
g_vec = LinRange(0.0, 4.0, num_points)
τ_chn_vec = LinRange(0.0, -3.0/N, num_points)
τ_rec = 0.0
phase_matrix = Array{Union{Int, Missing}}(undef, num_points, num_points)

@everywhere function classify_point(τ, g, J0, N)
    try
        return PhaseClassifier(τ, g, J0, N; n_samples = 24, tspan=(0.0, 200.0), saveat=0.5, burn_in=40.0)
    catch e
        @warn "PhaseClassifier failed at (g=$g, τchn=$(τ[1])): $e"
        return -99
    end
end

start = now()

for (i, g) in enumerate(g_vec)
    params_row = [((τc, τ_rec), g, J0, N) for τc in τ_chn_vec]

    results = pmap(params_row) do p
        τ, g, J0, N = p
        classify_point(τ, g, J0, N)
    end

    phase_matrix[i, :] .= results

    elapsed = now() - start
    println("Row $i / $num_points finished. Elapsed = $(round(Dates.value(elapsed)/1e3, digits=1)) seconds")
    flush(stdout)
end

file_name = joinpath(@__DIR__ ,"results/NegtauChn_g_PhaseDiagram.jld2")
jldsave(file_name;
    phase_matrix, τ_chn_vec, τ_rec, g_vec, N, J0)
