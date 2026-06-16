using LinearAlgebra
using Dates
using JLD2

const DMFT_PATH = joinpath(@__DIR__, "DMFT.jl")
const RESULTS_DIR = joinpath(@__DIR__, "results")

include(DMFT_PATH)
using .DMFT

BLAS.set_num_threads(1)
const 🔒 = ReentrantLock()

N = 2000
J0 = -5.0 / N
num_points = 20
g_vec1 = LinRange(1.1, 2.5, 10)
g_vec2 = LinRange(2.6, 4.0, 10)

τchn_step = 0.01 / N
τchn_min = 4.5/N
n_theta = 100

function boundary_delta(τchn::Float64, g::Float64, N::Int64, J0::Float64;
    n_theta::Int64=100)

    τ = (τchn, 0.0)
    xb = Float64[]
    try
        xb, _ = BoundaryCurve(N, J0, τ, g;
            n_theta=n_theta, boundary_method=:gaussian)
    catch e 
        println("g = $g, τchn = $τchn failed.")
        rethrow(e)
    end
    return maximum(xb) - 1.0
end

function find_nonzero_fp_bulk_τchn_boundary(g_vec, N::Int64, J0::Float64;
    τchn_init::Float64=10.0 / N,
    τchn_step::Float64=0.025 / N,
    τchn_min::Float64=0.0,
    tol::Float64=1e-3,
    n_theta::Int64=100)

    τchn_vec = fill(NaN, length(g_vec))
    start = now()

    Threads.@threads for i in eachindex(g_vec)
        g = g_vec[i]
        τchn = τchn_init
        delta = NaN

        while τchn >= τchn_min
            delta = boundary_delta(τchn, g, N, J0; n_theta=n_theta)

            if !isnan(delta) && (abs(delta) < tol || delta > 0.0)
                τchn_vec[i] = τchn
                break
            end

            τchn -= τchn_step
        end

        elapsed = now() - start
        lock(🔒)
        try
            println("Finished g = $(round(g, digits=6)) ($(i)/$(length(g_vec))); " *
                "N*τchn = $(round(N * τchn_vec[i], digits=4)); " *
                "delta = $(round(delta, digits=6)); " *
                "elapsed = $(round(Dates.value(elapsed) / 1e3, digits=1)) seconds")
            flush(stdout)
        finally
            unlock(🔒)
        end
    end

    return τchn_vec
end

τchn_boundary1 = find_nonzero_fp_bulk_τchn_boundary(g_vec1, N, J0;
    τchn_init=6.0/N,
    τchn_step=τchn_step,
    τchn_min=τchn_min,
    tol=1e-3,
    n_theta=n_theta)

τchn_boundary2 = find_nonzero_fp_bulk_τchn_boundary(g_vec2, N, J0;
    τchn_init=10.0/N,
    τchn_step=τchn_step,
    τchn_min=τchn_min,
    tol=1e-5,
    n_theta=n_theta)

g_vec = vcat(g_vec1, g_vec2)
τchn_boundary = vcat(τchn_boundary1, τchn_boundary2)

file_name = "PosChnSCFPBoundaryN$(N)J0$(parameter_token(N*J0))B0.jld2"
output_file = joinpath(RESULTS_DIR, file_name)
jldsave(output_file;
    N, J0, num_points, g_vec, τchn_step, n_theta,
    τchn_boundary)
