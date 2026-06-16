using LinearAlgebra
using Dates
using JLD2

const DMFT_PATH = joinpath(@__DIR__, "DMFT.jl")
const RESULTS_DIR = joinpath(@__DIR__, "results")
include(DMFT_PATH)
using .DMFT
BLAS.set_num_threads(4)

N = 2000
J0 = 4.0 / N
num_points = 16
Nτchnvec = range(-1.0, 0.0, num_points)

g_step = 0.0025
g_max = 3.0
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

function find_nonzero_fp_bulk_g_boundary(τchn_vec, N::Int64, J0::Float64;
    g_init::Float64=2.0, # This is the critical point for J0 = 4.0
    g_step::Float64=0.005,
    g_max::Float64=3.0,
    tol::Float64=1e-3,
    n_theta::Int64=100)

    g_boundary = fill(NaN, length(τchn_vec))
    start = now()

    if !isempty(τchn_vec)
        g_boundary[1] = g_init # We manually set this point
        println("Set N*τchn = $(round(N * Float64(τchn_vec[1]), digits=4)) (1/$(length(τchn_vec))); " *
            "g = $(round(g_boundary[1], digits=6)); " *
            "elapsed = $(DMFT.Utils.format_elapsed(start))")
        flush(stdout)
    else
        return g_boundary
    end

    next_g_init = g_boundary[1]

    for i in Iterators.drop(eachindex(τchn_vec), 1)
        τchn = Float64(τchn_vec[i])
        g = next_g_init
        delta = NaN

        while g <= g_max
            delta = boundary_delta(τchn, g, N, J0; n_theta=n_theta)

            if !isnan(delta) && (abs(delta) < tol || delta > 0.0)
                g_boundary[i] = g
                next_g_init = g
                break
            end

            g += g_step
        end

        println("Finished N*τchn = $(round(N * τchn, digits=4)) ($(i)/$(length(τchn_vec))); " *
            "g = $(round(g_boundary[i], digits=6)); " *
            "delta = $(round(delta, digits=6)); " *
            "elapsed = $(DMFT.Utils.format_elapsed(start))")
        flush(stdout)
    end

    return g_boundary
end

τchn_vec = collect(Nτchnvec) ./ N

g_boundary = find_nonzero_fp_bulk_g_boundary(τchn_vec, N, J0;
    g_init=2.0,
    g_step=g_step,
    g_max=g_max,
    tol=1e-3,
    n_theta=n_theta)

file_name = "NegChnSCFPBoundaryN$(N)J0$(DMFT.Utils.parameter_token(N*J0))B0.jld2"
output_file = joinpath(RESULTS_DIR, file_name)
jldsave(output_file;
    N, J0, num_points, Nτchnvec, τchn_vec, g_step, g_max, n_theta,
    g_boundary)
