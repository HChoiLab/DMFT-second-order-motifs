using JLD2
using LinearAlgebra, Dates
using CUDA
include("Utils.jl")
include("DMFT.jl")
using .DMFT_CUDA
using .Utils

BLAS.set_num_threads(1)
CUDA.allowscalar(false)

N = 2000
g_eff_vec = [2.0, 4.0]
num_points = 10
dt_vec = [0.01, 0.02, 0.025, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
T_vec = LinRange(40.0, 640.0, num_points)
nIter_list = (100, 50, 50, 20)
nTraj_list = (4096, 8192, 16384, 16384)
damp_R = (0.05, 0.2, 0.4, 0.8)
damp_C = (0.05, 0.2, 0.4, 0.8)
τ_rec_vec = LinRange(-0.5, 0.5, num_points)

const ResultEntry = NamedTuple{(:Cϕ, :χϕ), Tuple{Vector{Float64}, Vector{Float64}}}
result_rec = Matrix{ResultEntry}(undef, length(g_eff_vec), length(τ_rec_vec))

for (i, g_eff) in enumerate(g_eff_vec)
    for (j, τrec) in enumerate(τ_rec_vec)
        start = now()
        println("Computing g_eff = $g_eff, τ_rec = $τrec")
        flush(stdout)
        τ = (0.0, τrec)
        g = compute_g(g_eff, τ)
        model = CreateStationaryDMFTRateModel(N, 0.0, g, τ, T_vec[j],
            nIter_list, nTraj_list, damp_R, damp_C; dt=dt_vec[j])
        Cϕ, χϕ = DMFTStationaryMainloop(model; verbose = false, return_history=false, minimal_return=true)
        result_rec[i, j] = (Cϕ=Cϕ, χϕ=χϕ)
        println("g_eff = $g_eff, τ_rec = $τrec done. Elapsed = $(format_elapsed(start))")
        flush(stdout)
    end
end

file_name = joinpath(@__DIR__, "results/TheoreticalPRDrec.jld2")

jldsave(file_name;
    N, g_eff_vec, dt_vec, T_vec, τ_rec_vec,
    result_rec
)
