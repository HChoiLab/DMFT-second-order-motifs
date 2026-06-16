using JLD2
using LinearAlgebra, Dates
using CUDA
include("Utils.jl")
include("DMFT.jl")
using .DMFT_CUDA
using .Utils

BLAS.set_num_threads(1)
CUDA.allowscalar(false)
# This should take a few hours
N = 2000
g_eff_vec = [2.0, 4.0]
num_points = 10
dt = 0.06
T = 240.0
nIter_list = (20, 10, 20, 40)
damp_R = (0.2, 0.3, 0.5, 0.9)
damp_C = (0.2, 0.3, 0.5, 0.9)
τ_chn_vec = range(-20.0/N, 0.0/N, num_points)
const ResultEntry = NamedTuple{(:Cϕ, :χϕ), Tuple{Vector{Float64}, Vector{Float64}}}
result_chn = Matrix{ResultEntry}(undef, length(g_eff_vec), length(τ_chn_vec))

for (i, g_eff) in enumerate(g_eff_vec)
    for (j, τchn) in enumerate(τ_chn_vec)
        start = now()
        println("Computing g_eff = $(round(g_eff, digits=4)), N*τ_chn = $(round(N*τchn, digits=4))")
        flush(stdout)
        τ = (τchn, 0.0)
        g = compute_g(g_eff, τ)
        if g_eff == 2.0 && N*τchn > -8.0
            remove_χϕ_ft_spike = true # For g_eff=2.0 and small |τchn|, Novikov's theorem is noisy
            nTraj_list = (12800, 12800, 32000, 51200)
        else
            remove_χϕ_ft_spike = false
            nTraj_list = (8192, 8192, 12800, 16800)
        end
        model = CreateStationaryDMFTRateModel(N, 0.0, g, τ, T,
            nIter_list, nTraj_list, damp_R, damp_C; dt=dt, remove_χϕ_ft_spike=remove_χϕ_ft_spike)
        Cϕ, χϕ = DMFTStationaryMainloop(model; verbose=false, return_history=false, minimal_return=true)
        result_chn[i, j] = (Cϕ=Cϕ, χϕ=χϕ)
        println("g_eff = $(round(g_eff, digits=4)), N*τ_chn = $(round(N*τchn, digits=4)) done. Elapsed = $(format_elapsed(start))")
        flush(stdout)
    end
end

file_name = joinpath(@__DIR__, "results/TheoreticalPRDchn.jld2")

jldsave(file_name;
    N, g_eff_vec, dt, T, τ_chn_vec,
    result_chn
)
