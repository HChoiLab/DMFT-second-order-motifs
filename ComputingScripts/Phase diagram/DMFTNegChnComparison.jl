using JLD2
using Dates
using Random
using LinearAlgebra
using CUDA
include("Utils.jl")
include("DMFT.jl")

using .Utils
using .DMFT_CUDA

BLAS.set_num_threads(1)
CUDA.allowscalar(false)
# This should take about 50 mins

N = 2000
g_eff = 3.0
J0 = 0.0
τchn_vec = [0.0, -0.075, -0.15]

T = 150.0
nIter_list = (30, 30, 20, 20)
nTraj_list = (4096, 8192, 16384, 16384)
damp_R = (0.05, 0.2, 0.4, 0.8)
damp_C = (0.05, 0.2, 0.4, 0.8)
dmft_dt = 0.05

function compute_wrapper(τchn::Float64)
    # Log start directly to stdout
    println("Starting: τ_chn = $τchn")
    flush(stdout)
    start_time = now()
    
    τ = (τchn, 0.0)
    g = compute_g(g_eff, τ)
    
    times, _, _, _, Cϕ_mean, Cϕ_std = NumericalAutocorrelation(N, J0, g, τ;
        T=1000.0, n_samples=36, burn=100.0, exclude_bimodal=false, subtract_mean=false)
        
    NumericalResult = (times = times, Cϕ_mean = Cϕ_mean, Cϕ_std = Cϕ_std)
    
    println("τ_chn = $τchn numerical result finished.")
    flush(stdout)
    
    model = CreateStationaryDMFTRateModel(N, J0, g, τ, T,
        nIter_list, nTraj_list, damp_R, damp_C; dt=dmft_dt)
        
    Cϕ, χϕ = DMFTStationaryMainloop(model; verbose = false, return_history=false, minimal_return=true)

    DMFTResult = (times = collect(0:dmft_dt:T-dmft_dt), Cϕ = Cϕ, χϕ = χϕ)
    
    elapsed = now() - start_time
    dur = round(Dates.value(elapsed)/1e3, digits=1)
    println("Finished τ_chn = $τchn. Elapsed = $(dur)s\n")
    flush(stdout)
    
    return (NumericalResult = NumericalResult, DMFTResult = DMFTResult)
end

results = []
for τchn in τchn_vec
    push!(results, compute_wrapper(τchn))
end

output_filename = joinpath(@__DIR__, "results/DMFTChaosCompNegChn.jld2")

jldsave(output_filename; 
    results = results,
    g_eff = g_eff,
    N = N,
    J0 = J0,
    τchn_vec = τchn_vec,
    damp_C = damp_C,
    damp_R = damp_R,
    dmft_T = T,
    dmft_dt = dmft_dt,
    nIter_list = nIter_list,
    nTraj_list = nTraj_list
)
