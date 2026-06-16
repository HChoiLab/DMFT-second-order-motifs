using JLD2, Dates
using LinearAlgebra, FFTW
include("Utils.jl")
include("DMFT.jl")
using .Utils
using .DMFT

BLAS.set_num_threads(1)
FFTW.set_num_threads(16)


## For positive chain motif
const N = 2000

const g = 3.0
const J0 = -5.0/N
const τchn_vec = [2.0, 4.0, 8.0] ./ N

const T = 200.0
const dmft_dt = 0.1
const dmft_max_iter = 500
const dmft_damp = 0.8
const dmft_tol = 5e-5
const dmft_μ0 = 0.5

function compute_wrapper(τchn::Float64)
    # Log start directly to stdout
    println("Starting: τ_chn = $τchn")
    flush(stdout)
    start_time = now()
    
    τ = (τchn, 2*τchn)
    
    times, _, _, _, Cϕ_mean, Cϕ_std = NumericalAutocorrelation(N, J0, g, τ;
        T=1000.0, n_samples=36, burn=100.0, exclude_bimodal=false, subtract_mean=false)

    NumericalResult = (times = times, Cϕ_mean = Cϕ_mean, Cϕ_std = Cϕ_std)

    println("τ_chn = $τchn numerical result finished.")
    flush(stdout)

    sol = DMFT_Stationary_Solver_B0(N, J0, 
        g, τ; μ0=dmft_μ0,
        dt=dmft_dt,
        T=T, max_iter=dmft_max_iter,
        tol=dmft_tol, damp=dmft_damp,
        verbose=false)

    DMFTResult = (
        times = sol.t,
        Cϕ = sol.Cϕ,
        χϕ = sol.χϕ
    )
    
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

output_filename = joinpath(@__DIR__, "results/DMFTB0CompPositivetauChn.jld2")

jldsave(output_filename; 
    results = results,
    g = g,
    N = N,
    J0 = J0,
    τchn_vec = τchn_vec,
    dmft_T = T,
    dmft_dt = dmft_dt,
    dmft_max_iter = dmft_max_iter,
    dmft_damp = dmft_damp,
    dmft_tol = dmft_tol
)
