using LinearAlgebra
using StatsBase
using JLD2
using Dates
using Random
using DifferentialEquations
include("Utils.jl")
using .Utils
include("DMFT.jl")
using .DMFT
BLAS.set_num_threads(1)
# This should take ~10 seconds with 10 threads
N = 2000
g_eff = 3.0
J0_positive = -5.0 / N
τchn_vec_positive = [0.0, 0.5, 1.0] ./ N
J0_negative = 0.0
τchn_vec_negative = [-8.0, -4.0, 0.0] ./ N
T = 7.0

dt = 0.02
dt_vec_positive = fill(dt, length(τchn_vec_positive))
dt_vec_negative = [0.005, 0.01, 0.02]
μ_init = 1.5
σ_init = 0.2
nJ_sim = 5
sim_dt = 0.1
sim_tspan = (0.0, T)
sim_tgrid = collect(sim_tspan[1]:sim_dt:sim_tspan[2])

Random.seed!(123)

function normalized_initial_condition(N::Int, μ::Float64, σ::Float64)
    x0 = randn(N)
    x0 .-= mean(x0)
    x0 ./= std(x0)
    x0 .= σ .* x0 .+ μ
    return x0
end

function compute_simulated_trajectory(J0::Float64, g::Float64, τ;
    N::Int=N,
    μ0::Float64=μ_init,
    σ0::Float64=σ_init,
    tspan::Tuple{Float64, Float64}=sim_tspan,
    tgrid::Vector{Float64}=sim_tgrid,
    n_realizations::Int=nJ_sim)

    J_array = Matrix{Float64}[]
    sizehint!(J_array, n_realizations)
    for _ in 1:n_realizations
        J = CreateJ(N, J0, g, τ; parallel=true, verbose=false)
        push!(J_array, J)
    end

    x0_array = [normalized_initial_condition(N, μ0, σ0) for _ in 1:n_realizations]
    prob = ODEProblem(GaussianTanh!, copy(x0_array[1]), tspan, J_array[1])

    function prob_func(prob, i, repeat)
        remake(prob; u0=copy(x0_array[i]), p=J_array[i])
    end

    function output_func(sol, i)
        vec(mean(Array(sol); dims=1)), false
    end

    ensemble_prob = EnsembleProblem(prob;
        prob_func=prob_func,
        output_func=output_func)

    sim = solve(ensemble_prob, Tsit5(), EnsembleThreads();
        trajectories=n_realizations,
        saveat=tgrid,
        save_everystep=false)

    mx_by_realization = reduce(hcat, sim.u)
    mx_mean = vec(mean(mx_by_realization; dims=2))
    mx_std = vec(std(mx_by_realization; dims=2))

    return (
        times=tgrid,
        mx_mean=mx_mean,
        mx_std=mx_std,
        x0_mean=vec(mean(reduce(hcat, x0_array); dims=1)),
        x0_std=vec(std(reduce(hcat, x0_array); dims=1)),
        n_realizations=n_realizations,
    )
end

function compute_wrapper(τchn::Float64, J0::Float64, dmft_dt::Float64; chain_label::String)
    start_time = now()

    τrec = 2.0 * τchn
    τ = (τchn, τrec)
    g = compute_g(g_eff, τ)

    mx, _, _, _, χϕ = DMFT_Nonstationary_Solver_B0(N, J0, g, τ;
        Ttot=T,
        dt=dmft_dt,
        traj_init_μx=μ_init,
        traj_init_σx=σ_init)

    dmft_times = collect(0:(length(mx)-1)) .* dmft_dt
    @assert size(χϕ) == (length(mx), length(mx))

    DMFTResult = (
        times=dmft_times,
        mx=mx,
        χϕ=χϕ,
        dt=dmft_dt
    )

    SimulationResult = compute_simulated_trajectory(J0, g, τ)

    println("Finished $chain_label chain Nτchn = $(N*τchn). Elapsed = $(format_elapsed(start_time))")
    flush(stdout)

    return (
        chain_label=chain_label,
        τchn=τchn,
        τrec=τrec,
        τ=τ,
        J0=J0,
        g=g,
        DMFTResult=DMFTResult,
        SimulationResult=SimulationResult
    )
end

positive_results = []
for (τchn, dmft_dt) in zip(τchn_vec_positive, dt_vec_positive)
    push!(positive_results, compute_wrapper(Float64(τchn), J0_positive, Float64(dmft_dt); chain_label="positive"))
end

negative_results = []
for (τchn, dmft_dt) in zip(τchn_vec_negative, dt_vec_negative)
    push!(negative_results, compute_wrapper(Float64(τchn), J0_negative, Float64(dmft_dt); chain_label="negative"))
end

dmft_params = (
    dt_vec_positive=dt_vec_positive,
    dt_vec_negative=dt_vec_negative,
    μ_init=μ_init,
    σ_init=σ_init
)

file_name = joinpath(@__DIR__, "results/TransientDMFTVSSimulation.jld2")
jldsave(file_name; dmft_params, N, g_eff, J0_positive, J0_negative, T, nJ_sim, sim_dt,
     τchn_vec_positive, τchn_vec_negative, dt_vec_positive, dt_vec_negative,
     positive_results, negative_results)
