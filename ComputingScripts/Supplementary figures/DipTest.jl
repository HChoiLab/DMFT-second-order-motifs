using LinearAlgebra, Distributed, SlurmClusterManager, JLD2, Dates
using DifferentialEquations, Random, StatsBase

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "8"])
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include($UTILS_PATH)
end

@everywhere begin
    using LinearAlgebra
    using .Utils
    using DifferentialEquations
    using Random
    using StatsBase
    BLAS.set_num_threads(1)
end

const N_vec = [1000, 2000, 4000]
const g_eff = 3.0
const nInits = 24
const num_points = 8
const τ_con_vec = range(0.0, 0.35, num_points)
const τ_div_vec = range(0.0, 0.35, num_points)
const J0 = 0.0
const n_samples = 200
const example_N = 2000
const example_tspan = (0.0, 120.0)
const example_saveat = 0.1
const scan_seed = 12345
const burn_in = 50.0
const example_trajectory_seed = 123

function scan_tau_axis!(non_unimodal_fraction, τ_vec, make_τ, axis_label, seed_axis_offset, start_time;
        collect_example=false)
    example_result = nothing
    base_seed = scan_seed
    j0 = J0
    n_inits = nInits
    test_burn_in = burn_in

    for (iN, N) in enumerate(N_vec)
        println("Starting $(axis_label) N $(iN)/$(length(N_vec)): N=$(N).")
        flush(stdout)

        for (iτ, τvalue) in enumerate(τ_vec)
            need_example = τvalue >= 0.20 && collect_example && N == example_N && example_result === nothing 
            τ = make_τ(τvalue)
            g = compute_g(g_eff, τ)
            sample_results = pmap(1:n_samples) do sample_idx
                Random.seed!(base_seed + seed_axis_offset + 1000000 * iN + 10000 * iτ + sample_idx)
                J = CreateJ(N, j0, g, τ; parallel=true)
                p = bimodal_test(J; nInits=n_inits, burn_in=test_burn_in, parallel=true)
                J_example = need_example && p < 0.05 ? J : nothing
                (p, J_example)
            end
            non_unimodal_fraction[iN, iτ] = count(result -> result[1] < 0.05, sample_results) / n_samples

            if collect_example && N == example_N && example_result === nothing
                passing_idx = findfirst(result -> result[2] !== nothing, sample_results)
                if passing_idx !== nothing
                    example_p, example_J = sample_results[passing_idx]
                    example_result = (passing_idx, example_p, example_J, τ)
                end
            end

            println("Finished $(axis_label) $(iτ)/$(length(τ_vec)) for N $(iN)/$(length(N_vec)). Elapsed: $(format_elapsed(start_time))")
            flush(stdout)
        end
    end

    return example_result
end

function summarize_example(example_result)
    example_sample_idx, example_p, example_J, example_τ = example_result
    example_eigenvalues = eigvals(example_J)

    prob_rng = MersenneTwister(example_trajectory_seed)
    prob = ODEProblem(GaussianTanh!, randn(prob_rng, example_N) .* 0.5, example_tspan, example_J)

    function prob_func(prob, i, repeat)
        rng = MersenneTwister(example_trajectory_seed + i)
        remake(prob; u0=randn(rng, example_N) .* 0.5)
    end

    function output_func(sol, i)
        X = Array(sol)
        mean_activity = vec(mean(X, dims=1))
        return mean_activity, false
    end

    ensemble_prob = EnsembleProblem(prob;
        prob_func=prob_func,
        output_func=output_func)

    sim = solve(ensemble_prob, Tsit5();
        trajectories=nInits, saveat=example_saveat)

    return (
        sample_idx=example_sample_idx,
        p=example_p,
        τ=example_τ,
        eigenvalues=example_eigenvalues,
        mean_activity=sim.u,
    )
end

function main()
    non_unimodal_fraction_con = zeros(Float64, length(N_vec), length(τ_con_vec))
    non_unimodal_fraction_div = zeros(Float64, length(N_vec), length(τ_div_vec))
    start_time = now()

    example_result_con = scan_tau_axis!(
        non_unimodal_fraction_con, τ_con_vec, τcon -> (0.0, 0.0, τcon, 0.0),
        "τcon", 0, start_time; collect_example=true)

    example_result_div = scan_tau_axis!(
        non_unimodal_fraction_div, τ_div_vec, τdiv -> (0.0, 0.0, 0.0, τdiv),
        "τdiv", 500000000, start_time; collect_example=true)

    example_result_con === nothing && error("No N=$(example_N) τcon network with p < 0.05 was found.")
    example_result_div === nothing && error("No N=$(example_N) τdiv network with p < 0.05 was found.")

    example_con = summarize_example(example_result_con)
    example_div = summarize_example(example_result_div)
    example_times = collect(example_tspan[1]:example_saveat:example_tspan[2])

    jldsave("results/DipTest.jld2";
        non_unimodal_fraction_con, non_unimodal_fraction_div,
        example_con, example_div,
        N_vec, τ_con_vec, τ_div_vec, g_eff, nInits, num_points, J0, n_samples,
        scan_seed, example_times, example_N,
        example_saveat, example_trajectory_seed, burn_in)
end

main()
