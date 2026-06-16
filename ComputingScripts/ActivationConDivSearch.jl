using JLD2, Dates, Distributed, SlurmClusterManager

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

# One worker per τc value, each worker gets 48 threads for EnsembleThreads
# This should take < 2 hours if using 4 workers
addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "64"])
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include($UTILS_PATH)
end
@everywhere begin
    using .Utils
    using LinearAlgebra
    using DifferentialEquations
    using StatsBase
    using Random
    BLAS.set_num_threads(1)
    Random.seed!(123)
end

@everywhere begin
    const N = 2000
    const g = 5.0
    const J0 = -5 / N

    const t_warmup = 100.0
    const T = 1000.0
    const saveat = 0.5
    const num_points = 12
    const num_repeats = 128
    const τc_vec = range(0.1/N, 0.4/N, 4)
    const tspan = (0.0, T + t_warmup)
end

@everywhere function temporal_variance_fraction(X::Matrix{Float64})
    temporal_variance = mean(var(X, dims = 2, corrected = false))
    spatial_variance = var(vec(mean(X, dims = 2)), corrected = false)
    total_variance = temporal_variance + spatial_variance
    return temporal_variance / total_variance
end

@everywhere function compute_for_τc(τc::Float64, ODEFunc!::Function)
    τdiv_vec = logrange(sqrt(0.1 * τc^2), sqrt(10.0 * τc^2), num_points)
    τcon_vec = τc^2 ./ τdiv_vec
    ratio_row  = τdiv_vec .^ 2 ./ τc^2
    qmean_row  = zeros(num_points)
    qstd_row = zeros(num_points)

    for jj in eachindex(τcon_vec)
        J_array = [CreateJ(N, J0, g, (τc, 0.0, τcon_vec[jj], τdiv_vec[jj]);
                           parallel = true)
                   for _ in 1:num_repeats]
        
        if J_array[1] === nothing
            # The motif combination is invalid
            qmean_row[jj] = NaN
            qstd_row[jj] = NaN
            continue
        end
        prob = ODEProblem(ODEFunc!, zeros(N), tspan, J_array[1])

        function prob_func(prob, i, repeat)
            prob.u0 .= randn(N)
            prob.p .= J_array[i]
            return prob
        end

        function output_func(sol, i)
            X = Array(sol)
            X = X[:, sol.t .> t_warmup]
            return temporal_variance_fraction(X), false
        end

        ensemble_prob = EnsembleProblem(prob;
                                        prob_func = prob_func,
                                        output_func = output_func, 
                                        )

        sim = solve(ensemble_prob, Tsit5();
                    trajectories = num_repeats, saveat = saveat)

        var_temp = sim.u
        qmean_row[jj] = mean(var_temp)
        qstd_row[jj] = std(var_temp)
    end

    return (ratio_row, qmean_row, qstd_row)
end

activation_names = ["tanh", "nonnegative"]
activation_functions = [GaussianTanh!, GaussianPositiveActivation!]

ratio = zeros(length(τc_vec), num_points)
qmean = zeros(length(activation_names), length(τc_vec), num_points)
qstd  = zeros(length(activation_names), length(τc_vec), num_points)

start = now()

for (aa, (activation_name, activation_function)) in enumerate(zip(activation_names, activation_functions))
    println("Computing activation: $activation_name")
    flush(stdout)

    results = pmap(τc_vec) do τc
        compute_for_τc(τc, activation_function)
    end

    for (ii, (ratio_row, qmean_row, qstd_row)) in enumerate(results)
        ratio[ii, :] = ratio_row
        qmean[aa, ii, :] = qmean_row
        qstd[aa, ii, :]  = qstd_row
    end
end

println("Finished. Time elapsed: $(format_elapsed(start))")
flush(stdout)

jldsave("./results/ActivationConDivSearch.jld2"; N, g, J0, T, saveat, qmean, qstd, num_repeats,
        ratio, τc_vec, tspan, t_warmup, activation_names)
