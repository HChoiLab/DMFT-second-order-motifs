using JLD2, Dates, Distributed, SlurmClusterManager

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

# Every (activation, τc, jj) point is an independent pmap job; each job still runs
# its num_repeats ensemble over the worker's threads (EnsembleThreads).
# This should take less than 4h if using 24 workers, 48 threads
addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "48"])
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
    using Distributed
    using Dates
    BLAS.set_num_threads(2)
    Random.seed!(123)
end

@everywhere begin
    const N = 8000
    const g = 5.0
    const J0 = -5 / N

    const t_warmup = 400.0
    const T = 4000.0
    const saveat = 1.0
    const num_points = 8
    const num_repeats = 24
    const τc_vec = range(0.1/N, 0.3/N, 3)
    const tspan = (0.0, T + t_warmup)

    const activation_names = ["tanh", "nonnegative"]
    const activation_functions = [GaussianTanh!, GaussianPositiveActivation!]
end

@everywhere function temporal_variance_fraction(X::Matrix{Float64})
    temporal_variance = mean(var(X, dims = 2, corrected = false))
    spatial_variance = var(vec(mean(X, dims = 2)), corrected = false)
    total_variance = temporal_variance + spatial_variance
    return temporal_variance / total_variance
end


@everywhere function compute_point(job, log_channel::RemoteChannel)
    ODEFunc! = activation_functions[job.aa]
    label = "$(activation_names[job.aa]), Nτc=$(round(N * job.τc, digits=3)), ratio=$(round(job.ratio, digits=3))"
    start_time = now()

    J_array = [CreateJ(N, J0, g, (job.τc, 0.0, job.τcon, job.τdiv); parallel = true)
               for _ in 1:num_repeats]

    if J_array[1] === nothing
        # The motif combination is invalid
        put!(log_channel, "Worker $(myid()) invalid motif: $label")
        return (aa = job.aa, ic = job.ic, jj = job.jj, qmean = NaN, qstd = NaN, ratio = job.ratio)
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
    put!(log_channel, "Worker $(myid()) finished: $label. Elapsed = $(format_elapsed(start_time))")
    return (aa = job.aa, ic = job.ic, jj = job.jj,
            qmean = mean(var_temp), qstd = std(var_temp), ratio = job.ratio)
end

const 📫 = RemoteChannel(() -> Channel{String}(1000))
@async begin
    while true
        println(take!(📫))
        flush(stdout)
    end
end

jobs = NamedTuple[]
for aa in eachindex(activation_functions)
    for (ic, τc) in enumerate(τc_vec)
        τdiv_vec = logrange(sqrt(0.1 * τc^2), sqrt(10.0 * τc^2), num_points)
        τcon_vec = τc^2 ./ τdiv_vec
        for jj in 1:num_points
            push!(jobs, (aa = aa, ic = ic, jj = jj, τc = Float64(τc),
                         τcon = τcon_vec[jj], τdiv = τdiv_vec[jj],
                         ratio = τdiv_vec[jj]^2 / τc^2))
        end
    end
end

println("Dispatching $(length(jobs)) points over $(nworkers()) workers.")
flush(stdout)

start = now()
results = pmap(job -> compute_point(job, 📫), jobs)
println("Finished. Time elapsed: $(format_elapsed(start))")
flush(stdout)

ratio = zeros(length(τc_vec), num_points)
qmean = zeros(length(activation_names), length(τc_vec), num_points)
qstd  = zeros(length(activation_names), length(τc_vec), num_points)

for res in results
    ratio[res.ic, res.jj] = res.ratio
    qmean[res.aa, res.ic, res.jj] = res.qmean
    qstd[res.aa, res.ic, res.jj]  = res.qstd
end

jldsave("./results/ActivationConDivSearch.jld2"; N, g, J0, T, saveat, qmean, qstd, num_repeats,
        ratio, τc_vec, tspan, t_warmup, activation_names)
