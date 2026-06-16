using Distributed
using SlurmClusterManager
using Dates
using JLD2

const τrec_vec = range(0.5, 0.8, 8)
const τchn_vec = range(-0.225, -0.16, 8)
const N_vec = [250, 500, 750]
const example_N = 500

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "24"])
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include("Utils.jl")
end

const 📫 = RemoteChannel(()->Channel{String}(1000))
@async begin
    while true
        ✉️ = take!(📫)
        println(✉️)
        flush(stdout)
    end
end

@everywhere begin
    using DifferentialEquations, LinearAlgebra, StatsBase, Dates, Random
    using .Utils

    BLAS.set_num_threads(1)

    const g_eff = 3.0
    const J0 = 0.0
    const T = 1000.0
    const nNets = 10
    const nInits = 4800
    const example_nInits = 12
    const example_tspan = (0.0, 2000.0)
    const example_saveat = 0.5
    const example_seed = 12345
    const σ0 = 1.0
    const tONS = 0.25
    const lle_window = 50.0
    const lle_window_count = Int(round(lle_window / tONS))
    const lle_tol = 1e-3

    function sem(x::AbstractVector{<:Real})
        return length(x) ≤ 1 ? 0.0 : std(x) / sqrt(length(x))
    end

    mutable struct LLEState
        J::Matrix{Float64}
        tangent_buffer::Vector{Float64}
        lle_buffer::Vector{Float64}
        lle_sum::Float64
        lle_count::Int
        lle_buffer_idx::Int
        last_lle::Float64
        window_lle::Float64
    end

    function LLEState(J::Matrix{Float64})
        return LLEState(
            J,
            zeros(size(J, 1)),
            zeros(lle_window_count),
            0.0,
            0,
            1,
            NaN,
            NaN,
        )
    end

    function initial_lle_state(N::Integer)
        x0 = σ0 .* randn(N)
        q0 = randn(N)
        q0 ./= norm(q0)
        return vcat(x0, q0)
    end

    function GaussianTanhLLE!(du, u, state::LLEState, t)
        N = size(state.J, 1)
        x = @view u[1:N]
        q = @view u[(N + 1):(2 * N)]
        dx = @view du[1:N]
        dq = @view du[(N + 1):(2 * N)]

        GaussianTanh!(dx, x, state.J, t)
        @. state.tangent_buffer = (1.0 - tanh(x)^2) * q
        mul!(dq, state.J, state.tangent_buffer)
        @. dq = dq - q

        return nothing
    end

    function update_lle!(integrator)
        state = integrator.p
        N = size(state.J, 1)
        q = @view integrator.u[(N + 1):(2 * N)]
        q_norm = norm(q)

        state.last_lle = log(q_norm) / tONS
        q ./= q_norm

        if state.lle_count == lle_window_count
            state.lle_sum -= state.lle_buffer[state.lle_buffer_idx]
        else
            state.lle_count += 1
        end

        state.lle_buffer[state.lle_buffer_idx] = state.last_lle
        state.lle_sum += state.last_lle
        state.lle_buffer_idx = mod1(state.lle_buffer_idx + 1, lle_window_count)

        if state.lle_count == lle_window_count
            state.window_lle = state.lle_sum / lle_window_count
        end
    end

    const lle_callback_times = collect(tONS:tONS:T)
    const lle_callback = PresetTimeCallback(
        lle_callback_times,
        update_lle!,
        save_positions=(false, false)
    )

    function simulate_convergence_fraction(J::AbstractMatrix{Float64})
        N = size(J, 1)

        function prob_func(prob, i, repeat)
            remake(prob; u0=initial_lle_state(N), p=LLEState(J))
        end

        function output_func(sol, i)
            state = sol.prob.p

            return (
                converged=state.lle_count == lle_window_count && state.window_lle ≤ lle_tol,
                lle=state.window_lle,
                local_lle=state.last_lle,
            ), false
        end

        prob = ODEProblem(GaussianTanhLLE!, initial_lle_state(N), (0.0, T), LLEState(J))
        ensemble_prob = EnsembleProblem(prob;
            prob_func=prob_func,
            output_func=output_func)

        sim = solve(ensemble_prob, Tsit5(), EnsembleThreads();
            trajectories=nInits,
            callback=lle_callback,
            save_everystep=false,
            save_start=false,
            save_end=false,
            maxiters=Inf)

        return sim.u
    end

    function simulate_mean_trajectories(J::AbstractMatrix{Float64}; seed::Int64=example_seed)
        N = size(J, 1)
        save_times = collect(example_tspan[1]:example_saveat:example_tspan[2])
        rng = MersenneTwister(seed)
        u0_list = [σ0 .* randn(rng, N) for _ in 1:example_nInits]

        function prob_func(prob, i, repeat)
            remake(prob; u0=copy(u0_list[i]))
        end

        function output_func(sol, i)
            return (
                t=Vector(sol.t),
                mean_x=Float64[mean(u) for u in sol.u],
            ), false
        end

        prob = ODEProblem(GaussianTanh!, copy(u0_list[1]), example_tspan, J)
        ensemble_prob = EnsembleProblem(prob;
            prob_func=prob_func,
            output_func=output_func)

        sim = solve(ensemble_prob, Tsit5(), EnsembleThreads();
            trajectories=example_nInits,
            saveat=save_times,
            save_everystep=false,
            save_start=true,
            save_end=true,
            maxiters=Inf)

        trajectory_t = sim.u[1].t
        mean_x = Matrix{Float64}(undef, length(trajectory_t), example_nInits)
        for init_idx in 1:example_nInits
            mean_x[:, init_idx] .= sim.u[init_idx].mean_x
        end

        return (
            t=trajectory_t,
            mean_x=mean_x,
            seed=seed,
        )
    end

    function compute_converge_fraction_for_param(param, log_channel::RemoteChannel)
        start_time = now()
        N = param.N
        τ = param.τ
        g = compute_g(g_eff, τ)
        put!(
            log_channel,
            "Worker $(myid()) starting N=$N $(param.kind) index $(param.index), τ=$(param.τ_value), T=$T"
        )

        fraction_converged = zeros(nNets)
        converged_by_net = falses(nNets, nInits)
        lle_values_by_net = fill(NaN, nNets, nInits)

        for net_idx in 1:nNets
            J = CreateJ(N, J0, g, τ; parallel=true, verbose=false)

            results = simulate_convergence_fraction(J)
            converged_by_net[net_idx, :] .= [result.converged for result in results]
            lle_values_by_net[net_idx, :] .= [result.lle for result in results]
            fraction_converged[net_idx] = mean(converged_by_net[net_idx, :])

            put!(
                log_channel,
                "Worker $(myid()) N=$N $(param.kind) index $(param.index), τ=$(param.τ_value): finished network $net_idx/$nNets, " *
                "fraction converged=$(fraction_converged[net_idx])"
            )
        end

        put!(log_channel, "Worker $(myid()) N=$N $(param.kind) index $(param.index), τ=$(param.τ_value) finished in $(format_elapsed(start_time))")

        return (
            N=N,
            kind=param.kind,
            index=param.index,
            τ_value=param.τ_value,
            τ=τ,
            g=g,
            fraction_converged=fraction_converged,
            fraction_converged_mean=mean(fraction_converged),
            fraction_converged_sem=sem(fraction_converged),
            converged=converged_by_net,
            lle_values=lle_values_by_net,
        )
    end

    function simulate_mean_trajectory_example(param, log_channel::RemoteChannel)
        start_time = now()
        N = param.N
        τ = param.τ
        g = compute_g(g_eff, τ)
        seed = example_seed + param.seed_offset
        put!(
            log_channel,
            "Worker $(myid()) starting mean trajectory example N=$N $(param.kind): " *
            "τ=$(param.τ_value), seed=$seed"
        )

        Random.seed!(seed)
        J = CreateJ(N, J0, g, τ; parallel=false, verbose=false)
        trajectory_data = simulate_mean_trajectories(J; seed=seed)

        put!(
            log_channel,
            "Worker $(myid()) finished mean trajectory example N=$N $(param.kind): " *
            "τ=$(param.τ_value). Elapsed = $(format_elapsed(start_time))"
        )

        return (
            N=N,
            kind=param.kind,
            τ_value=param.τ_value,
            τ=τ,
            g=g,
            seed=seed,
            t=trajectory_data.t,
            mean_x=trajectory_data.mean_x,
        )
    end
end

function compute_converge_fraction_scan()
    rec_params = [
        (N=N, kind=:rec, index=i, τ_value=Float64(τrec), τ=(0.0, Float64(τrec)))
        for N in N_vec
        for (i, τrec) in enumerate(τrec_vec)
    ]
    chn_params = [
        (N=N, kind=:chn, index=i, τ_value=Float64(τchn), τ=(Float64(τchn), 0.0))
        for N in N_vec
        for (i, τchn) in enumerate(τchn_vec)
    ]

    results = pmap([rec_params; chn_params]; batch_size=1) do param
        compute_converge_fraction_for_param(param, 📫)
    end

    n_rec = length(rec_params)
    rec_results = results[1:n_rec]
    chn_results = results[(n_rec + 1):end]

    rec_by_N = [[result for result in rec_results if result.N == N] for N in N_vec]
    chn_by_N = [[result for result in chn_results if result.N == N] for N in N_vec]
    by_N = [
        (N=N, rec=rec_by_N[i], chn=chn_by_N[i])
        for (i, N) in enumerate(N_vec)
    ]
    return by_N
end

function simulate_mean_trajectory_examples()
    params = [
        (N=example_N, kind=:rec, τ_value=0.8, τ=(0.0, 0.8), seed_offset=1),
        (N=example_N, kind=:chn, τ_value=-0.2, τ=(-0.2, 0.0), seed_offset=2),
    ]

    results = pmap(params; batch_size=1) do param
        simulate_mean_trajectory_example(param, 📫)
    end

    return (rec=results[1], chn=results[2])
end

start = now()
scan_results = compute_converge_fraction_scan()
println("Convergence-fraction scan finished. Elapsed = $(format_elapsed(start))")
flush(stdout)

example_start = now()
mean_trajectory_examples = simulate_mean_trajectory_examples()
println("Mean trajectory examples finished. Elapsed = $(format_elapsed(example_start))")
flush(stdout)

results_dir = joinpath(@__DIR__, "results")
output_filename = joinpath(results_dir, "ConvergeFractionSearch.jld2")
jldsave(output_filename;
    ConvergenceFractionByN=scan_results,
    MeanTrajectoryExamples=mean_trajectory_examples,
    τrec_vec, τchn_vec,
    N_vec, example_N, g_eff, J0, T, nNets, nInits,
    example_nInits, example_tspan, example_saveat, example_seed, σ0,
    tONS, lle_window, lle_window_count, lle_tol)
close(📫)
