using LinearAlgebra, DifferentialEquations, Random, StatsBase, JLD2, Dates
include("Utils.jl")
using .Utils
BLAS.set_num_threads(4)

N = 4000
τchn = -2.7142857142857144/N
τrec = 2*τchn
J0 = 4.0/N
τ = (τchn, τrec)
tspan = (0.0, 2000.0)
burn_in = 1900.0
saveat = 0.2
ntrajs = [8, 12, 8]
g_vec = [1.6943289218023703, 2.1358430536940127, 2.552828622702786]
σ_init = 0.5
μ_init_zero = 0.0
μ_init_nonzero = 0.6
base_seed = 12366

function simulate_ϕmean(
    J::AbstractMatrix{Float64},
    tspan::Tuple{Float64, Float64},
    burn_in::Float64,
    ntraj::Int64;
    seed::Int64
)
    iseven(ntraj) || throw(ArgumentError("ntraj must be even so half the initial conditions can use each mean."))

    rng = MersenneTwister(seed)
    tspan[1] <= burn_in < tspan[2] || throw(ArgumentError("burn_in must be inside tspan."))
    save_times = collect(burn_in:saveat:tspan[2])
    init_means = vcat(
        fill(μ_init_zero, ntraj ÷ 2),
        fill(μ_init_nonzero, ntraj ÷ 2),
    )
    x0_list = [σ_init .* randn(rng, N) .+ μ for μ in init_means]

    function prob_func(prob, i, repeat)
        remake(prob; u0=copy(x0_list[i]))
    end

    function output_func(sol, i)
        return (
            t=Vector(sol.t),
            mean_ϕ=Float64[mean(tanh.(u)) for u in sol.u],
        ), false
    end

    prob = ODEProblem(GaussianTanh!, copy(x0_list[1]), tspan, J)
    ensemble_prob = EnsembleProblem(prob;
        prob_func=prob_func,
        output_func=output_func)

    sim = solve(ensemble_prob, Tsit5(), EnsembleThreads();
        trajectories=ntraj,
        saveat=save_times,
        save_everystep=false,
        save_start=false,
        save_end=true,
        maxiters=Inf)

    trajectory_t = sim.u[1].t
    mean_ϕ = Matrix{Float64}(undef, length(trajectory_t), ntraj)
    for traj_idx in 1:ntraj
        mean_ϕ[:, traj_idx] .= sim.u[traj_idx].mean_ϕ
    end

    return (
        t=trajectory_t,
        mean_ϕ=mean_ϕ,
        init_means=init_means,
        seed=seed,
    )
end

examples = Vector{NamedTuple}(undef, length(g_vec))
for (i, g) in enumerate(g_vec)
    println("Starting example $(i)/$(length(g_vec)): g=$(g), ntraj=$(ntrajs[i]).")
    flush(stdout)
    start = now()
    Random.seed!(base_seed + 1000 * i)
    J = CreateJ(N, J0, g, (τchn, τrec); parallel = true)
    J === nothing && error("CreateJ failed for g=$(g), τ=$(τ).")
    eigenvalues = eigvals(J)
    trajectory_data = simulate_ϕmean(J, tspan, burn_in, ntrajs[i]; seed=base_seed + i)

    examples[i] = (
        g=g, 
        ntraj=ntrajs[i], eigenvalues=eigenvalues,
        t=trajectory_data.t,
        mean_ϕ=trajectory_data.mean_ϕ,
        init_means=trajectory_data.init_means,
        seed=trajectory_data.seed)

    println("Finished example $(i)/$(length(g_vec)): g=$(round(g, digits=4)), elapsed = $(format_elapsed(start)).")
    flush(stdout)
end

file_name = "NegChnExamplesN$(N)J0$(replace(string(round(N * J0, digits=1)), "." => "p"))B0.jld2"
output_file = joinpath("results", file_name)
jldsave(output_file;
    examples, N, J0, τ, tspan, burn_in, saveat, ntrajs,
    g_vec, σ_init, μ_init_zero, μ_init_nonzero, base_seed)
