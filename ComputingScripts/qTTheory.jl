using LinearAlgebra, FFTW, Distributed, SlurmClusterManager, JLD2, Dates

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")
const DMFT_PATH = joinpath(@__DIR__, "DMFT.jl")
addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "16"])
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include($UTILS_PATH)
    include($DMFT_PATH)
end
# This should take less than 4min if using 36 workers
@everywhere begin
    using Distributed, LinearAlgebra, FFTW
    using .Utils, .DMFT
    using Dates
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(16)

    const N = 8000
    const g = 5.0
    const J0 = -5.0 / N
    const num_points = 12
    const τc_vec = range(0.1 / N, 0.3 / N, 3)

    const dt = 0.05
    const T = 300.0
    const max_iter = 500
    const tol = 1e-6
    const damp = 0.9
    const n_quad = 512
    const nTime = round(Int, T / dt)
end


const 📫 = RemoteChannel(() -> Channel{String}(1000))
@async begin
    while true
        println(take!(📫))
        flush(stdout)
    end
end

@everywhere function solve_stationary(job, log_channel::RemoteChannel)
    τ = (job.τc, 2.0 * job.τc, job.τcon, job.τdiv)
    label = "N*τc=$(round(N * job.τc, digits=3)), ratio=$(round(job.ratio, digits=3))"
    try
        sol = DMFT_Stationary_Solver_B0(N, J0, g, τ;
            ϕ=ϕ_positive, ϕ_prime=ϕ_positive_p,
            dt=dt, T=T,
            μ0=nothing,
            max_iter=max_iter, tol=tol, damp=damp,
            verbose=false, minimal_return=false, n_quad=n_quad)

        sol.converged || put!(log_channel, "Worker $(myid()) did not converge: $label")

        return (ic=job.ic, jj=job.jj, converged=sol.converged,
                mx=sol.mx, mϕ=sol.mϕ, Δx=sol.Δx,
                Cx=sol.Cx, Cϕ=sol.Cϕ, ratio=job.ratio)
    catch e
        put!(log_channel, "Worker $(myid()) FAILED: $label: $e")
        return (ic=job.ic, jj=job.jj, converged=false,
                mx=NaN, mϕ=NaN, Δx=NaN,
                Cx=fill(NaN, nTime), Cϕ=fill(NaN, nTime), ratio=job.ratio)
    end
end

jobs = NamedTuple[]
for (ic, τc) in enumerate(τc_vec)
    τdiv_vec = logrange(sqrt(0.1 * τc^2), sqrt(10.0 * τc^2), num_points)
    τcon_vec = τc^2 ./ τdiv_vec
    for jj in 1:num_points
        push!(jobs, (ic=ic, jj=jj, τc=Float64(τc),
                     τcon=τcon_vec[jj], τdiv=τdiv_vec[jj],
                     ratio=τdiv_vec[jj]^2 / τc^2))
    end
end

println("Dispatching $(length(jobs)) DMFT solves over $(nworkers()) workers.")
flush(stdout)

start = now()
results = pmap(job -> solve_stationary(job, 📫), jobs)
println("Finished all DMFT solves. Time elapsed: $(format_elapsed(start))")
flush(stdout)

nτc = length(τc_vec)
mx = fill(NaN, nτc, num_points)
mϕ = fill(NaN, nτc, num_points)
Δx = fill(NaN, nτc, num_points)
ratio = fill(NaN, nτc, num_points)
converged = falses(nτc, num_points)
Cx = fill(NaN, nTime, nτc, num_points)
Cϕ = fill(NaN, nTime, nτc, num_points)

for res in results
    mx[res.ic, res.jj] = res.mx
    mϕ[res.ic, res.jj] = res.mϕ
    Δx[res.ic, res.jj] = res.Δx
    ratio[res.ic, res.jj] = res.ratio
    converged[res.ic, res.jj] = res.converged
    Cx[:, res.ic, res.jj] .= res.Cx
    Cϕ[:, res.ic, res.jj] .= res.Cϕ
end

t_grid = collect((0:nTime-1) .* dt)

output_dir = joinpath(@__DIR__, "results")
output_file = joinpath(output_dir, "qTTheory.jld2")
jldsave(output_file;
    N, g, J0, num_points, τc_vec, n_quad,
    dt, T, max_iter, tol, damp,
    mx, mϕ, Δx, converged, ratio,
    Cx, Cϕ, t_grid)
println("Saved results to $output_file")
flush(stdout)
