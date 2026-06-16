using Distributed
using SlurmClusterManager
using JLD2
using Dates
const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")
const DMFT_PATH = joinpath(@__DIR__, "DMFT.jl")
addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "16"])

@everywhere begin
    include($UTILS_PATH)
    include($DMFT_PATH)
end
# With 12 workers, this should take about 20 mins
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    using Distributed, LinearAlgebra, FFTW
    using .Utils, .DMFT
    using Dates
    BLAS.set_num_threads(8)

    const N = 4000
    const g_vec = [2.0, 3.0, 4.0]
    const tONS = 0.1
    const num_npoints = 12
    const τ_chn_vec_num = LinRange(0.0, 10.0/N, num_npoints)
    const theo_npoints = 36
    const τ_chn_vec_theo = LinRange(0.0, 10.0/N, theo_npoints)
    const J0 = -5.0 / N
    const nNets = 10
    const T_SIM = 100.0
    const n_quad = 512
end

const 📫 = RemoteChannel(()->Channel{String}(1000))
@async begin
    while true
        ✉️ = take!(📫)
        println(✉️)
        flush(stdout)
    end
end

@everywhere function ComputeNumericalLLE(g::Float64, τchn::Float64, log_channel::RemoteChannel)
    start = now()
    τrec = 2.0 * τchn
    τ = (τchn, τrec)

    try
        result = ComputeLSPR(N, J0, g, τ;
            burn_in=50.0, 
            T=T_SIM, 
            n_samples=nNets, 
            tONS=tONS, 
            verbose=false,
            chaos_only=false
        )

        if isnothing(result)
            return (mean=NaN, std=NaN)
        elseif result.LS_mean isa Number
            return (mean=Float64(result.LS_mean), std=Float64(result.LS_std))
        else
            return (mean=result.LS_mean[1], std=result.LS_std[1])
        end
    catch e
        put!(log_channel, "Worker $(myid()) failed numerical N*τchn=$(N * τchn), g=$g: $e")
        return (mean=NaN, std=NaN)
    end
end

@everywhere function ComputeTheoreticalLLE(g::Float64, τchn::Float64, log_channel::RemoteChannel)
    start = now()
    τrec = 2.0 * τchn
    τ = (τchn, τrec)
    
    if g <= 2.0 && τchn < 1.0/N
        μ0 = 0.0
        J0_temp = 0.0 # Otherwise the iteration is unstable
    else
        μ0 = nothing
        J0_temp = J0
    end

    try
        sol = DMFT_Stationary_Solver_B0(
            N, J0_temp, g, τ;
            dt = 0.05, T = 200.0,
            μ0 = μ0,
            max_iter = 500,
            tol = 5e-5,
            damp = 0.8,
            verbose = false,
            minimal_return = false,
            n_quad = n_quad
        )

        if !sol.converged
            put!(log_channel, "Worker $(myid()) did not converge theoretical N*τchn=$(N * τchn), g=$g")
            return NaN
        end

        res = LLE_DMFT(sol, N, g, τ; n_quad=n_quad)
        elapsed = now() - start
        dur = round(Dates.value(elapsed)/1e3, digits=1)

        return res.λmax
    catch e
        put!(log_channel, "Worker $(myid()) failed theoretical N*τchn=$(N * τchn), g=$g: $e")
        return NaN
    end
end

LLE_mean_num = fill(NaN, length(g_vec), length(τ_chn_vec_num))
LLE_std_num = fill(NaN, length(g_vec), length(τ_chn_vec_num))
LLE_mean_theo = fill(NaN, length(g_vec), length(τ_chn_vec_theo))

total_start = now()

for (ig, g) in enumerate(g_vec)
    put!(📫, "Starting numerical row $ig / $(length(g_vec)): g=$g")

    numerical_row = pmap(τchn -> ComputeNumericalLLE(g, τchn, 📫), τ_chn_vec_num)
    for (iτ, result) in enumerate(numerical_row)
        LLE_mean_num[ig, iτ] = result.mean
        LLE_std_num[ig, iτ] = result.std
    end

    put!(📫, "Finished numerical row $ig / $(length(g_vec)): g=$g")
end

@everywhere begin
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(16)
end

for (ig, g) in enumerate(g_vec)
    put!(📫, "Starting theoretical row $ig / $(length(g_vec)): g=$g")

    theoretical_row = pmap(τchn -> ComputeTheoreticalLLE(g, τchn, 📫), τ_chn_vec_theo)
    for (iτ, λmax) in enumerate(theoretical_row)
        LLE_mean_theo[ig, iτ] = λmax
    end

    put!(📫, "Finished theoretical row $ig / $(length(g_vec)): g=$g")
end

elapsed = now() - total_start
put!(📫, "Finished all LLE scans. Total elapsed=$(round(Dates.value(elapsed)/1e3, digits=1))s")

output_file = joinpath("results", "LLENumericalVSTheoretical.jld2")
jldsave(output_file;
    LLE_mean_num, LLE_std_num,
    LLE_mean_theo, N, g_vec, tONS, num_npoints, τ_chn_vec_num,
    theo_npoints, τ_chn_vec_theo, J0, nNets, T_SIM, n_quad)
