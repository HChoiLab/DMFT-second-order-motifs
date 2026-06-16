using LinearAlgebra
using Distributed
using SlurmClusterManager
using JLD2
using Dates

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "32"])
println("Total workers: $(nworkers())")
flush(stdout)
# With 20 workers, each motif can take several hours
@everywhere begin
    include($UTILS_PATH)
end

@everywhere begin
    using LinearAlgebra
    using .Utils
    using Dates
    BLAS.set_num_threads(1)

    const N = 2000
    const g_eff_vec = [2.0, 4.0]
    const num_points = 10
    const T = 10.0 * N
    # const τ_rec_vec = LinRange(-0.5,  0.5, num_points)
    const τ_chn_vec = LinRange(-0.20, 0.0, num_points)
    # const τ_con_vec = LinRange(0.0, 60.0/N, num_points)
    # const τ_div_vec = LinRange(0.0, 60.0/N, num_points)
    const J0 = 0.0
    const n_samples = 128
end

const 📫 = RemoteChannel(() -> Channel{String}(1000))
@async begin
    while true
        ✉️ = take!(📫)
        println(✉️)
        flush(stdout)
    end
end
@everywhere function compute_wrapper(g_eff::Float64, τ_tuple::Tuple{Vararg{Float64}},
                                     log_channel::RemoteChannel)
    g = compute_g(g_eff, τ_tuple)
    τ_log = round.(τ_tuple; digits=4)
    put!(log_channel, "Worker $(myid()) starting: g_eff=$g_eff, τ=$τ_log")
    start = now()

    if τ_tuple[2] - 2τ_tuple[4] > 0.4
        burn_in = 400.0
    elseif τ_tuple[2] - 2τ_tuple[4] > 0.1
        burn_in = 100.0
    else
        burn_in = 50.0
    end

    exclude_bimodal = (τ_tuple[3] > 0.0 || τ_tuple[4] > 0.0) && abs(τ_tuple[1]) < 0.05

    Dϕ_mean, Dϕ_std, C4_mean, C4_std, C2_mean, C2_std = NumericalPRD(N, J0, g, τ_tuple;
        n_samples=n_samples, burn_in=burn_in, T=T, exclude_bimodal=exclude_bimodal)

    dur = round(Dates.value(now() - start) / 1e3, digits=1)
    put!(log_channel, "Worker $(myid()) finished g_eff=$g_eff, τ=$τ_log. Elapsed = $(dur)s")
    return (Dϕ_mean, Dϕ_std, C4_mean, C4_std, C2_mean, C2_std)
end

function collect_results(raw, τ_vec)
    ng = length(g_eff_vec)
    nτ = length(τ_vec)
    reshape_t(i) = reshape([r[i] for r in raw], nτ, ng)'
    return (Dϕ_mean = reshape_t(1), Dϕ_std = reshape_t(2),
            C2_mean  = reshape_t(3), C2_std  = reshape_t(4),
            C4_mean  = reshape_t(5), C4_std  = reshape_t(6),
            τ = τ_vec)
end

chn_params = [(g_eff, (τ_chn, 0.0, abs(τ_chn), abs(τ_chn)))
              for g_eff in g_eff_vec for τ_chn in τ_chn_vec]
result_chn = collect_results(pmap(p -> compute_wrapper(p[1], p[2], 📫), chn_params), τ_chn_vec)
println("Chain motif done."); flush(stdout)

# rec_params = [(g_eff, (0.0, τ_rec, 0.0, 0.0))
#               for g_eff in g_eff_vec for τ_rec in τ_rec_vec]
# result_rec = collect_results(pmap(p -> compute_wrapper(p[1], p[2], 📫), rec_params), τ_rec_vec)
# println("Reciprocal motif done."); flush(stdout)

# con_params = [(g_eff, (0.0, 0.0, τ_con, 0.0))
#               for g_eff in g_eff_vec for τ_con in τ_con_vec]
# result_con = collect_results(pmap(p -> compute_wrapper(p[1], p[2], 📫), con_params), τ_con_vec)
# println("Convergent motif done."); flush(stdout)

# div_params = [(g_eff, (0.0, 0.0, 0.0, τ_div))
#               for g_eff in g_eff_vec for τ_div in τ_div_vec]
# result_div = collect_results(pmap(p -> compute_wrapper(p[1], p[2], 📫), div_params), τ_div_vec)
# println("Divergent motif done."); flush(stdout)

file_name = joinpath(@__DIR__, "results/NumericalPRDChnFullRange.jld2")

jldsave(file_name; N, J0, g_eff_vec, n_samples, T, result_chn)
