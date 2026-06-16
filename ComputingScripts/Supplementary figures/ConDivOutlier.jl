using LinearAlgebra, Distributed, SlurmClusterManager, JLD2, Dates
using Random
const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "16"])
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include($UTILS_PATH)
end

@everywhere begin
    using LinearAlgebra
    using Random
    using .Utils
    BLAS.set_num_threads(16)

    function largest_real_part(N::Int, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}})
        J = CreateJ(N, J0, g, τ; parallel=true)
        J === nothing && error("CreateJ failed for τ = $(τ).")
        return maximum(real, eigvals(J))
    end
end

const N_vec = [1000, 2000, 4000]
const g_eff = 3.0
const τ_con_vec = [0.0, 0.1, 0.2, 0.3]
const τ_div_vec = [0.0, 0.1, 0.2, 0.3]
const J0 = 0.0
const n_samples = 200
const scan_seed = 12345

function scan_tau_axis!(largest_real_parts, τ_vec, make_τ, axis_label, seed_axis_offset, start_time)
    base_seed = scan_seed

    for (iN, N) in enumerate(N_vec)
        println("Starting $(axis_label) N $(iN)/$(length(N_vec)): N=$(N).")
        flush(stdout)

        for (iτ, τvalue) in enumerate(τ_vec)
            τ = make_τ(τvalue)
            g = compute_g(g_eff, τ)
            sample_results = pmap(1:n_samples) do sample_idx
                Random.seed!(base_seed + seed_axis_offset + 1000000 * iN + 10000 * iτ + sample_idx)
                largest_real_part(N, J0, g, τ)
            end

            largest_real_parts[iN, iτ, :] .= sample_results
            println("Finished $(axis_label) $(iτ)/$(length(τ_vec)) for N $(iN)/$(length(N_vec)). Elapsed: $(format_elapsed(start_time))")
            flush(stdout)
        end
    end

    return largest_real_parts
end

function main()
    largest_real_parts_con = zeros(Float64, length(N_vec), length(τ_con_vec), n_samples)
    largest_real_parts_div = zeros(Float64, length(N_vec), length(τ_div_vec), n_samples)
    start_time = now()

    scan_tau_axis!(
        largest_real_parts_con, τ_con_vec, τcon -> (0.0, 0.0, τcon, 0.0),
        "τcon", 0, start_time)

    scan_tau_axis!(
        largest_real_parts_div, τ_div_vec, τdiv -> (0.0, 0.0, 0.0, τdiv),
        "τdiv", 500000000, start_time)

    jldsave("results/ConDivOutlier.jld2";
        largest_real_parts_con, largest_real_parts_div,
        N_vec, τ_con_vec, τ_div_vec, g_eff, J0, n_samples,
        scan_seed)
end

main()
