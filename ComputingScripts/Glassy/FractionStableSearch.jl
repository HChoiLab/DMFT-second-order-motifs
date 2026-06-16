using Distributed
using SlurmClusterManager
using Dates
using JLD2
using Random

const τrec = parse(Float64, get(ENV, "TAUREC", "0.8"))
const τchn = parse(Float64, get(ENV, "TAUCHN", "0.0"))
const seed = parse(Int, get(ENV, "SEED", "1"))

function output_parameter_token(x::Real)
    s = string(round(Float64(x), digits=2))
    return replace(s, "." => "p")
end

const results_dir = joinpath(@__DIR__, "results")
const output_filename = joinpath(
    results_dir,
    "FractionStableSearch_taurec$(output_parameter_token(τrec))_tauchn$(output_parameter_token(τchn))_seed$(seed).jld2"
)
const checkpoint_dir = joinpath(results_dir, "tmp")
const checkpoint_filename = joinpath(
    checkpoint_dir,
    "FractionStableSearch_taurec$(output_parameter_token(τrec))_tauchn$(output_parameter_token(τchn))_seed$(seed)_checkpoint.jld2"
)

function valid_final_output(path::String)
    isfile(path) || return false
    try
        data = load(path)
        haskey(data, "search_result") || return false
        result = data["search_result"]
        return hasproperty(result, :distinct_fp_by_ic) && hasproperty(result, :fixed_points)
    catch err
        @warn "Existing output file could not be loaded; will recompute." path exception=(err, catch_backtrace())
        return false
    end
end

if isfile(output_filename)
    if valid_final_output(output_filename)
        println("Output file already exists, skipping computation: $output_filename")
        if isfile(checkpoint_filename)
            rm(checkpoint_filename; force=true)
            println("Deleted stale checkpoint file $checkpoint_filename")
        end
        flush(stdout)
        exit(0)
    else
        println("Output file exists but is not a valid completed result; continuing from checkpoint or recomputing: $output_filename")
        flush(stdout)
    end
end

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "32"])
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
    using LinearAlgebra
    using Dates
    using JLD2
    using Random
    using .Utils

    BLAS.set_num_threads(1)

    const DEFAULT_CONFIG = FixedPointSearchConfig(
        $seed, 100, 3.0, 0.0, 10^7,
        4096, 1.0, 200.0, 1e-10, 1e-3, 1e-10,
        1e-10, 100, 1e-3, 16
    )

    const N = DEFAULT_CONFIG.N
    const g_eff = DEFAULT_CONFIG.g_eff
    const J0 = DEFAULT_CONFIG.J0
    const nInits = DEFAULT_CONFIG.n_inits
    const batchsize = DEFAULT_CONFIG.batchsize
    const μ0σ = DEFAULT_CONFIG.mu0_sigma
    const σ0 = DEFAULT_CONFIG.sigma0
    const residual_tol = DEFAULT_CONFIG.residual_tol
    const fp_distance_tol = DEFAULT_CONFIG.fp_distance_tol
    const nl_reltol = DEFAULT_CONFIG.nl_reltol
    const nl_abstol = DEFAULT_CONFIG.nl_abstol
    const max_nl_iters = DEFAULT_CONFIG.max_nl_iters
    const stability_tol = DEFAULT_CONFIG.stability_tol

    function checkpoint_config_signature(config::FixedPointSearchConfig)
        return (
            seed=config.seed,
            N=config.N,
            g_eff=config.g_eff,
            J0=config.J0,
            n_inits=config.n_inits,
            batchsize=config.batchsize,
            mu0_sigma=config.mu0_sigma,
            sigma0=config.sigma0,
            residual_tol=config.residual_tol,
            fp_distance_tol=config.fp_distance_tol,
            nl_reltol=config.nl_reltol,
            nl_abstol=config.nl_abstol,
            max_nl_iters=config.max_nl_iters,
            stability_tol=config.stability_tol,
            distributed_task_batches=config.distributed_task_batches,
        )
    end

    function validate_checkpoint!(checkpoint, τrec::Float64, τchn::Float64, τ, g::Float64, config::FixedPointSearchConfig)
        required_keys = (
            "τrec", "τchn", "τ", "g",
            "config_signature", "J", "fixed_points", "completed_batches",
            "residual_sum", "residual_max", "n_converged", "distinct_fp_by_ic"
        )
        for key in required_keys
            haskey(checkpoint, key) || error("Checkpoint is missing required key: $key")
        end
        checkpoint["τrec"] == τrec ||
            error("Checkpoint τrec=$(checkpoint["τrec"]) does not match requested τrec=$τrec.")
        checkpoint["τchn"] == τchn ||
            error("Checkpoint τchn=$(checkpoint["τchn"]) does not match requested τchn=$τchn.")
        checkpoint["τ"] == τ ||
            error("Checkpoint τ=$(checkpoint["τ"]) does not match requested τ=$τ.")
        checkpoint["g"] == g ||
            error("Checkpoint g=$(checkpoint["g"]) does not match requested g=$g.")
        checkpoint["config_signature"] == checkpoint_config_signature(config) ||
            error("Checkpoint configuration does not match the current FixedPointSearchConfig.")
        return checkpoint
    end

    function load_search_checkpoint(checkpoint_path::String, τrec::Float64, τchn::Float64, τ, g::Float64, config::FixedPointSearchConfig)
        isfile(checkpoint_path) || return nothing
        checkpoint = load(checkpoint_path)
        validate_checkpoint!(checkpoint, τrec, τchn, τ, g, config)
        return checkpoint
    end

    function save_search_checkpoint(
        checkpoint_path::String,
        context;
        fixed_points,
        completed_batches,
        residual_sum::Float64,
        residual_max::Float64,
        n_converged::Int,
        distinct_fp_by_ic,
        status::String="running",
    )
        mkpath(dirname(checkpoint_path))
        tmp_path = checkpoint_path * ".tmp"
        isfile(tmp_path) && rm(tmp_path; force=true)
        jldsave(tmp_path;
            status,
            saved_at=string(now()),
            τrec=context.τrec,
            τchn=context.τchn,
            τ=context.τ,
            g=context.g,
            config_signature=context.config_signature,
            J=context.J,
            fixed_points,
            completed_batches,
            residual_sum,
            residual_max,
            n_converged,
            distinct_fp_by_ic,
        )
        mv(tmp_path, checkpoint_path; force=true)
        return nothing
    end

    function checkpoint_every_batches(n_batches::Int, config::FixedPointSearchConfig)
        allocated_batches = max(1, nworkers() * max(1, config.distributed_task_batches))
        return min(n_batches, max(1, cld(allocated_batches, 16)))
    end

    function solve_fixed_point_batches(
        τ_label::String,
        log_channel::RemoteChannel,
        config::FixedPointSearchConfig,
        checkpoint_path::String,
        checkpoint,
        checkpoint_context,
    )
        n_batches = cld(config.n_inits, config.batchsize)
        if checkpoint === nothing
            representative_index = FixedPointRepresentativeIndex(config)
            completed_batches = falses(n_batches)
            distinct_fp_by_ic = NamedTuple{(:n_ic, :n_fp),Tuple{Int,Int}}[]
            residual_sum = 0.0
            residual_max = NaN
            n_converged = 0
        else
            representative_index = restore_representative_index(checkpoint["fixed_points"], config)
            completed_batches = BitVector(checkpoint["completed_batches"])
            length(completed_batches) == n_batches ||
                error("Checkpoint completed batch count $(length(completed_batches)) does not match expected $n_batches.")
            distinct_fp_by_ic = Vector{NamedTuple{(:n_ic, :n_fp),Tuple{Int,Int}}}(checkpoint["distinct_fp_by_ic"])
            length(distinct_fp_by_ic) == count(completed_batches) ||
                error("Checkpoint distinct_fp_by_ic length $(length(distinct_fp_by_ic)) does not match completed batch count $(count(completed_batches)).")
            residual_sum = Float64(checkpoint["residual_sum"])
            residual_max = Float64(checkpoint["residual_max"])
            n_converged = Int(checkpoint["n_converged"])
            put!(
                log_channel,
                "$τ_label: resumed from checkpoint with $(count(completed_batches))/$n_batches batches merged, " *
                "converged=$n_converged, distinct candidates=$(length(representative_index.representatives))"
            )
        end
        completed_count = count(completed_batches)
        pending_batch_indices = findall(!, completed_batches)
        if isempty(pending_batch_indices)
            residual_mean = n_converged == 0 ? NaN : residual_sum / n_converged
            return representative_index.representatives, residual_mean, residual_max, n_converged, distinct_fp_by_ic
        end

        batch_ranges = distributed_batch_ranges(pending_batch_indices, config)
        result_channel = RemoteChannel(() -> Channel{Any}(max(1, 4 * nworkers())))
        log_every = max(1, n_batches ÷ 100)
        checkpoint_every = checkpoint_every_batches(n_batches, config)
        n_merged_batches = completed_count
        next_log_batch = max(log_every, cld(completed_count + 1, log_every) * log_every)
        next_checkpoint_batch = max(
            completed_count + checkpoint_every,
            cld(completed_count + 1, checkpoint_every) * checkpoint_every,
        )

        put!(
            log_channel,
            "$τ_label: checkpointing every $checkpoint_every merged batches"
        )

        function save_current_checkpoint(status::String)
            save_search_checkpoint(
                checkpoint_path,
                checkpoint_context;
                fixed_points=representative_index.representatives,
                completed_batches=completed_batches,
                residual_sum=residual_sum,
                residual_max=residual_max,
                n_converged=n_converged,
                distinct_fp_by_ic=distinct_fp_by_ic,
                status=status,
            )
            return nothing
        end

        producer = @async begin
            try
                pmap(batch_ranges; batch_size=1) do batch_range
                    solve_fixed_point_batch_range(batch_range, result_channel, config)
                    return nothing
                end
            catch err
                put!(result_channel, CapturedException(err, catch_backtrace()))
            end
        end

        while n_merged_batches < n_batches
            message = take!(result_channel)
            if message isa CapturedException
                wait(producer)
                throw(message)
            end

            batch_idx, batch_result = message::Tuple{Int,FixedPointBatchSummary}
            completed_batches[batch_idx] && error("Received duplicate result for completed batch $batch_idx.")
            previous_n_ic = n_merged_batches == 0 ? 0 : distinct_fp_by_ic[end].n_ic
            completed_batches[batch_idx] = true
            merge_distinct_fixed_points!(representative_index, batch_result.fixed_points)
            n_merged_batches += 1
            push!(distinct_fp_by_ic, (
                n_ic=previous_n_ic + batch_result.n_inits,
                n_fp=length(representative_index.representatives),
            ))

            residual_sum += batch_result.residual_sum
            if !isnan(batch_result.residual_max)
                residual_max = isnan(residual_max) ? batch_result.residual_max : max(residual_max, batch_result.residual_max)
            end
            n_converged += batch_result.n_converged

            if n_merged_batches == n_batches || n_merged_batches >= next_checkpoint_batch
                save_current_checkpoint(n_merged_batches == n_batches ? "batches_complete" : "running")
                put!(
                    log_channel,
                    "$τ_label: saved checkpoint after $n_merged_batches/$n_batches merged batches to $checkpoint_path"
                )
                while next_checkpoint_batch <= n_merged_batches
                    next_checkpoint_batch += checkpoint_every
                end
            end

            if n_merged_batches == n_batches || n_merged_batches >= next_log_batch
                put!(log_channel,
                    "$τ_label: " *
                    "merged $n_merged_batches/$n_batches batches, " *
                    "converged=$n_converged, distinct candidates=$(length(representative_index.representatives))"
                )
                while next_log_batch <= n_merged_batches
                    next_log_batch += log_every
                end
            end
        end
        wait(producer)

        residual_mean = n_converged == 0 ? NaN : residual_sum / n_converged
        return representative_index.representatives, residual_mean, residual_max, n_converged, distinct_fp_by_ic
    end

    function compute_stable_fixed_point_for_τ(
        τrec::Float64,
        τchn::Float64,
        log_channel::RemoteChannel,
        config::FixedPointSearchConfig
    )
        start_time = now()
        τ = (τchn, τrec)
        τ_label = "τrec=$τrec, τchn=$τchn"
        g = compute_g(config.g_eff, τ)
        put!(log_channel, "Starting $τ_label, seed=$(config.seed), nInits=$(config.n_inits)")

        checkpoint_path = $checkpoint_filename
        checkpoint = load_search_checkpoint(checkpoint_path, τrec, τchn, τ, g, config)
        if checkpoint === nothing
            Random.seed!(config.seed)
            J = CreateJ(config.N, config.J0, g, τ; parallel=false, verbose=false)
            J === nothing && error("CreateJ failed for τ = $τ.")
        else
            J = Matrix{Float64}(checkpoint["J"])
            put!(log_channel, "$τ_label: loaded J from checkpoint $checkpoint_path")
        end
        checkpoint_context = (
            τrec=τrec,
            τchn=τchn,
            τ=τ,
            g=g,
            config_signature=checkpoint_config_signature(config),
            J=J,
        )
        if checkpoint === nothing
            n_batches = cld(config.n_inits, config.batchsize)
            save_search_checkpoint(
                checkpoint_path,
                checkpoint_context;
                fixed_points=FixedPointCandidate[],
                completed_batches=falses(n_batches),
                residual_sum=0.0,
                residual_max=NaN,
                n_converged=0,
                distinct_fp_by_ic=NamedTuple{(:n_ic, :n_fp),Tuple{Int,Int}}[],
            )
        end
        set_worker_J!(J)
        @sync for pid in workers()
            @async remotecall_wait(set_worker_J!, pid, J)
        end

        fixed_point_results, residual_mean, residual_max, n_converged, distinct_fp_by_ic =
            solve_fixed_point_batches(τ_label, log_channel, config, checkpoint_path, checkpoint, checkpoint_context)
        put!(
            log_channel,
            "$τ_label: checking stability for $(length(fixed_point_results)) distinct candidates "
        )
        distinct_points = summarize_fixed_points(fixed_point_results, J, config)

        n_fp = length(distinct_points)
        n_stable_fp = count(fp -> fp.stable, distinct_points)
        fraction_stable = n_fp == 0 ? NaN : n_stable_fp / n_fp
        convergence_rate = n_converged / config.n_inits

        put!(log_channel, "$τ_label: stability check finished, stable=$n_stable_fp/$n_fp")
        put!(log_channel, "$τ_label finished in $(format_elapsed(start_time))")

        return (
            τrec=τrec,
            τchn=τchn,
            τ=τ,
            g=g,
            n_fp=n_fp,
            n_stable_fp=n_stable_fp,
            fraction_stable=fraction_stable,
            residual_mean=residual_mean,
            residual_max=residual_max,
            convergence_rate=convergence_rate,
            distinct_fp_by_ic=distinct_fp_by_ic,
            fixed_points=distinct_points,
            J=J,
        )
    end
end

function compute_fixed_point_search()
    return compute_stable_fixed_point_for_τ(τrec, τchn, 📫, DEFAULT_CONFIG)
end

start = now()
search_result = compute_fixed_point_search()
output_tmp_filename = output_filename * ".tmp"
isfile(output_tmp_filename) && rm(output_tmp_filename; force=true)
jldsave(output_tmp_filename;
    search_result, τrec, τchn, seed, N, g_eff, J0,
    nInits, batchsize, μ0σ, σ0, residual_tol, fp_distance_tol,
    nl_reltol, nl_abstol, max_nl_iters, stability_tol)
mv(output_tmp_filename, output_filename; force=true)
println("Saved results to $output_filename")
flush(stdout)
if isfile(checkpoint_filename)
    rm(checkpoint_filename; force=true)
end

println("fraction_converged = $(search_result.convergence_rate)")
println("distinct_fixed_points = $(search_result.n_fp)")
println("stable_fixed_points = $(search_result.n_stable_fp)")
println("fraction_stable = $(search_result.fraction_stable)")
println("residual_max = $(search_result.residual_max)")
flush(stdout)

close(📫)
