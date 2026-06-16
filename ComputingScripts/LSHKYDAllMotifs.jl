using Distributed
using SlurmClusterManager
using JLD2
using Dates
using Random

const UTILS_PATH = joinpath(@__DIR__, "Utils.jl")

addprocs(SlurmManager(), exeflags=["--project=./MotifNets", "-t", "24"])
# This should take <36 hours if using 48 workers
println("Total workers: $(nworkers())")
flush(stdout)

@everywhere begin
    include($UTILS_PATH)
end

@everywhere begin
    using LinearAlgebra
    using .Utils
    using Dates
    BLAS.set_num_threads(1)

    const N = 1000
    const g_eff_vec = [2.0, 3.0, 4.0]
    const num_points = 16
    const τ_rec_vec = LinRange(-0.7, 0.7, num_points)
    const τ_chn_vec = LinRange(-0.2, 0.0, num_points)
    const τ_con_vec = LinRange(0.0, 0.35, num_points)
    const τ_div_vec = LinRange(0.0, 0.35, num_points)
    const τ_rec_max = maximum(τ_rec_vec)
    const τ_chn_abs_max = maximum(abs, τ_chn_vec)
    const J0 = 0.0

    const CHAOS_ONLY = true
    const RETURN_TRAJ = false
    const RETURN_HD = true
    const T_SIM_FAST = 100.0
    const T_SIM_BASE = 1000.0
    const T_SIM_SLOW = 8000.0
    const NNETS_RECCHN = 24
    const NNETS_CONDIV = 48

    const TONS_MIN = 0.01
    const TONS_MAX = 0.25
    const TONS_CONDIV = 0.1
end

const 📫 = RemoteChannel(()->Channel{String}(1000))
@async begin
    while true
        ✉️ = take!(📫)
        println(✉️)
        flush(stdout)
    end
end


@everywhere function is_condiv_motif(τ_tuple::Tuple{Vararg{Float64}})
    τchn, τrec, _, _ = τ_parser(τ_tuple)
    return isapprox(τchn, 0.0; atol=1e-12) &&
           isapprox(τrec, 0.0; atol=1e-12)
end

@everywhere function determine_nNets(τ_tuple::Tuple{Vararg{Float64}})
    return is_condiv_motif(τ_tuple) ? NNETS_CONDIV : NNETS_RECCHN
end

@everywhere function determine_T_SIM(τ_tuple::Tuple{Vararg{Float64}})
    τchn, τrec, _, _ = τ_parser(τ_tuple)

    if is_condiv_motif(τ_tuple)
        return T_SIM_BASE
    end

    if τrec < 0.0
        T_SIM = T_SIM_BASE + (T_SIM_BASE - T_SIM_FAST) * τrec / τ_rec_max
    elseif τrec > 0.0
        T_SIM = T_SIM_BASE + (T_SIM_SLOW - T_SIM_BASE) * τrec / τ_rec_max
    else
        T_SIM = T_SIM_BASE
    end

    if τchn < 0.0
        T_SIM += (T_SIM_SLOW - T_SIM_BASE) * (-τchn) / τ_chn_abs_max
    end

    return clamp(T_SIM, T_SIM_FAST, T_SIM_SLOW)
end

@everywhere function determine_tONS(τ_tuple::Tuple{Vararg{Float64}})
    τchn, τrec, _, _ = τ_parser(τ_tuple)

    if is_condiv_motif(τ_tuple)
        return TONS_CONDIV
    end

    if τrec < 0.0
        tONS = TONS_CONDIV + (TONS_CONDIV - TONS_MIN) * τrec / τ_rec_max
    elseif τrec > 0.0
        tONS = TONS_CONDIV + (TONS_MAX - TONS_CONDIV) * τrec / τ_rec_max
    else
        tONS = TONS_CONDIV
    end

    if τchn < 0.0
        tONS += (TONS_MAX - TONS_CONDIV) * (-τchn) / τ_chn_abs_max
    end

    return clamp(tONS, TONS_MIN, TONS_MAX)
end

@everywhere function compute_wrapper(g::Float64, τ_tuple::Tuple{Vararg{Float64}}, log_channel::RemoteChannel)
    
    tONS = determine_tONS(τ_tuple)
    nNets = determine_nNets(τ_tuple)
    T_SIM = determine_T_SIM(τ_tuple)
    put!(log_channel, "Worker $(myid()) starting: g=$(round(g, sigdigits=4)), τ=$(round.(τ_tuple, sigdigits=4)), T=$(round(T_SIM, sigdigits=4)), nNets=$nNets, tONS=$(round(tONS, sigdigits=4))")
    start = now()

    if τ_tuple[2] - 2τ_tuple[4] > 0.4
        burn_in = 1000.0
    elseif τ_tuple[2] - 2τ_tuple[4] > 0.1
        burn_in = 500.0
    else
        burn_in = 100.0
    end

    if τ_tuple[2] < -0.0
        nLE = 500
    else
        nLE = 100
    end

    if (τ_tuple[3] > 0.0 || τ_tuple[4] > 0.0) && abs(τ_tuple[1]) < 0.1
        exclude_bimodal = true
    else
        exclude_bimodal = false
    end

    result = ComputeLSPR(N, J0, g, τ_tuple;
        burn_in=burn_in, 
        T=T_SIM,
        nLE=nLE, 
        n_samples=nNets,
        tONS=tONS, 
        verbose=false,
        return_traj=RETURN_TRAJ, 
        chaos_only=CHAOS_ONLY, 
        return_KSEandKYD=RETURN_HD,
        exclude_bimodal = exclude_bimodal
    )
    put!(log_channel, "Worker $(myid()) finished g=$(round(g, sigdigits=4)), τ=$(round.(τ_tuple, sigdigits=4)), T=$(round(T_SIM, sigdigits=4)), nNets=$nNets, tONS=$(round(tONS, sigdigits=4)). Elapsed = $(format_elapsed(start))")
    return result
end

@everywhere function motif_task_priority(task)
    τchn, τrec, _, _ = τ_parser(task.τ)

    if task.motif === :rec && τrec < 0.0
        return (1, task.τ_index, task.g_index)
    elseif task.motif === :chn && τchn < 0.0
        return (2, task.τ_index, task.g_index)
    elseif task.motif === :rec
        return (3, task.τ_index, task.g_index)
    elseif task.motif === :chn
        return (4, task.τ_index, task.g_index)
    elseif task.motif === :con
        return (5, task.τ_index, task.g_index)
    elseif task.motif === :div
        return (6, task.τ_index, task.g_index)
    else
        return (7, task.τ_index, task.g_index)
    end
end

@everywhere function build_motif_tasks()
    tasks = []

    for (g_index, g_eff) in enumerate(g_eff_vec)
        for (τ_index, τ_val) in enumerate(τ_rec_vec)
            τ_tuple = (0.0, τ_val, 0.0, 0.0)
            g = compute_g(g_eff, τ_tuple)
            push!(tasks, (motif=:rec, τ_index=τ_index, g_index=g_index, g=g, τ=τ_tuple))
        end

        for (τ_index, τ_val) in enumerate(τ_con_vec)
            τ_tuple = (0.0, 0.0, τ_val, 0.0)
            g = compute_g(g_eff, τ_tuple)
            push!(tasks, (motif=:con, τ_index=τ_index, g_index=g_index, g=g, τ=τ_tuple))
        end

        for (τ_index, τ_val) in enumerate(τ_div_vec)
            τ_tuple = (0.0, 0.0, 0.0, τ_val)
            g = compute_g(g_eff, τ_tuple)
            push!(tasks, (motif=:div, τ_index=τ_index, g_index=g_index, g=g, τ=τ_tuple))
        end

        for (τ_index, τ_val) in enumerate(τ_chn_vec)
            τ_abs = abs(τ_val)
            τ_tuple = (τ_val, 0.0, τ_abs, τ_abs)
            g = compute_g(g_eff, τ_tuple)
            push!(tasks, (motif=:chn, τ_index=τ_index, g_index=g_index, g=g, τ=τ_tuple))
        end
    end

    sort!(tasks; by=motif_task_priority)
    return tasks
end

@everywhere function run_motif_scan()
    tasks = build_motif_tasks()

    results = pmap(tasks; batch_size=1) do task
        result = compute_wrapper(task.g, task.τ, 📫)
        return (motif=task.motif, τ_index=task.τ_index, g_index=task.g_index, result=result)
    end

    scan_rec = Array{Any}(undef, length(τ_rec_vec), length(g_eff_vec))
    scan_con = Array{Any}(undef, length(τ_con_vec), length(g_eff_vec))
    scan_div = Array{Any}(undef, length(τ_div_vec), length(g_eff_vec))
    scan_chn = Array{Any}(undef, length(τ_chn_vec), length(g_eff_vec))

    for item in results
        if item.motif === :rec
            scan_rec[item.τ_index, item.g_index] = item.result
        elseif item.motif === :con
            scan_con[item.τ_index, item.g_index] = item.result
        elseif item.motif === :div
            scan_div[item.τ_index, item.g_index] = item.result
        elseif item.motif === :chn
            scan_chn[item.τ_index, item.g_index] = item.result
        else
            throw(ArgumentError("Unknown motif $(item.motif)."))
        end
    end

    return (
        rec = (data=scan_rec, τ=τ_rec_vec,
               T=[determine_T_SIM((0.0, τ_val, 0.0, 0.0)) for τ_val in τ_rec_vec],
               tONS=[determine_tONS((0.0, τ_val, 0.0, 0.0)) for τ_val in τ_rec_vec],
               nNets=[determine_nNets((0.0, τ_val, 0.0, 0.0)) for τ_val in τ_rec_vec]),
        con = (data=scan_con, τ=τ_con_vec,
               T=[determine_T_SIM((0.0, 0.0, τ_val, 0.0)) for τ_val in τ_con_vec],
               tONS=[determine_tONS((0.0, 0.0, τ_val, 0.0)) for τ_val in τ_con_vec],
               nNets=[determine_nNets((0.0, 0.0, τ_val, 0.0)) for τ_val in τ_con_vec]),
        div = (data=scan_div, τ=τ_div_vec,
               T=[determine_T_SIM((0.0, 0.0, 0.0, τ_val)) for τ_val in τ_div_vec],
               tONS=[determine_tONS((0.0, 0.0, 0.0, τ_val)) for τ_val in τ_div_vec],
               nNets=[determine_nNets((0.0, 0.0, 0.0, τ_val)) for τ_val in τ_div_vec]),
        chn = (data=scan_chn, τ=τ_chn_vec,
               T=[determine_T_SIM((τ_val, 0.0, abs(τ_val), abs(τ_val))) for τ_val in τ_chn_vec],
               tONS=[determine_tONS((τ_val, 0.0, abs(τ_val), abs(τ_val))) for τ_val in τ_chn_vec],
               nNets=[determine_nNets((τ_val, 0.0, abs(τ_val), abs(τ_val))) for τ_val in τ_chn_vec])
    )
end


all_data = run_motif_scan()
close(📫)
output_filename = joinpath(@__DIR__, "results/LSHKYDAllMotifs.jld2")

jldsave(output_filename;
    T_SIM_FAST,
    T_SIM_BASE,
    T_SIM_SLOW,
    TONS_MIN,
    TONS_MAX,
    TONS_CONDIV,
    NNETS_RECCHN,
    NNETS_CONDIV,
    N = N,
    g_eff_vec,
    Reciprocal = all_data.rec,
    Convergent = all_data.con,
    Divergent = all_data.div,
    Chain = all_data.chn
)
