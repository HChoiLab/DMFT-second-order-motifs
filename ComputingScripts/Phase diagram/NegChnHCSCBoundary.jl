using LinearAlgebra, FFTW
using Dates
using JLD2

const ROOT_DIR = @__DIR__
const UTILS_PATH = joinpath(ROOT_DIR, "Utils.jl")
const DMFT_PATH = joinpath(ROOT_DIR, "DMFT.jl")


include(UTILS_PATH)
using .Utils
include(DMFT_PATH)
using .DMFT

BLAS.set_num_threads(1)
FFTW.set_num_threads(16)

N = 2000
J0 = 4.0 / N
Nτchn_vec1 = range(0.0, -1.0, 10)
Nτchn_vec2_search = range(-1.1, -3.0, 8)
critical_Nτchn = -1.0
critical_g = 2.0
g_init = 3.9
g_step = 0.05
g_min = 1.85
lc_tol = 1e-5

T = 120.0
dt = 0.05
dmft_max_iter = 500
dmft_damp = 0.7
dmft_tol = 5e-5
dmft_μ0 = 0.0
Δx_seed = 1.0

phi_prime(x) = sech(x)^2

function hcsc_residual(τchn::Float64, g::Float64, N::Int64, J0::Float64, Δx_seed::Float64;
    T::Float64=120.0,
    dt::Float64=0.05,
    dmft_max_iter::Int64=500,
    dmft_damp::Float64=0.7,
    dmft_tol::Float64=5e-5,
    dmft_μ0::Float64=0.0)

    τ = (τchn, 2 * τchn)
    sol = DMFT_Stationary_Solver_B0(N, J0, g, τ;
        ϕ_prime=phi_prime,
        μ0=dmft_μ0,
        Δ0=Δx_seed,
        dt=dt,
        T=T,
        max_iter=dmft_max_iter,
        tol=dmft_tol,
        damp=dmft_damp,
        verbose=false)

    next_Δx_seed = isfinite(sol.Δx) ? max(sol.Δx, 0.0) : Δx_seed
    ϕ_prime_mean = gauss_expectation(phi_prime, sol.mx, max(sol.Δx, 0.0))
    residual = 1.0 - ϕ_prime_mean * (N * J0 + g^2 * (N * τchn) * ϕ_prime_mean)

    return residual, next_Δx_seed
end

function lc_frequency(α::Float64, τchn::Float64, g::Float64, N::Int64, J0::Float64)
    discriminant = (N * J0)^2 + 4.0 * g^2 * (N * τchn)
    return discriminant < 0.0 ? 0.5 * α * sqrt(-discriminant) : NaN
end

function lc_growth_residual(τchn::Float64, g::Float64, N::Int64, J0::Float64, Δx_seed::Float64;
    T::Float64=120.0,
    dt::Float64=0.05,
    dmft_max_iter::Int64=500,
    dmft_damp::Float64=0.7,
    dmft_tol::Float64=5e-5,
    dmft_μ0::Float64=0.0)

    τ = (τchn, 2 * τchn)
    sol = DMFT_Stationary_Solver_B0(N, J0, g, τ;
        ϕ_prime=phi_prime,
        μ0=dmft_μ0,
        Δ0=Δx_seed,
        dt=dt,
        T=T,
        max_iter=dmft_max_iter,
        tol=dmft_tol,
        damp=dmft_damp,
        verbose=false)

    next_Δx_seed = isfinite(sol.Δx) ? max(sol.Δx, 0.0) : Δx_seed
    α = gauss_expectation(phi_prime, sol.mx, max(sol.Δx, 0.0))
    residual = -1.0 + 0.5 * α * (N * J0)
    ω = lc_frequency(α, τchn, g, N, J0)

    return residual, ω, next_Δx_seed
end

function valid_lc_eval(residual::Float64, ω::Float64)
    return isfinite(residual) && isfinite(ω) && ω > 0.0
end

function find_hcsc_bulk_g_boundary(τchn_vec, N::Int64, J0::Float64;
    g_init::Float64=3.9,
    g_step::Float64=0.025,
    g_min::Float64=1.8,
    critical_Nτchn::Float64=-1.0,
    critical_g::Float64=2.0,
    T::Float64=120.0,
    dt::Float64=0.05,
    dmft_max_iter::Int64=500,
    dmft_damp::Float64=0.7,
    dmft_tol::Float64=5e-5,
    dmft_μ0::Float64=0.0,
    Δx_seed::Float64=1.0)

    g_boundary = fill(NaN, length(τchn_vec))
    start = now()
    next_g_init = g_init
    next_Δx_seed = Δx_seed

    for i in eachindex(τchn_vec)
        τchn = Float64(τchn_vec[i])
        g = next_g_init
        residual = NaN

        if N * τchn ≈ critical_Nτchn
            g_boundary[i] = critical_g
            next_g_init = critical_g
            println("Finished N*τchn = $(round(N * τchn, digits=4)) ($(i)/$(length(τchn_vec))); " *
                "g = $(round(g_boundary[i], digits=6)); " *
                "residual = manual critical point; " *
                "elapsed = $(format_elapsed(start))")
            flush(stdout)
            continue
        end

        while g >= g_min
            residual, next_Δx_seed = hcsc_residual(τchn, g, N, J0, next_Δx_seed;
                T=T,
                dt=dt,
                dmft_max_iter=dmft_max_iter,
                dmft_damp=dmft_damp,
                dmft_tol=dmft_tol,
                dmft_μ0=dmft_μ0)

            if isfinite(residual) && residual < 0.0
                g_boundary[i] = g
                next_g_init = g
                break
            end

            g -= g_step
        end

        println("Finished N*τchn = $(round(N * τchn, digits=4)) ($(i)/$(length(τchn_vec))); " *
            "g = $(round(g_boundary[i], digits=6)); " *
            "residual = $(round(residual, digits=6)); " *
            "elapsed = $(format_elapsed(start))")
        flush(stdout)
    end

    return g_boundary
end

function find_hclc_bulk_g_ω_boundary(τchn_vec, N::Int64, J0::Float64;
    g_init::Float64=1.96,
    g_step::Float64=0.025,
    g_min::Float64=1.8,
    g_max::Float64=2.0,
    lc_tol::Float64=5e-4,
    bracket_max_iter::Int64=25,
    bisect_max_iter::Int64=30,
    T::Float64=120.0,
    dt::Float64=0.05,
    dmft_max_iter::Int64=500,
    dmft_damp::Float64=0.7,
    dmft_tol::Float64=5e-5,
    dmft_μ0::Float64=0.0,
    Δx_seed::Float64=1.0)

    g_boundary = fill(NaN, length(τchn_vec))
    ω_boundary = fill(NaN, length(τchn_vec))
    start = now()
    next_g_init = g_init
    next_Δx_seed = Δx_seed

    for i in eachindex(τchn_vec)
        τchn = Float64(τchn_vec[i])
        g_guess = next_g_init
        if i >= 3 && isfinite(g_boundary[i-1]) && isfinite(g_boundary[i-2])
            g_guess = 2.0 * g_boundary[i-1] - g_boundary[i-2]
        elseif i >= 2 && isfinite(g_boundary[i-1])
            g_guess = g_boundary[i-1]
        end

        g = clamp(g_guess, g_min, g_max)
        residual = NaN
        ω = NaN

        evals = NamedTuple{(:g, :residual, :ω),Tuple{Float64,Float64,Float64}}[]
        bracket = nothing

        function eval_at(g_eval::Float64)
            g_eval = clamp(g_eval, g_min, g_max)
            cached_idx = findfirst(item -> isapprox(item.g, g_eval; atol=1e-12, rtol=0.0), evals)
            if cached_idx !== nothing
                return evals[cached_idx]
            end

            residual_eval, ω_eval, next_Δx_seed = lc_growth_residual(τchn, g_eval, N, J0, next_Δx_seed;
                T=T,
                dt=dt,
                dmft_max_iter=dmft_max_iter,
                dmft_damp=dmft_damp,
                dmft_tol=dmft_tol,
                dmft_μ0=dmft_μ0)
            item = (g=g_eval, residual=residual_eval, ω=ω_eval)
            push!(evals, item)
            return item
        end

        function update_bracket!()
            sort!(evals; by=x -> x.g)
            for k in 1:(length(evals)-1)
                left = evals[k]
                right = evals[k+1]
                if valid_lc_eval(left.residual, left.ω) && valid_lc_eval(right.residual, right.ω) &&
                   left.residual * right.residual <= 0.0
                    bracket = (left=left, right=right)
                    return true
                end
            end
            return false
        end

        function accept_item!(item)
            g_boundary[i] = item.g
            ω_boundary[i] = item.ω
            next_g_init = item.g
            residual = item.residual
            ω = item.ω
            return nothing
        end

        function best_valid_eval()
            best = nothing
            for item in evals
                if valid_lc_eval(item.residual, item.ω) &&
                   (best === nothing || abs(item.residual) < abs(best.residual))
                    best = item
                end
            end
            return best
        end

        item0 = eval_at(g)
        residual = item0.residual
        ω = item0.ω

        if valid_lc_eval(item0.residual, item0.ω) && abs(item0.residual) <= lc_tol
            accept_item!(item0)
        else
            g_low = g - g_step
            if g_low >= g_min
                eval_at(g_low)
            end

            g_high = g + g_step
            if g_high <= g_max
                eval_at(g_high)
            end
            update_bracket!()

            width = g_step
            for _ in 1:bracket_max_iter
                bracket !== nothing && break
                width *= 2.0

                g_low = g - width
                if g_low >= g_min
                    eval_at(g_low)
                end

                g_high = g + width
                if g_high <= g_max
                    eval_at(g_high)
                end

                update_bracket!()
            end

            if bracket !== nothing
                left = bracket.left
                right = bracket.right
                best = abs(left.residual) <= abs(right.residual) ? left : right

                for _ in 1:bisect_max_iter
                    denom = right.residual - left.residual
                    g_trial = if isfinite(denom) && abs(denom) > eps(Float64)
                        (left.g * right.residual - right.g * left.residual) / denom
                    else
                        0.5 * (left.g + right.g)
                    end

                    if !isfinite(g_trial) || g_trial <= left.g || g_trial >= right.g
                        g_trial = 0.5 * (left.g + right.g)
                    end

                    trial = eval_at(g_trial)
                    if !valid_lc_eval(trial.residual, trial.ω)
                        trial = eval_at(0.5 * (left.g + right.g))
                        valid_lc_eval(trial.residual, trial.ω) || break
                    end

                    if valid_lc_eval(trial.residual, trial.ω) && abs(trial.residual) < abs(best.residual)
                        best = trial
                    end

                    if valid_lc_eval(trial.residual, trial.ω) && abs(trial.residual) <= lc_tol
                        best = trial
                        break
                    elseif valid_lc_eval(trial.residual, trial.ω) && left.residual * trial.residual <= 0.0
                        right = trial
                    else
                        left = trial
                    end
                end

                residual = best.residual
                ω = best.ω

                if valid_lc_eval(residual, ω) && abs(residual) <= lc_tol
                    accept_item!(best)
                end
            else
                best = best_valid_eval()
                if best !== nothing
                    residual = best.residual
                    ω = best.ω
                end
            end
        end

        println("Finished LC N*τchn = $(round(N * τchn, digits=4)) ($(i)/$(length(τchn_vec))); " *
            "g = $(round(g_boundary[i], digits=6)); " *
            "ω = $(round(ω_boundary[i], digits=6)); " *
            "residual = $(round(residual, digits=6)); " *
            "elapsed = $(format_elapsed(start))")
        flush(stdout)
    end

    return g_boundary, ω_boundary
end

τchn_vec1 = collect(Nτchn_vec1) ./ N
g_boundary1 = find_hcsc_bulk_g_boundary(τchn_vec1, N, J0;
    g_init=g_init,
    g_step=g_step,
    g_min=g_min,
    critical_Nτchn=critical_Nτchn,
    critical_g=critical_g,
    T=T,
    dt=dt,
    dmft_max_iter=dmft_max_iter,
    dmft_damp=dmft_damp,
    dmft_tol=dmft_tol,
    dmft_μ0=dmft_μ0,
    Δx_seed=Δx_seed)

g_init2 = 2.0
g_step2 = 0.005
g_max2 = 2.1
τchn_vec2_search = collect(Nτchn_vec2_search) ./ N
g_boundary2_search, _ = find_hclc_bulk_g_ω_boundary(τchn_vec2_search, N, J0;
    g_init=g_init2,
    g_step=g_step2,
    g_min=g_min,
    g_max=g_max2,
    lc_tol=lc_tol,
    T=T,
    dt=dt,
    dmft_max_iter=dmft_max_iter,
    dmft_damp=dmft_damp,
    dmft_tol=dmft_tol,
    dmft_μ0=dmft_μ0,
    Δx_seed=Δx_seed)

Nτchn_vec2 = vcat(critical_Nτchn, collect(Nτchn_vec2_search))
τchn_vec2 = Nτchn_vec2 ./ N
g_boundary2 = vcat(critical_g, g_boundary2_search)

file_name = "NegChnHCSCBoundaryN$(N)J0$(replace(string(round(N * J0, digits=1)), "." => "p"))B0.jld2"
output_file = joinpath(ROOT_DIR, "results", file_name)
jldsave(output_file;
    N, J0, Nτchn_vec1, Nτchn_vec2, τchn_vec1, τchn_vec2,
    critical_Nτchn, critical_g,
    g_init, g_init2, g_step, g_step2, g_min, g_max2, lc_tol, T,
    dt, dmft_max_iter, dmft_damp, dmft_tol, dmft_μ0,
    g_boundary1, g_boundary2)
