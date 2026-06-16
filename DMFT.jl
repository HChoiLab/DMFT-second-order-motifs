module DMFT
using QuadGK, KrylovKit, Trapz
using FFTW, StatsBase, Roots
using LinearAlgebra, DifferentialEquations
using Random, LoopVectorization
include("Utils.jl")
using .Utils
export DMFT_FP_Gaussian, DMFT_FP_Generic, CreateDMFTRateModel,
    DMFTMainloop, DMFTRateModel, LLE_DMFT,
    DMFT_Stationary_Solver_B0, DMFT_Nonstationary_Solver_B0, 
    BoundaryCurve, gauss_expectation

# Gaussian expectation helper: ⟨f⟩ = ∫ ϕ_N(0,1)(z) f( μ + √Δ z ) dz
function gauss_expectation(f::Function, μ::Real, Δ::Real)
    if Δ < 0
        error("Encountered negative Δ = $Δ")
    end
    σ = sqrt(Δ)
    integrand(z) = exp(-z^2 / 2) / sqrt(2π) * f(μ + σ * z)
    val, _ = quadgk(integrand, -Inf, Inf)
    return val
end

"""
`DMFT_FP_Gaussian` is a static mean-field fixed point solver. It is used as a 
fast solver for the fixed point phase.
"""

function DMFT_FP_Gaussian(J0::Float64, g::Float64, N::Integer,
    τ::Tuple{Vararg{Float64}};
    ϕ::Function=tanh,
    ϕ_prime::Function=x -> 1 - tanh(x)^2,
    μ0::Float64=0.05,
    Δ0::Float64=5.0,
    χ0::Float64=0.2,
    max_iter::Integer=500,
    threshold::Float64=1e-6,
    damping::Float64=0.9,
    verbose::Bool=true,
    adaptive_damping::Bool=true)

    @assert 1e-12 < Δ0 "Δ0 must be positive."
    @assert 0 < damping <= 1 "Damping must be in (0,1]."

    # If the outlier is inside the bulk
    _, λ2 = TheoryOutlier(N, J0, g, τ)
    R = TheoryR(g, τ)

    if real(λ2) < 1 && R <= 1
        return (μ=0.0, Δ=0.0, χ=1.0, converged=true, iters=0)
        # elseif real(λ2) < R && R > 1
        #     error("The network is chaotic.")
    else # The network might have a nonzero FP
        τ_chn, τ_rec, τ_con, τ_div = τ_parser(τ)
        A = 1 - τ_con - τ_div
        B = τ_rec - 2τ_chn
        if verbose
            if abs(B) > 0.1
                @warn "Using this solver for large |B| is not recommended, because the distribution of x is non-Gaussian."
            end
        end

        prev_resid = Inf
        current_damping = damping

        # Step-size capping helper
        cap = (cur, prop, scale) -> begin
            max_step = scale * max(1.0, abs(cur))
            δ = prop - cur
            abs(δ) > max_step ? cur + sign(δ) * max_step : prop
        end

        eps = 1e-12

        μ, Δ, χ = μ0, Δ0, χ0
        for it in 1:max_iter
            # Gaussian averages for current (μ, Δ)
            ϕ_mean = gauss_expectation(ϕ, μ, Δ)
            ϕ2_mean = gauss_expectation(x -> ϕ(x)^2, μ, Δ)
            ϕprime_mean = gauss_expectation(ϕ_prime, μ, Δ)

            # Cap and damp χ step using the current iteration's damping value
            χ_prop = cap(χ, ϕprime_mean, 0.25)
            χ_new = (1 - current_damping) * χ + current_damping * χ_prop

            # μ, Δ updates using χ_new
            μ_prop = N * J0 * ϕ_mean + g^2 * (N * τ_chn + B) * ϕ_mean * χ_new
            Δ_prop = A * g^2 * ϕ2_mean + N * g^2 * τ_con * (ϕ_mean^2)
            Δ_prop = max(Δ_prop, 1e-12)

            μ_prop = cap(μ, μ_prop, 0.25)
            Δ_prop = cap(Δ, Δ_prop, 0.25)

            μ_new = (1 - current_damping) * μ + current_damping * μ_prop
            Δ_new = (1 - current_damping) * Δ + current_damping * Δ_prop

            # Residual including χ
            resid = abs(μ_new - μ) / (abs(μ) + eps)
            resid += abs(Δ_new - Δ) / (abs(Δ) + eps)
            resid += abs(χ_new - χ) / (abs(χ) + eps)

            if resid < threshold
                return (μ=μ_new, Δ=Δ_new, χ=χ_new, converged=true, iters=it)
            end

            old_prev_resid = prev_resid
            μ, Δ, χ = μ_new, Δ_new, χ_new

            # Adapt the proposal weight for the next iteration only.
            if adaptive_damping && it > 1
                if resid > old_prev_resid  # Oscillating or diverging
                    current_damping = max(0.3, current_damping * 0.9)
                elseif resid < 0.5 * old_prev_resid  # Good progress
                    current_damping = min(0.95, current_damping * 1.2)
                end
            end

            prev_resid = resid

            if !isfinite(μ) || !isfinite(Δ) || !isfinite(χ) && verbose
                @warn "Divergence to infinity in DMFT iteration."
                return (μ=μ, Δ=Δ, χ=χ, converged=false, iters=it)
            end
        end
    end
    if verbose
        @warn "Reached max_iter but didn't converge."
    end
    return (μ=μ, Δ=Δ, χ=χ, converged=false, iters=max_iter)
end

"""
`DMFT_FP_Generic` is the generic mean-field fixed point solver for B ≠ 0
"""

function DMFT_FP_Generic(J0::Float64, g::Float64, N::Integer, τ::Tuple{Vararg{Float64}};
    ϕ::Function=tanh,
    ϕ_prime::Function=x -> (1 - tanh(x)^2),
    x_grid=-30.0:0.05:30.0,
    μ0::Float64=0.5,
    Δ0::Float64=1.0,
    χ0::Float64=0.2,
    max_iter::Integer=1000,
    threshold::Float64=1e-7,
    damping::Float64=0.5,
    min_variance::Float64=1e-12,
    verbose::Bool=true,
    return_info::Bool=false)
    0 < damping ≤ 1 || throw(ArgumentError("damping must be in (0, 1]."))
    x_values = collect(Float64.(x_grid))
    length(x_values) ≥ 3 || throw(ArgumentError("x_grid must contain at least 3 points."))
    dxs = diff(x_values)
    all(dxs .> 0) || throw(ArgumentError("x_grid must be strictly increasing."))
    all(isapprox.(dxs, first(dxs); rtol=1e-10, atol=1e-12)) ||
        throw(ArgumentError("x_grid must be evenly spaced."))

    τ_chn, τ_rec, τ_con, τ_div = τ_parser(τ)
    A = 1 - τ_con - τ_div
    B = τ_rec - 2τ_chn
    if τ_con < 0 || τ_div < 0 || abs(τ_chn) > sqrt(τ_con * τ_div) || abs(B) > A
        throw(ArgumentError("Invalid motif strengths."))
    end

    # Use the Gaussian fixed-point DMFT as an initial condition
    fp0 = DMFT_FP_Gaussian(J0, g, N, τ; ϕ=ϕ, ϕ_prime=ϕ_prime,
        μ0=μ0, Δ0=max(Δ0, min_variance), χ0=χ0, max_iter=max_iter,
        threshold=threshold, damping=0.9, verbose=false)

    μ_init = isfinite(fp0.μ) ? fp0.μ : μ0
    Δ_init = max(isfinite(fp0.Δ) ? fp0.Δ : Δ0, min_variance)
    χ = isfinite(fp0.χ) ? fp0.χ : χ0
    ϕ_mean = gauss_expectation(ϕ, μ_init, Δ_init)
    ϕ2_mean = gauss_expectation(x -> ϕ(x)^2, μ_init, Δ_init)

    η_var_init = max(g^2 * A * ϕ2_mean + g^2 * N * τ_con * ϕ_mean^2, min_variance)

    dx = x_values[2] - x_values[1]
    trapz(y) = dx * (sum(y) - 0.5 * (first(y) + last(y)))

    ϕx = ϕ.(x_values)
    ϕpx = ϕ_prime.(x_values)
    p_grid = zeros(Float64, length(x_values))
    converged = false
    η_var = η_var_init
    min_Fprime = Inf
    iters = 0

    for it in 1:max_iter
        iters = it
        η_var = max(g^2 * A * ϕ2_mean + g^2 * N * τ_con * ϕ_mean^2, min_variance)
        drive = (N * J0 + g^2 * N * τ_chn * χ) * ϕ_mean

        F = @. x_values - g^2 * B * χ * ϕx - drive
        Fprime = @. 1 - g^2 * B * χ * ϕpx
        min_Fprime = minimum(Fprime)
        raw_p = @. abs(Fprime) / sqrt(2π * η_var) * exp(-F^2 / (2η_var))
        Z = trapz(raw_p)
        isfinite(Z) && Z > 0 || error("PDF normalization failed. Try a wider x_grid.")
        p_grid .= raw_p ./ Z

        ϕ_mean_prop = trapz(ϕx .* p_grid)
        ϕ2_mean_prop = trapz((ϕx .^ 2) .* p_grid)
        response_integrand = @. ϕpx / (1 - g^2 * B * χ * ϕpx)
        χ_prop = trapz(response_integrand .* p_grid)

        resid = maximum((
            abs(ϕ_mean_prop - ϕ_mean) / (abs(ϕ_mean) + 1e-12),
            abs(ϕ2_mean_prop - ϕ2_mean) / (abs(ϕ2_mean) + 1e-12),
            abs(χ_prop - χ) / (abs(χ) + 1e-12)))

        ϕ_mean = (1 - damping) * ϕ_mean + damping * ϕ_mean_prop
        ϕ2_mean = (1 - damping) * ϕ2_mean + damping * ϕ2_mean_prop
        χ = (1 - damping) * χ + damping * χ_prop

        if resid < threshold
            converged = true
            break
        end
    end

    η_var = max(g^2 * A * ϕ2_mean + g^2 * N * τ_con * ϕ_mean^2, min_variance)
    drive = (N * J0 + g^2 * N * τ_chn * χ) * ϕ_mean
    F = @. x_values - g^2 * B * χ * ϕx - drive
    Fprime = @. 1 - g^2 * B * χ * ϕpx
    min_Fprime = minimum(Fprime)
    raw_p = @. abs(Fprime) / sqrt(2π * η_var) * exp(-F^2 / (2η_var))
    Z = trapz(raw_p)
    isfinite(Z) && Z > 0 || error("PDF normalization failed. Try a wider x_grid.")
    p_grid .= raw_p ./ Z

    if verbose
        converged || @warn "DMFT_FP_Generic reached max_iter without convergence."
        min_Fprime > 0 || @warn "F is not strictly increasing on the grid."
        max(first(p_grid), last(p_grid)) < 1e-5 * maximum(p_grid) ||
            @warn "PDF has non-negligible mass at the grid boundary. Try a wider x_grid."
    end

    if return_info
        return (x=x_values, p=p_grid, ϕ_mean=ϕ_mean, ϕ2_mean=ϕ2_mean,
            χ_int=χ, η_var=η_var, converged=converged, iters=iters,
            min_Fprime=min_Fprime)
    end
    return x_values, p_grid
end

## ============== Jacobian bulk boundary functions ==========================

tanh_phi_prime(x::Float64) = 1 - tanh(x)^2

function boundary_AB(τ::Tuple{Vararg{Float64}})
    tau_chn, tau_rec, tau_con, tau_div = τ_parser(τ)
    A = 1.0 - tau_con - tau_div
    B = tau_rec - 2.0 * tau_chn
    A > 0 || throw(ArgumentError("We require A = 1 - tau_con - tau_div > 0."))
    abs(B) < A || throw(ArgumentError("We require requires |B| < A."))
    return τ, A, B
end

function _boundary_moments(S1::Float64, θ::Float64, τ::Tuple{Vararg{Float64}}, μ0::Float64, Δ0::Float64;
    ϕ_prime::Function=tanh_phi_prime, rtol::Float64=1e-8)
    # S1 = A g^2 ⟨α/K⟩
    _, _, B = boundary_AB(τ)
    c2 = cos(θ)^2
    s2 = sin(θ)^2
    alpha = B * S1 / (A + B)
    beta = B * S1 / (A - B)

    j1 = gauss_expectation(μ0, Δ0) do x
        D = ϕ_prime(x)
        q = c2 * (1 - alpha * D)^2 + s2 * (1 + beta * D)^2
        D / q
    end

    j2 = gauss_expectation(μ0, Δ0) do x
        D = ϕ_prime(x)
        q = c2 * (1 - alpha * D)^2 + s2 * (1 + beta * D)^2
        D^2 / q
    end
    return j1, j2
end

function _boundary_residual(S1::Float64, θ::Float64, τ::Tuple{Vararg{Float64}}, μ0::Float64, Δ0::Float64;
    ϕ_prime::Function=tanh_phi_prime, rtol::Float64=1e-8)
    # S1 = A g^2 ⟨α/K⟩
    j1, j2 = _boundary_moments(S1, θ, τ, μ0, Δ0; ϕ_prime=ϕ_prime, rtol=rtol)
    return S1 * j2 - j1
end


function _solve_boundary_root_from_residual(f::Function, θ::Float64, s_anchor::Float64;
    positive_only::Bool=false, residual_tol::Float64=1e-8, step::Float64=1e-3,
    factor::Float64=2.0, max_iter::Int64=80)

    bisect_checked(a, b) = begin
        s = find_zero(f, (a, b), Bisection())
        abs(f(s)) <= residual_tol || error("Boundary solve failed at θ = $(θ).")
        s
    end

    f_anchor = f(s_anchor)
    if isfinite(f_anchor) && abs(f_anchor) <= residual_tol
        return s_anchor
    end

    if positive_only
        a = 0.0
        fa = f(a)
        isfinite(fa) || error("Residual is not finite at s = 0.")

        b = max(step, s_anchor)
        for _ in 1:max_iter
            fb = f(b)
            if isfinite(fb) && fa * fb <= 0
                return bisect_checked(a, b)
            end
            a = b
            fa = fb
            b *= factor
        end
        error("Could not bracket the positive boundary root at θ = $(θ).")
    end

    width = step
    for _ in 1:max_iter
        a = s_anchor - width
        fa = f(a)
        if isfinite(fa) && fa * f_anchor <= 0
            return bisect_checked(a, s_anchor)
        end

        b = s_anchor + width
        fb = f(b)
        if isfinite(fb) && f_anchor * fb <= 0
            return bisect_checked(s_anchor, b)
        end

        width *= factor
    end
    error("Could not bracket a local boundary root at θ = $(θ).")
end


function _solve_boundary_root(θ::Float64, τ::Tuple{Vararg{Float64}}, μ0::Float64, 
    Δ0::Float64, s_anchor::Float64;
    ϕ_prime::Function=tanh_phi_prime, positive_only::Bool=false,
    residual_tol::Float64=1e-8, step::Float64=1e-3, 
    factor::Float64=2.0, max_iter::Int64=80)

    f(s) = _boundary_residual(s, θ, τ, μ0, Δ0; ϕ_prime=ϕ_prime)
    return _solve_boundary_root_from_residual(f, θ, s_anchor;
        positive_only=positive_only, residual_tol=residual_tol,
        step=step, factor=factor, max_iter=max_iter)
end

function trivial_boundary_curve(τ::Tuple{Vararg{Float64}}; g::Float64=1.0,
     n_theta::Int64=100)
    _, A, B = boundary_AB(τ)
    a = g * (A + B) / sqrt(A)
    b = g * (A - B) / sqrt(A)
    theta_grid = range(0.0, 2pi; length=4 * n_theta + 1)
    xb = a .* cos.(theta_grid)
    yb = b .* sin.(theta_grid)
    return xb, yb
end


function _assemble_boundary_curve(A::Float64, g::Float64, τ::Tuple{Vararg{Float64}},
    moments_fn::Function, solve_root_fn::Function;
    n_theta::Int64=100, theta_eps::Float64=5e-3)

    n_quadrant = n_theta + 2
    theta_vals = collect(range(theta_eps, pi / 2 - theta_eps; length=n_theta))
    xq = Vector{Float64}(undef, n_quadrant)
    yq = Vector{Float64}(undef, n_quadrant)

    S1 = solve_root_fn(0.0, 1.0; positive_only=true)
    _, j2 = moments_fn(S1, 0.0)

    xq[1] = g * sqrt(A * j2)
    yq[1] = 0.0

    for (idx, theta) in enumerate(theta_vals)
        S1 = solve_root_fn(theta, S1)
        _, j2 = moments_fn(S1, theta)
        r = g * sqrt(A * j2)
        xq[idx + 1] = r * cos(theta)
        yq[idx + 1] = r * sin(theta)
    end

    S1 = solve_root_fn(pi / 2, S1)
    _, j2 = moments_fn(S1, pi / 2)
    xq[end] = 0.0
    yq[end] = g * sqrt(A * j2)

    n_boundary = 4 * n_quadrant + 1
    xb = Vector{Float64}(undef, n_boundary)
    yb = Vector{Float64}(undef, n_boundary)
    xb[1:n_quadrant] = xq
    yb[1:n_quadrant] = yq
    xb[n_quadrant + 1:2 * n_quadrant] = -xq[end:-1:1]
    yb[n_quadrant + 1:2 * n_quadrant] = yq[end:-1:1]
    xb[2 * n_quadrant + 1:3 * n_quadrant] = -xq
    yb[2 * n_quadrant + 1:3 * n_quadrant] = -yq
    xb[3 * n_quadrant + 1:4 * n_quadrant] = xq[end:-1:1]
    yb[3 * n_quadrant + 1:4 * n_quadrant] = -yq[end:-1:1]
    xb[end] = xq[1]
    yb[end] = yq[1]
    return xb, yb
end

function collect_generic_fp_derivatives(N::Int64, J0::Float64, τ::Tuple{Vararg{Float64}}, g::Float64;
    ϕ::Function=tanh, ϕ_prime::Function=tanh_phi_prime, x_grid=-30.0:0.05:30.0,
    μ0::Float64=0.5, Δ0::Float64=1.0, χ0::Float64=0.2,
    max_iter::Int64=1000, threshold::Float64=1e-7, damping::Float64=0.5,
    zero_tol::Float64=1e-6, verbose::Bool=false)

    τ_tuple, _, _ = boundary_AB(τ)
    fp_dist = DMFT_FP_Generic(J0, g, N, τ_tuple; ϕ=ϕ, ϕ_prime=ϕ_prime,
        x_grid=x_grid, μ0=μ0, Δ0=Δ0, χ0=χ0, max_iter=max_iter,
        threshold=threshold, damping=damping, verbose=verbose, return_info=true)

    fp_dist.converged || error("DMFT_FP_Generic did not converge to a fixed-point distribution.")

    x = fp_dist.x
    p = fp_dist.p

    p_norm_factor = Trapz.trapz(x, p)
    isfinite(p_norm_factor) && p_norm_factor > 0 ||
        error("The generic fixed-point PDF has invalid normalization.")
    p_norm = p ./ p_norm_factor
    mean_abs_x = Trapz.trapz(x, abs.(x) .* p_norm)
    if mean_abs_x <= zero_tol
        return (d=Float64[], weights=Float64[])
    end

    dx = x[2] - x[1]
    weights = dx .* p_norm
    weights[begin] *= 0.5
    weights[end] *= 0.5

    return (d=ϕ_prime.(x), weights=weights)
end


function _generic_boundary_moments(S1::Float64, θ::Float64, τ::Tuple{Vararg{Float64}},
    d_values::Vector{Float64}, weights::Vector{Float64})

    _, A, B = boundary_AB(τ)
    c2 = cos(θ)^2
    s2 = sin(θ)^2
    alpha = B * S1 / (A + B)
    beta = B * S1 / (A - B)

    q(d) = c2 * (1 - alpha * d)^2 + s2 * (1 + beta * d)^2

    j1 = sum(w * d / q(d) for (d, w) in zip(d_values, weights))
    j2 = sum(w * d^2 / q(d) for (d, w) in zip(d_values, weights))
    return j1, j2
end

function _generic_boundary_residual(S1::Float64, θ::Float64, τ::Tuple{Vararg{Float64}},
    d_values::Vector{Float64}, weights::Vector{Float64})
    j1, j2 = _generic_boundary_moments(S1, θ, τ, d_values, weights)
    return S1 * j2 - j1
end

function _solve_boundary_root_generic(θ::Float64, τ::Tuple{Vararg{Float64}},
    d_values::Vector{Float64}, weights::Vector{Float64}, s_anchor::Float64;
    positive_only::Bool=false, residual_tol::Float64=1e-8, step::Float64=1e-3,
    factor::Float64=2.0, max_iter::Int64=80)

    f(s) = _generic_boundary_residual(s, θ, τ, d_values, weights)
    return _solve_boundary_root_from_residual(f, θ, s_anchor;
        positive_only=positive_only, residual_tol=residual_tol,
        step=step, factor=factor, max_iter=max_iter)
end

function gaussian_boundary_curve(N::Int64, J0::Float64, τ::Tuple{Vararg{Float64}}, g::Float64;
    n_theta::Int64=100, theta_eps::Float64=5e-3, ϕ::Function=tanh,
    ϕ_prime::Function=tanh_phi_prime)

    τ_tuple, A, B = boundary_AB(τ)
    if abs(B) > 0.1
        @warn "The boundary curve is inaccurate when using this function for large |B|.
        Consider using `generic_boundary_curve` instead."
    end
    fp_info = DMFT_FP_Gaussian(J0, g, N, τ_tuple; μ0=0.1, Δ0=0.1, damping=0.9, ϕ=ϕ, threshold=1e-4,
        ϕ_prime=ϕ_prime, verbose=false, max_iter=5000)

    if iszero(fp_info.μ) && iszero(fp_info.Δ)
        return trivial_boundary_curve(τ_tuple; g=g, n_theta=n_theta)
    end

    fp_info.converged || error("DMFT_FP_Gaussian did not converge to a nontrivial fixed point.")
    mu_fp = Float64(fp_info.μ)
    Delta_fp = Float64(fp_info.Δ)

    moments_fn = (S1, theta) -> _boundary_moments(S1, theta, τ_tuple, mu_fp, Delta_fp; ϕ_prime=ϕ_prime)
    solve_root_fn = (theta, s_anchor; positive_only=false) -> _solve_boundary_root(
        theta, τ_tuple, mu_fp, Delta_fp, s_anchor;
        ϕ_prime=ϕ_prime, positive_only=positive_only)

    return _assemble_boundary_curve(A, g, τ_tuple, moments_fn, solve_root_fn;
        n_theta=n_theta, theta_eps=theta_eps)
end


function generic_boundary_curve(N::Int64, J0::Float64, τ::Tuple{Vararg{Float64}}, g::Float64;
    n_theta::Int64=100, theta_eps::Float64=5e-3, ϕ::Function=tanh,
    ϕ_prime::Function=tanh_phi_prime, x_grid=-30.0:0.05:30.0,
    μ0::Float64=0.5, Δ0::Float64=1.0, χ0::Float64=0.2,
    fp_max_iter::Int64=1000, fp_threshold::Float64=1e-7,
    fp_damping::Float64=0.5, zero_tol::Float64=1e-6)

    τ_tuple, A, _ = boundary_AB(τ)
    fp_derivatives = collect_generic_fp_derivatives(N, J0, τ_tuple, g;
        ϕ=ϕ, ϕ_prime=ϕ_prime, x_grid=x_grid, μ0=μ0, Δ0=Δ0, χ0=χ0,
        max_iter=fp_max_iter, threshold=fp_threshold, damping=fp_damping,
        zero_tol=zero_tol)

    if isempty(fp_derivatives.d)
        return trivial_boundary_curve(τ_tuple; g=g, n_theta=n_theta)
    end

    d_values = fp_derivatives.d
    weights = fp_derivatives.weights
    moments_fn = (S1, theta) -> _generic_boundary_moments(S1, theta, τ_tuple, d_values, weights)
    solve_root_fn = (theta, s_anchor; positive_only=false) -> _solve_boundary_root_generic(
        theta, τ_tuple, d_values, weights, s_anchor; positive_only=positive_only)

    return _assemble_boundary_curve(A, g, τ_tuple, moments_fn, solve_root_fn;
        n_theta=n_theta, theta_eps=theta_eps)
end

function BoundaryCurve(N::Int64, J0::Float64, τ::Tuple{Vararg{Float64}}, g::Float64;
    n_theta::Int64=100, theta_eps::Float64=5e-3, ϕ::Function=tanh,
    ϕ_prime::Function=tanh_phi_prime, boundary_method::Symbol=:generic,
    x_grid=-30.0:0.05:30.0, μ0::Float64=0.5, Δ0::Float64=1.0,
    χ0::Float64=0.2, fp_max_iter::Int64=1000, fp_threshold::Float64=1e-7,
    fp_damping::Float64=0.5, zero_tol::Float64=1e-6)

    _, _, B = boundary_AB(τ)
    if abs(B) > 0.1 && boundary_method == :gaussian
        @warn "The boundary curve is inaccurate when using the gaussian method for large |B|.
        Consider using `boundary_method=:generic` instead."
    end

    if boundary_method == :gaussian
        return gaussian_boundary_curve(N, J0, τ, g;
            n_theta=n_theta, theta_eps=theta_eps, ϕ=ϕ, ϕ_prime=ϕ_prime)
    elseif boundary_method == :generic
        return generic_boundary_curve(N, J0, τ, g;
            n_theta=n_theta, theta_eps=theta_eps, ϕ=ϕ, ϕ_prime=ϕ_prime,
            x_grid=x_grid, μ0=μ0, Δ0=Δ0, χ0=χ0, fp_max_iter=fp_max_iter,
            fp_threshold=fp_threshold, fp_damping=fp_damping, zero_tol=zero_tol)
    else
        error("Unknown boundary_method = $(boundary_method). Use :gaussian or :generic.")
    end
end


## ========== DMFT nonstationary CPU prototype version ==============
# This version is equivalent to the CUDA nonstationary version in the DMFT_CUDA module, 
# but is easier to understand. Adpated from Zou and Huang 2024.

# One can use this algorithm for non-stationary dynamics for B ≠ 0. 
# We didn't use it in the paper.

"""
The following code implements the DMFT iteration loop.
"""

struct DMFTRateModel
    N::Int32
    J0::Float32
    g::Float32
    τchn::Float32
    τrec::Float32
    τcon::Float32
    τdiv::Float32
    Ttot::Float32
    dt::Float32
    nTime::Int32
    traj_init_μx::Float32
    traj_init_σx::Float32
    nTraj_list::Tuple{Vararg{Int32}}
    nIte_list::Tuple{Vararg{Int32}}
    damp_R::Tuple{Vararg{Float32}}
    damp_C::Tuple{Vararg{Float32}}
    traj_stride::Int32
    threshold::Float32
    ϕ::Function
    ϕ_prime::Function
end

function CreateDMFTRateModel(N::Int,
    J0::Real, g::Real, τ::Tuple{Vararg{Float64}},
    Ttot::Real,
    nIte_list::Tuple{Vararg{Int}},
    nTraj_list::Tuple{Vararg{Int}},
    damp_R::Tuple{Vararg{Float64}},
    damp_C::Tuple{Vararg{Float64}};
    dt::Real=0.1,
    traj_stride::Int=2,
    threshold::Real=1e-5,
    traj_init_μx::Real=0.0,
    traj_init_σx::Real=sqrt(0.1),
    ϕ::Function=tanh,
    ϕ_prime::Function=x -> 1 - tanh(x)^2
)

    if traj_init_σx < 0
        throw(ArgumentError("traj_init_σx must be nonnegative. Provided traj_init_σx=$traj_init_σx."))
    end

    τchn, τrec, τcon, τdiv = τ_parser(τ) .|> Float32

    N = Int32(N)
    nTime = Int32(round(Ttot / dt))
    J0 = Float32(J0)
    g = Float32(g)
    Ttot = Float32(Ttot)
    traj_init_μx = Float32(traj_init_μx)
    traj_init_σx = Float32(traj_init_σx)
    nIte_list = Int32.(nIte_list)
    nTraj_list = Int32.(nTraj_list)
    dt = Float32(dt)
    damp_R = Float32.(damp_R)
    damp_C = Float32.(damp_C)
    traj_stride = Int32.(traj_stride)
    threshold = Float32(threshold)

    return DMFTRateModel(N, J0, g,
        τchn, τrec, τcon, τdiv,
        Ttot, dt, nTime, traj_init_μx, traj_init_σx,
        nTraj_list, nIte_list,
        damp_R, damp_C, traj_stride, threshold,
        ϕ, ϕ_prime)
end

function sample_η!(η::Matrix{Float32},
    Cη::Matrix{Float32}, Cη_work::Matrix{Float32}, Z::Matrix{Float32},
    model::DMFTRateModel, Cϕ::Matrix{Float32}, mϕ::Vector{Float32};
    shift_rtol::Float32=1f-6, shift_growth::Float32=8f0, shift_max::Float32=1f-2
)
    A = 1f0 - model.τcon - model.τdiv

    copyto!(Cη, Cϕ)
    rmul!(Cη, A * model.g^2)

    if model.τcon != 0f0
        BLAS.ger!(model.N * model.g^2 * model.τcon,
            mϕ, mϕ, Cη)
    end

    maxabs = mapreduce(abs, max, Cη)
    isfinite(maxabs) || error("Cη contains NaN or Inf.")

    base_shift = shift_rtol * max(1f0, maxabs)

    shift = 0f0
    while true
        copyto!(Cη_work, Cη)
        Cη_work[diagind(Cη_work)] .+= shift
        F = cholesky!(Symmetric(Cη_work), check=false)
        if issuccess(F)
            randn!(Z)
            mul!(η, Z, transpose(F.L))
            return η
        end

        if shift == shift_max
            error("Cholesky failed up to diagonal shift = $(shift_max)")
        end

        shift = (shift == 0f0) ? base_shift : min(shift * shift_growth, shift_max)
    end
end

function integ_xtraj!(x_traj::Matrix{Float32}, ϕx::Matrix{Float32},
    mx_t::Vector{Float32}, mϕ_t::Vector{Float32}, f::Vector{Float32},
    temp::Vector{Float32}, new_temp::Vector{Float32}, model::DMFTRateModel,
    χϕ::Matrix{Float32}, η::Matrix{Float32}
)

    _, nTime = size(x_traj)

    randn!(temp)

    @. temp = model.traj_init_σx * temp + model.traj_init_μx

    x_traj[:, 1] .= temp
    ϕx[:, 1] .= model.ϕ.(temp)

    mx_t[1] = mean(temp)
    mϕ_t[1] = mean(@view ϕx[:, 1])

    B = model.τrec - 2f0 * model.τchn

    C1 = model.dt * model.g^2 * B
    C2 = model.dt * model.N * model.τchn * model.g^2
    C3 = model.N * model.J0

    α = exp(-model.dt)
    β = 1f0 - α

    @inbounds for t in 1:(nTime-1)

        @views f .= η[:, t]
        if t > 1
            w = @view χϕ[t, 1:(t-1)]
            Φ = @view ϕx[:, 1:(t-1)]
            mul!(f, Φ, w, C1, 1f0)
            scalar_add = C2 * dot(w, @view mϕ_t[1:(t-1)]) +
                         C3 * mϕ_t[t]
            @. f += scalar_add
        else
            @. f += C3 * mϕ_t[t]
        end

        @. new_temp = α * temp + β * f

        x_traj[:, t+1] .= new_temp
        ϕx[:, t+1] .= model.ϕ.(new_temp)

        mx_t[t+1] = mean(new_temp)
        mϕ_t[t+1] = mean(@view ϕx[:, t+1])

        copyto!(temp, new_temp)
    end
end


@inline function _dot_col_segment(χϕ_prev_CM::Matrix{Float32},
    col::Int, col_buf::Vector{Float32}, lo::Int, hi::Int
)
    s = 0f0
    @turbo for k in lo:hi
        s += χϕ_prev_CM[k, col] * col_buf[k]
    end
    return s
end

function integ_χ_χϕ!(χ::Matrix{Float32},
    χϕ::Matrix{Float32}, ϕ_prime_x::Matrix{Float32},
    χϕ_col_buf::Vector{Vector{Float32}}, model::DMFTRateModel,
    x_traj::Matrix{Float32}, χϕ_prev_CM::Matrix{Float32};
    traj_stride::Int32=Int32(2), traj_offset::Int32=Int32(1)
)

    nTraj, nTime = size(x_traj)
    off = mod1(traj_offset, traj_stride)
    n_use = length(off:traj_stride:nTraj)

    fill!(χ, 0f0)
    fill!(χϕ, 0f0)

    @. ϕ_prime_x = model.ϕ_prime(x_traj)

    B = model.τrec - 2f0 * model.τchn

    C1 = model.dt * model.g^2 * B

    α = exp(-model.dt)
    β = 1f0 - α

    inv_use = 1f0 / Float32(n_use)

    Threads.@threads :static for t2 in 1:(nTime-1)
        tid = Threads.threadid()
        col_buf = χϕ_col_buf[tid]

        colχ = @view χ[:, t2]
        colϕ = @view χϕ[:, t2]

        for j in off:traj_stride:nTraj

            ϕp = @view ϕ_prime_x[j, :]
            temp = 1f0
            col_buf[t2] = 0f0
            vϕ = ϕp[t2+1]

            col_buf[t2+1] = vϕ

            colχ[t2+1] += temp
            colϕ[t2+1] += vϕ

            @inbounds for t1 in (t2+1):(nTime-1)
                add = _dot_col_segment(χϕ_prev_CM,
                    t1, col_buf, t2, t1)

                temp = α * temp + β * C1 * add
                vϕ = temp * ϕp[t1+1]

                col_buf[t1+1] = vϕ
                colχ[t1+1] += temp
                colϕ[t1+1] += vϕ
            end
        end

        @turbo for t1 in (t2+1):nTime
            colχ[t1] *= inv_use
            colϕ[t1] *= inv_use
        end
    end
end

# --- Blocked GEMM Version ---
function integ_χ_χϕ!(χ::Matrix{Float32},
    χϕ::Matrix{Float32}, ϕ_prime_x::Matrix{Float32},
    V_buf::Vector{Matrix{Float32}}, H_buf::Vector{Matrix{Float32}},
    temp_buf::Vector{Vector{Float32}}, add_buf::Vector{Vector{Float32}},
    w_buf::Vector{Vector{Float32}},
    model::DMFTRateModel, x_traj::Matrix{Float32}, χϕ_prev::Matrix{Float32};
    traj_stride::Int32=Int32(2), traj_offset::Int32=Int32(1), B::Int=64
)
    nTraj, nTime = size(x_traj)
    off = mod1(traj_offset, traj_stride)
    n_use = length(off:traj_stride:nTraj)

    fill!(χ, 0f0)
    fill!(χϕ, 0f0)

    @. ϕ_prime_x = model.ϕ_prime(x_traj)
    ϕp_T = Matrix(transpose(@view ϕ_prime_x[off:traj_stride:end, :]))

    B_val = model.τrec - 2f0 * model.τchn
    C1 = model.dt * model.g^2 * B_val
    α = exp(-model.dt)
    β = 1f0 - α
    inv_use = 1f0 / Float32(n_use)

    Threads.@threads :static for t2 in 1:(nTime-1)
        tid = Threads.threadid()

        V_tid = V_buf[tid]
        H_tid = H_buf[tid]
        temp_tid = view(temp_buf[tid], 1:n_use)
        add_tid = view(add_buf[tid], 1:n_use)
        w_buf_tid = w_buf[tid]

        fill!(temp_tid, 1f0)

        V_t2 = view(V_tid, t2, 1:n_use)
        fill!(V_t2, 0f0)

        ϕp_start = view(ϕp_T, t2 + 1, 1:n_use)
        V_start = view(V_tid, t2 + 1, 1:n_use)
        copyto!(V_start, ϕp_start)

        χ[t2+1, t2] = Float32(n_use)
        χϕ[t2+1, t2] = sum(ϕp_start)

        for t_start in (t2+1):B:(nTime-1)
            t_end = min(t_start + B - 1, nTime - 1)
            L_chunk = t_end - t_start + 1

            # History chunk from k = t2:(t_start-1)
            W_hist = view(χϕ_prev, t_start:t_end, t2:(t_start-1))
            V_past = view(V_tid, t2:(t_start-1), 1:n_use)
            H_view = view(H_tid, 1:L_chunk, 1:n_use)
            mul!(H_view, W_hist, V_past)

            # Local chunk history from k = t_start:t1
            for t1 in t_start:t_end
                h_idx = t1 - t_start + 1
                copyto!(add_tid, view(H_tid, h_idx, 1:n_use))

                len = h_idx
                w_loc = view(w_buf_tid, 1:len)
                V_loc = view(V_tid, t_start:t1, 1:n_use)

                copyto!(w_loc, view(χϕ_prev, t1, t_start:t1))
                mul!(add_tid, transpose(V_loc), w_loc, 1f0, 1f0)

                ϕp_next = view(ϕp_T, t1 + 1, 1:n_use)
                V_next = view(V_tid, t1 + 1, 1:n_use)

                @. temp_tid = α * temp_tid + β * C1 * add_tid
                @. V_next = temp_tid * ϕp_next

                χ[t1+1, t2] = sum(temp_tid)
                χϕ[t1+1, t2] = sum(V_next)
            end
        end

        # Normalize the column average
        χ[(t2+1):nTime, t2] .*= inv_use
        χϕ[(t2+1):nTime, t2] .*= inv_use
    end
end

function _nearest_posdef!(A::Matrix{Float32}; ε::Float32=1f-5)
    N = size(A, 1)
    x0 = rand(Float32, N)

    # Use `eigen` may cause overflow
    # Time complexity O(k N^2) 
    # Memory O(k N), k ≪ N
    vals, _, info = eigsolve(Symmetric(A), x0, 1, :SR;
        issymmetric=true, tol=1f-3, krylovdim=100,
        maxiter=300)

    # Check for convergence
    margin = 1.02f0
    if info.converged < 1
        @warn "KrylovKit failed to converge. Info: iterations = $(info.numiter), residual = $(info.normres[1])."
        margin = 1.10f0
    end

    λ_min = real(vals[1])

    # Shift diagonal if the smallest eigenvalue is less than epsilon
    if λ_min < ε
        shift = (ε - λ_min) * margin
        A[diagind(A)] .+= shift
    end
    return A # Note that A is not necessarily symmetric
end

# function initialize_mC(model::DMFTRateModel;
#     T::Float32=200f0, burn::Float32=50f0, n_samples::Int64=36)

#     τ = (model.τchn, model.τrec, model.τcon, model.τdiv)

#     J_array = [Matrix{Float32}(CreateJ(Int64(model.N), Float64(model.J0),
#         Float64(model.g), τ)) for _ in 1:n_samples]

#     tspan = (0f0, T)
#     M = model.nTime

#     function prob_func(prob, i, repeat)
#         u0 = 0.5f0 .* randn(Float32, model.N)
#         cache = zeros(Float32, model.N) # Thread-local cache
#         return remake(prob, u0=u0, p=(J_array[i], cache))
#     end

#     function ODEFunc!(du, u, p, t)
#         J, ϕu = p
#         @. ϕu = model.ϕ(u)
#         mul!(du, J, ϕu)
#         du .-= u
#         return nothing
#     end

#     function output_func(sol, i)
#         times = sol.t
#         burn_idx = findfirst(>=(burn), times)
#         X = reduce(hcat, @view sol.u[burn_idx:burn_idx+M-1])
#         ϕX = model.ϕ.(X)
#         mx = vec(mean(X, dims=1))
#         mϕ = vec(mean(ϕX, dims=1))
#         Cx = zeros(Float32, M, M)
#         Cϕ = zeros(Float32, M, M)
#         mul!(Cx, transpose(X), X, 1f0 / N, 0f0)
#         mul!(Cϕ, transpose(ϕX), ϕX, 1f0 / N, 0f0)
#         return ((mx, mϕ, Cx, Cϕ), false)
#     end

#     prob = ODEProblem(ODEFunc!, zeros(Float32, model.N),
#         tspan, (zeros(Float32, model.N, model.N), zeros(Float32, model.N)))

#     ensemble_problem = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)

#     sim = solve(ensemble_problem, Tsit5(), EnsembleThreads();
#         trajectories=n_samples, saveat=model.dt)

#     mx_mean = zeros(Float32, M)
#     mϕ_mean = zeros(Float32, M)
#     Cx_mean = zeros(Float32, M, M)
#     Cϕ_mean = zeros(Float32, M, M)

#     for i in 1:n_samples
#         mx, mϕ, Cx, Cϕ = sim.u[i]
#         mx_mean .+= mx
#         mϕ_mean .+= mϕ
#         Cx_mean .+= Cx
#         Cϕ_mean .+= Cϕ
#     end

#     mx_mean ./= n_samples
#     mϕ_mean ./= n_samples
#     Cx_mean ./= n_samples
#     Cϕ_mean ./= n_samples

#     # Enforce positive-definiteness
#     _nearest_posdef!(Cx_mean)
#     _nearest_posdef!(Cϕ_mean)

#     return mx_mean, mϕ_mean, Cx_mean, Cϕ_mean
# end

function DMFTMainloop(model::DMFTRateModel; verbose::Bool=true)

    if 1.0 - 2model.τcon - 2model.τdiv - abs(model.τrec) < 0
        return nothing, nothing, nothing, nothing, nothing, nothing
    end

    nT = model.nTime
    # Initial guess

    # Tinit = min(1200f0, model.Ttot * 3)
    # mx, mϕ, Cx, Cϕ = initialize_mC(model; T=Tinit, burn=Tinit / 5)

    mx = zeros(Float32, nT)
    mϕ = zeros(Float32, nT)
    Cx = 0.1 .* I(nT) |> Matrix{Float32}
    Cϕ = 0.1 .* I(nT) |> Matrix{Float32}
    χ = diagm(-1 => ones(nT - 1)) |> Matrix{Float32}
    χϕ = diagm(-1 => ones(nT - 1)) |> Matrix{Float32}
    χϕ = χϕ .* 0.2f0

    Ite_count = 0
    nSteps = length(model.nIte_list)

    new_mx = similar(mx)
    new_mϕ = similar(mϕ)

    new_Cx = similar(Cx)
    new_Cϕ = similar(Cϕ)

    new_χ = similar(χ)
    new_χϕ = similar(χϕ)

    χϕ_prev = copy(χϕ)

    Cη = zeros(Float32, nT, nT)
    Cη_work = similar(Cη)
    mx_t = Vector{Float32}(undef, nT)
    mϕ_t = Vector{Float32}(undef, nT)

    for block in 1:nSteps
        nIte = model.nIte_list[block]
        nTraj = model.nTraj_list[block]
        damp_R = model.damp_R[block]
        damp_C = model.damp_C[block]

        η = zeros(Float32, nTraj, nT)
        Z = similar(η)

        x_traj = zeros(Float32, nTraj, nT)
        ϕx = similar(x_traj)
        ϕp = similar(x_traj)

        f = Vector{Float32}(undef, nTraj)
        temp = similar(f)
        new_temp = similar(f)

        B_size = 64
        V_buf = [zeros(Float32, nT, nTraj) for _ in 1:Threads.nthreads()]
        H_buf = [zeros(Float32, B_size, nTraj) for _ in 1:Threads.nthreads()]
        temp_buf = [zeros(Float32, nTraj) for _ in 1:Threads.nthreads()]
        add_buf = [zeros(Float32, nTraj) for _ in 1:Threads.nthreads()]
        w_buf = [zeros(Float32, B_size) for _ in 1:Threads.nthreads()]

        cov_factor = (1f0 - damp_C) / Float32(nTraj)

        for _ in 1:nIte
            Ite_count += 1

            # integrate x-trajectories
            sample_η!(η, Cη, Cη_work, Z, model, Cϕ, mϕ)
            integ_xtraj!(x_traj, ϕx, mx_t, mϕ_t, f, temp, new_temp,
                model, χϕ, η)

            # update m and C
            new_mx .= damp_C .* mx .+
                      (1f0 - damp_C) .* vec(mean(x_traj; dims=1))
            new_mϕ .= damp_C .* mϕ .+
                      (1f0 - damp_C) .* vec(mean(ϕx; dims=1))

            # Cx update
            copyto!(new_Cx, Cx)
            rmul!(new_Cx, damp_C)
            mul!(new_Cx, transpose(x_traj), x_traj, cov_factor, 1f0)

            # Cϕ update
            copyto!(new_Cϕ, Cϕ)
            rmul!(new_Cϕ, damp_C)
            mul!(new_Cϕ, transpose(ϕx), ϕx, cov_factor, 1f0)

            # responses χ, χϕ
            # Use a subset of trajectories cyclically ^_^
            integ_χ_χϕ!(new_χ, new_χϕ, ϕp,
                V_buf, H_buf, temp_buf, add_buf, w_buf,
                model, x_traj, χϕ_prev;
                traj_stride=model.traj_stride, traj_offset=mod1(Ite_count, model.traj_stride), B=B_size)

            @. new_χ = damp_R * χ + (1f0 - damp_R) * new_χ
            @. new_χϕ = damp_R * χϕ + (1f0 - damp_R) * new_χϕ

            q_old = diag(Cϕ)
            q_new = diag(new_Cϕ)

            diff_q = norm(q_new - q_old) / max(norm(q_old), eps(Float32)) / (1f0 - damp_C)

            if verbose
                println(
                    "Iteration $Ite_count, " *
                    "normalized |ΔC| = $(diff_q)"
                )
                flush(stdout)
            end

            if diff_q < model.threshold
                return new_mx, new_mϕ, new_Cx, new_Cϕ, new_χ, new_χϕ
            end

            mx .= new_mx
            mϕ .= new_mϕ
            Cx .= new_Cx
            Cϕ .= new_Cϕ
            χ .= new_χ
            χϕ .= new_χϕ
            copyto!(χϕ_prev, χϕ)
        end
    end

    if verbose
        @warn "DMFT loop didn't converge."
    end

    return new_mx, new_mϕ, new_Cx, new_Cϕ, new_χ, new_χϕ

end

# ================= Nonstationary (B ≠ 0) CPU prototype version end =======================

## Solve the stationary DMFT with B = 0.0
function _compute_transfer_function_unitary!(
    χx_h::Vector{ComplexF64},
    χϕ_h::Vector{ComplexF64},
    ω::Vector{Float64},
    α::Float64
)
    @. χx_h = 1.0 / (sqrt(2π) * (1.0 + im * ω))
    @. χϕ_h = α * χx_h
end

function _estimate_covariance_plateau(Kx::Vector{Float64})
    n = length(Kx)
    half_window = min(n ÷ 32, 8)
    center = fld(n, 2) + 1
    lo = max(1, center - half_window)
    hi = min(n, center + half_window)
    q = mean(@view Kx[lo:hi])
    Δx = max(Kx[1], 0.0)
    return clamp(q, 0.0, Δx)
end

"""
Solve the stationary DMFT in the frequency domain for the Gaussian B = 0 case.
"""
function DMFT_Stationary_Solver_B0(N::Int64,
    J0::Float64,
    g::Float64,
    τ::Tuple{Vararg{Float64}};
    ϕ::Function=tanh,
    ϕ_prime::Function=x -> 1 - tanh(x)^2,
    μ0::Union{Nothing,Float64}=nothing,
    Δ0::Float64=1.0,
    dt::Float64=0.1,
    T::Float64=100.0,
    max_iter::Int=500,
    tol::Float64=5e-5,
    damp::Float64=0.8,
    verbose::Bool=true,
    minimal_return::Bool=false,
    n_quad::Int64=128
)
    τchn, τrec, τcon, τdiv = τ_parser(τ)

    A = 1.0 - τcon - τdiv
    B = τrec - 2.0 * τchn
    if abs(B) > 1e-5
        throw(ArgumentError("🚫 `DMFT_Stationary_Solver_B0` only works for B = τrec - 2τchn = 0."))
    end

    nTime = round(Int, T / dt)
    t_grid = collect((0:nTime-1) .* dt)

    ω_r = collect(2π .* rfftfreq(nTime, 1.0 / dt))
    nFreq = length(ω_r)
    ω = collect(2π .* fftfreq(nTime, 1.0 / dt))

    Kx = exp.(-min.(t_grid, T .- t_grid)) .* Δ0 # Cx - μx^2
    Kx_new = similar(Kx)
    Cϕ = similar(Kx)
    Cη_fluct = similar(Kx)
    Kx_fluct_new = similar(Kx)
    nodes, weights = _normal_hermite_rule(n_quad)
    znodes = sqrt(2.0) .* nodes
    ϕ_nodes = Vector{Float64}(undef, n_quad)
    ϕ_weighted_nodes = Vector{Float64}(undef, n_quad)

    Sη = zeros(Float64, nFreq)
    Sx_new = zeros(ComplexF64, nFreq)
    H_non = zeros(ComplexF64, nFreq)

    plan_r = plan_rfft(Cϕ)
    plan_ir_corr = plan_irfft(Sx_new, nTime)

    μx = if isnothing(μ0)
        if τchn > 1/N || J0 > 0.0
            0.5
        else
            0.0
        end
    else
        μ0
    end

    mϕ = 0.0
    α = 0.0
    converged = false
    it_conv = max_iter

    for iter in 1:max_iter
        Δx = max(Kx[1], 0.0)
        mϕ, α, _ = _normal_moments_quad(ϕ, ϕ_prime, μx, Δx, znodes, weights)
        _normal_phi_nodes_weighted!(ϕ_nodes, ϕ_weighted_nodes, ϕ, μx, Δx, znodes, weights)
        _fill_stationary_Cϕ_B0!(Cϕ, ϕ, μx, Δx, Kx, ϕ_nodes, ϕ_weighted_nodes, znodes, weights)

        Cϕ_inf = _estimate_covariance_plateau(Cϕ)
        K_static = g^2 * A * Cϕ_inf + N * g^2 * τcon * mϕ^2

        @. Cη_fluct = g^2 * A * (Cϕ - Cϕ_inf)
        Sη .= real.(plan_r * Cη_fluct) .* dt
        Sη .= max.(Sη, 0.0)

        @. H_non = 1.0 / (1.0 + im * ω_r)
        @. Sx_new = ComplexF64(abs2(H_non) * Sη)

        Kx_fluct_new .= (plan_ir_corr * Sx_new) ./ dt
        @. Kx_new = Kx_fluct_new + K_static

        μ_prop = N * mϕ * (J0 + g^2 * τchn * α)
        μ_new = (1.0 - damp) * μx + damp * μ_prop

        diff_K = norm(Kx_new .- Kx) / max(norm(Kx), eps(Float64))
        diff_μ = abs(μ_new - μx) / max(abs(μx), 0.01)
        diff = max(diff_K, diff_μ)


        @. Kx = (1.0 - damp) * Kx + damp * Kx_new
        μx = μ_new

        if verbose
            println(
                "Iter $iter: residual=$(round(diff, digits=6)), " *
                "mx=$(round(μx, digits=6)), " *
                "mϕ=$(round(mϕ, digits=6)), " *
                "Δx=$(round(Kx[1], digits=6))"
            )
            flush(stdout)
        end

        if diff < tol
            converged = true
            it_conv = iter
            break
        end
    end

    if converged
        if verbose
            println("✓ Converged successfully in $it_conv iterations.")
            flush(stdout)
        end
    else
        @warn "DMFT loop did not converge within max_iter"
        flush(stdout)
    end

    Δx = max(Kx[1], 0.0)
    mϕ, α, _ = _normal_moments_quad(ϕ, ϕ_prime, μx, Δx, znodes, weights)
    _normal_phi_nodes_weighted!(ϕ_nodes, ϕ_weighted_nodes, ϕ, μx, Δx, znodes, weights)
    _fill_stationary_Cϕ_B0!(Cϕ, ϕ, μx, Δx, Kx, ϕ_nodes, ϕ_weighted_nodes, znodes, weights)

    Cx = Kx .+ μx^2
    Cx_h = (dt / sqrt(2π)) .* fft(Cx)
    Cϕ_h = (dt / sqrt(2π)) .* fft(Cϕ)

    χx_h = zeros(ComplexF64, nTime)
    χϕ_h = zeros(ComplexF64, nTime)
    _compute_transfer_function_unitary!(χx_h, χϕ_h, ω, α)

    if minimal_return
        return (
            Cϕ_h=Cϕ_h,
            χϕ_h=χϕ_h
        )
    end

    χ_t = exp.(-t_grid)
    χϕ_t = α .* χ_t

    return (
        converged=converged,
        t=t_grid,
        mx=μx,
        mϕ=mϕ,
        Δx=Δx,
        Cx=copy(Cx),
        Cϕ=copy(Cϕ),
        χ=χ_t,
        χϕ=χϕ_t,
        ω=ω,
        Cx_h=Cx_h,
        Cϕ_h=Cϕ_h,
        χx_h=χx_h,
        χϕ_h=χϕ_h
    )
end

# ======================== Stationary solver end ==============================

# ======================== Nonstationary solver  ==============================

"""
    For the limit cycle case with B = 0, we can use this nonstationary solver.
"""

function _normal_hermite_rule(n_quad::Int64)
    n_quad > 0 || throw(ArgumentError("n_quad must be positive."))
    offdiag = n_quad == 1 ? Float64[] : sqrt.(collect(1.0:(n_quad - 1)) ./ 2.0)
    return QuadGK.gauss(SymTridiagonal(zeros(Float64, n_quad), offdiag), 1.0)
end


function _normal_moments_quad(ϕ::Function, ϕ_prime::Function, μ::Float64, Δ::Float64, znodes, weights)
    Δ = max(Δ, 0.0)
    if Δ <= 1e-14
        y = ϕ(μ)
        return y, ϕ_prime(μ), y^2
    end

    σ = sqrt(Δ)
    m = 0.0
    α = 0.0
    C = 0.0
    @inbounds for k in eachindex(znodes)
        x = μ + σ * znodes[k]
        w = weights[k]
        y = ϕ(x)
        m += w * y
        α += w * ϕ_prime(x)
        C += w * y^2
    end
    return m, α, C
end

function _normal_phi_nodes!(out, ϕ::Function, μ::Float64, Δ::Float64, znodes)
    Δ = max(Δ, 0.0)
    if Δ <= 1e-14
        fill!(out, ϕ(μ))
        return out
    end

    σ = sqrt(Δ)
    @inbounds for k in eachindex(znodes)
        out[k] = ϕ(μ + σ * znodes[k])
    end
    return out
end

function _normal_phi_nodes_weighted!(
    out::Vector{Float64},
    weighted::Vector{Float64},
    ϕ::Function,
    μ::Float64,
    Δ::Float64,
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    _normal_phi_nodes!(out, ϕ, μ, Δ, znodes)
    @tturbo for k in eachindex(out)
        weighted[k] = weights[k] * out[k]
    end
    return out, weighted
end

function _normal_pair_phi_product_quad(ϕ::Function, μ1::Float64, μ2::Float64,
    Δ1::Float64, Δ2::Float64, K12::Float64, znodes, weights, ϕ1_nodes)

    Δ1 = max(Δ1, 0.0)
    Δ2 = max(Δ2, 0.0)

    if Δ1 <= 1e-14 && Δ2 <= 1e-14
        return ϕ(μ1) * ϕ(μ2)
    elseif Δ2 <= 1e-14
        y2 = ϕ(μ2)
        val = 0.0
        @inbounds for a in eachindex(znodes)
            val += weights[a] * ϕ1_nodes[a] * y2
        end
        return val
    elseif Δ1 <= 1e-14
        y1 = ϕ(μ1)
        σ2 = sqrt(Δ2)
        val = 0.0
        @inbounds for b in eachindex(znodes)
            val += weights[b] * y1 * ϕ(μ2 + σ2 * znodes[b])
        end
        return val
    end

    σ1 = sqrt(Δ1)
    σ2 = sqrt(Δ2)
    ρ = clamp(K12 / (σ1 * σ2), -1.0, 1.0)
    ρperp = sqrt(max(1.0 - ρ^2, 0.0))
    val = 0.0

    @inbounds for a in eachindex(znodes)
        u = znodes[a]
        ϕ1 = ϕ1_nodes[a]
        inner = 0.0
        for b in eachindex(znodes)
            v = znodes[b]
            x2 = μ2 + σ2 * (ρ * u + ρperp * v)
            inner += weights[b] * ϕ(x2)
        end
        val += weights[a] * ϕ1 * inner
    end
    return val
end

function _fill_stationary_Cϕ_B0!(
    Cϕ::Vector{Float64},
    ϕ::Function,
    μ::Float64,
    Δ::Float64,
    Kx::Vector{Float64},
    ϕ_nodes::Vector{Float64},
    ϕ_weighted_nodes::Vector{Float64},
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    Δ = max(Δ, 0.0)
    if Δ <= 1e-14
        fill!(Cϕ, ϕ(μ)^2)
        return Cϕ
    end

    @inbounds for i in eachindex(Kx)
        Cϕ[i] = _normal_pair_phi_product_quad(
            ϕ, μ, μ, Δ, Δ, Kx[i], znodes, weights, ϕ_nodes
        )
    end
    return Cϕ
end

function _fill_stationary_Cϕ_B0!(
    Cϕ::Vector{Float64},
    ::typeof(tanh),
    μ::Float64,
    Δ::Float64,
    Kx::Vector{Float64},
    ϕ_nodes::Vector{Float64},
    ϕ_weighted_nodes::Vector{Float64},
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    Δ = max(Δ, 0.0)
    if Δ <= 1e-14
        fill!(Cϕ, tanh(μ)^2)
        return Cϕ
    end

    σ = sqrt(Δ)
    @inbounds for i in eachindex(Kx)
        Kτ = clamp(Kx[i], -Δ, Δ)
        ρ = clamp(Kτ / Δ, -1.0, 1.0)
        ρperp = sqrt(max(1.0 - ρ^2, 0.0))
        Cϕ[i] = _normal_pair_tanh_product_tturbo(
            μ, σ, ρ, ρperp, ϕ_weighted_nodes, znodes, weights
        )
    end
    return Cϕ
end

function _normal_pair_tanh_product_tturbo(
    μ2::Float64,
    σ2::Float64,
    ρ::Float64,
    ρperp::Float64,
    ϕ1_weighted_nodes::Vector{Float64},
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    s = 0.0
    @tturbo for b in eachindex(znodes), a in eachindex(znodes)
        x2 = μ2 + σ2 * (ρ * znodes[a] + ρperp * znodes[b])
        s += ϕ1_weighted_nodes[a] * weights[b] * tanh(x2)
    end
    return s
end

function _fill_Cϕ_column_B0!(
    Cϕ::Matrix{Float64},
    jp1::Int,
    ϕ::Function,
    mx::Vector{Float64},
    Kx::Matrix{Float64},
    mϕ::Vector{Float64},
    ϕ_new_nodes::Vector{Float64},
    ϕ_new_weighted_nodes::Vector{Float64},
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    Δ1 = max(Kx[jp1, jp1], 0.0)
    μ1 = mx[jp1]
    j = jp1 - 1

    @inbounds for i in 1:j
        Δ2 = max(Kx[i, i], 0.0)
        val = if Δ1 <= 1e-14 && Δ2 <= 1e-14
            ϕ(μ1) * ϕ(mx[i])
        elseif Δ2 <= 1e-14
            mϕ[jp1] * ϕ(mx[i])
        elseif Δ1 <= 1e-14
            ϕ(μ1) * mϕ[i]
        else
            _normal_pair_phi_product_quad(
                ϕ, μ1, mx[i], Δ1, Δ2, Kx[jp1, i], znodes, weights, ϕ_new_nodes
            )
        end
        Cϕ[jp1, i] = val
        Cϕ[i, jp1] = val
    end
    return Cϕ
end

function _fill_Cϕ_column_B0!(
    Cϕ::Matrix{Float64},
    jp1::Int,
    ::typeof(tanh),
    mx::Vector{Float64},
    Kx::Matrix{Float64},
    mϕ::Vector{Float64},
    ϕ_new_nodes::Vector{Float64},
    ϕ_new_weighted_nodes::Vector{Float64},
    znodes::Vector{Float64},
    weights::Vector{Float64},
)
    Δ1 = max(Kx[jp1, jp1], 0.0)
    μ1 = mx[jp1]
    j = jp1 - 1

    @inbounds for i in 1:j
        Δ2 = max(Kx[i, i], 0.0)
        val = if Δ1 <= 1e-14 && Δ2 <= 1e-14
            tanh(μ1) * tanh(mx[i])
        elseif Δ2 <= 1e-14
            mϕ[jp1] * tanh(mx[i])
        elseif Δ1 <= 1e-14
            tanh(μ1) * mϕ[i]
        else
            μ2 = mx[i]
            σ1 = sqrt(Δ1)
            σ2 = sqrt(Δ2)
            ρ = clamp(Kx[jp1, i] / (σ1 * σ2), -1.0, 1.0)
            ρperp = sqrt(max(1.0 - ρ^2, 0.0))
            _normal_pair_tanh_product_tturbo(
                μ2, σ2, ρ, ρperp, ϕ_new_weighted_nodes, znodes, weights
            )
        end
        Cϕ[jp1, i] = val
        Cϕ[i, jp1] = val
    end
    return Cϕ
end

function _B0_response_from_alpha(α::Vector{Float64}, dt::Float64)
    nT = length(α)
    χϕ = zeros(nT, nT)
    decay = exp(-dt)
    decay_powers = Vector{Float64}(undef, nT)
    decay_powers[1] = decay
    @inbounds for k in 2:nT
        decay_powers[k] = decay * decay_powers[k - 1]
    end
    @tturbo for j in 1:nT, i in 1:nT
        δ = i - j
        k = ifelse(δ > 0, δ, 1)
        χϕ[i, j] = ifelse(δ > 0, α[i] * decay_powers[k], 0.0)
    end
    return χϕ
end

function DMFT_Nonstationary_Solver_B0(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    Ttot::Float64=100.0,
    dt::Float64=0.1,
    ϕ::Function=tanh,
    ϕ_prime::Function= x -> (1 - tanh(x)^2),
    traj_init_μx::Float64=0.0,
    traj_init_σx::Float64=sqrt(0.1),
    n_quad::Int64=128)

    τchn, τrec, τcon, τdiv = τ_parser(τ)
    B = τrec - 2.0 * τchn
    abs(B) <= 1e-5 || throw(ArgumentError("DMFT_Nonstationary_Solver_B0 requires B = τrec - 2τchn = 0. Got B = $B."))
    traj_init_σx >= 0.0 || throw(ArgumentError("traj_init_σx must be nonnegative."))

    nT = round(Int, Ttot/dt)
    nT > 1 || throw(ArgumentError("Ttot / dt must give at least two time points."))

    nodes, weights = _normal_hermite_rule(n_quad)
    znodes = sqrt(2.0) .* nodes
    decay = exp(-dt)
    β = 1.0 - decay

    mx = zeros(nT)
    Kx = zeros(nT, nT)
    Cx = zeros(nT, nT)
    mϕ = zeros(nT)
    Cϕ = zeros(nT, nT)
    α = zeros(nT)
    ϕ_new_nodes = Vector{Float64}(undef, n_quad)
    ϕ_new_weighted_nodes = Vector{Float64}(undef, n_quad)

    mx[1] = traj_init_μx
    Kx[1, 1] = traj_init_σx^2
    Cx[1, 1] = Kx[1, 1] + mx[1]^2
    mϕ[1], α[1], Cϕ[1, 1] = _normal_moments_quad(ϕ, ϕ_prime, mx[1], Kx[1, 1], znodes, weights)

    A = 1.0 - τcon - τdiv
    g2 = g^2
    Ag2 = A * g2
    Nτcong2 = N * τcon * g2
    NJ0 = N * J0
    Nτchng2dt = dt * N * τchn * g2
    Cη = similar(Cϕ)
    Cη[1, 1] = Ag2 * Cϕ[1, 1] + Nτcong2 * mϕ[1]^2

    memory = 0.0
    @inbounds for j in 1:(nT - 1)
        feedback = NJ0 * mϕ[j] + Nτchng2dt * α[j] * memory
        mx[j + 1] = decay * mx[j] + β * feedback

        Cxη = 0.0
        for i in 1:(j + 1)
            if i > 1
                Cxη = decay * Cxη + β * Cη[i - 1, j]
            end
            if i <= j
                Kx[i, j + 1] = decay * Kx[i, j] + β * Cxη
                Kx[j + 1, i] = Kx[i, j + 1]
            else
                Kx[j + 1, j + 1] = decay * Kx[j + 1, j] + β * Cxη
            end
        end

        Cx[j + 1, j + 1] = Kx[j + 1, j + 1] + mx[j + 1]^2
        mxjp1 = mx[j + 1]
        rows = 1:j
        @views Cx[rows, j + 1] .= Kx[rows, j + 1] .+ mx[rows] .* mxjp1
        @views Cx[j + 1, rows] .= Cx[rows, j + 1]

        Δnew = max(Kx[j + 1, j + 1], 0.0)
        mϕ[j + 1], α[j + 1], Cϕ[j + 1, j + 1] = _normal_moments_quad(ϕ, ϕ_prime, mx[j + 1], Δnew, znodes, weights)
        _normal_phi_nodes_weighted!(
            ϕ_new_nodes, ϕ_new_weighted_nodes, ϕ, mx[j + 1], Δnew, znodes, weights
        )
        _fill_Cϕ_column_B0!(
            Cϕ, j + 1, ϕ, mx, Kx, mϕ, ϕ_new_nodes, ϕ_new_weighted_nodes, znodes, weights
        )

        mϕjp1 = mϕ[j + 1]
        rows = 1:(j + 1)
        @views Cη[rows, j + 1] .= Ag2 .* Cϕ[rows, j + 1] .+ Nτcong2 .* mϕ[rows] .* mϕjp1
        @views Cη[j + 1, rows] .= Cη[rows, j + 1]

        memory = decay * (memory + mϕ[j])
    end

    χϕ = _B0_response_from_alpha(α, dt)
    return mx, mϕ, Cx, Cϕ, χϕ
end

# ======================== Nonstationary solver end ==============================

# ======================= DMFT functions end =============================

## Theoretical largest Lyapunov exponent for B ≈ 0
# C_{φ'}(τ) = ⟨φ'(t) φ'(t+τ)⟩ via Price's theorem.
function compute_Cphip_grid(
    ϕp::Function,
    mx::Float64,
    Cx::AbstractVector{Float64};
    n_quad::Int64=128,
)
    n = length(Cx)

    Cpp = Vector{Float64}(undef, n)
    Kx = Vector{Float64}(undef, n)
    mx2 = mx * mx
    Δx = max(Float64(Cx[1]) - mx2, 0.0)

    @inbounds for i in eachindex(Cx)
        Kx[i] = Float64(Cx[i]) - mx2
    end

    nodes, weights = _normal_hermite_rule(n_quad)
    znodes = sqrt(2.0) .* nodes
    ϕp_nodes = Vector{Float64}(undef, n_quad)
    ϕp_weighted_nodes = Vector{Float64}(undef, n_quad)
    _normal_phi_nodes_weighted!(ϕp_nodes, ϕp_weighted_nodes, ϕp, mx, Δx, znodes, weights)
    _fill_stationary_Cϕ_B0!(Cpp, ϕp, mx, Δx, Kx, ϕp_nodes, ϕp_weighted_nodes, znodes, weights)

    return Cpp, Kx, Δx
end

# Ground state of -ψ'' + Wψ = Eψ on the symmetric interval implied by W_half.
function ground_state_energy_even_potential(W_half::Vector{Float64}, dt::Float64)
    nh = length(W_half)

    n_full = 2 * nh - 1
    n_inner = n_full - 2
    invdt2 = inv(dt * dt)
    center_idx = nh

    diag = Vector{Float64}(undef, n_inner)
    @inbounds for j in 1:n_inner
        full_idx = j + 1 
        half_idx = abs(full_idx - center_idx) + 1
        diag[j] = 2.0 * invdt2 + W_half[half_idx]
    end

    off = fill(-invdt2, n_inner - 1)
    H = SymTridiagonal(diag, off)
    return eigvals(H, 1:1)[1]
end

function LLE_DMFT(sol, N::Int64, g::Float64, τ::Tuple{Vararg{Float64}};
    ϕ_prime::Function=x->(1-tanh(x)^2), n_quad::Int64=128)

    τchn, τrec, τcon, τdiv = τ_parser(τ)
    if abs(τrec - 2τchn) > 5.0/N
        throw(ArgumentError("This function only works for B ≈ 0."))
    end
    A = 1.0 - τcon - τdiv
    
    t = sol.t
    Cx = sol.Cx

    dt = Float64(t[2] - t[1])
    nh = div(length(t), 2) + 1
    t_half = @view t[1:nh]
    Cx_half = @view Cx[1:nh]

    Cphip_half, _, Δx = compute_Cphip_grid(ϕ_prime, Float64(sol.mx), Cx_half; n_quad=n_quad)

    W_half = Vector{Float64}(undef, nh)
    scale = A * g * g
    @. W_half = 1.0 - scale * Cphip_half

    E0 = ground_state_energy_even_potential(W_half, dt)
    λmax = -1.0 + sqrt(max(0.0, 1.0 - E0))

    return (
        λmax = λmax,
        E0 = E0,
        Cphip = Cphip_half,
        τ = collect(t_half),
        W = W_half,
        Δx = Δx,
    )
end


end


## CUDA implementation of the DMFT solver for B ≠ 0.
module DMFT_CUDA
using FFTW, StatsBase, Dates, CairoMakie
using LinearAlgebra, DifferentialEquations
using Random, CUDA
include("Utils.jl")
using .Utils
export CreateDMFTRateModel, DMFTMainloop,CreateStationaryDMFTRateModel, 
        DMFTStationaryMainloop, DMFTStationaryMainloop_TimeInteg

# This version is adpated from Zou and Huang 2024.
# We can use it for non-stationary dynamics.

# ======================== Nonstationary DMFT ===================================

struct DMFTRateModel_CUDA
    N::Int32
    J0::Float32
    g::Float32
    τchn::Float32
    τrec::Float32
    τcon::Float32
    τdiv::Float32
    Ttot::Float32
    dt::Float32
    nTime::Int32
    traj_init_μx::Float32
    traj_init_σx::Float32
    nTraj_list::Tuple{Vararg{Int32}}
    nIte_list::Tuple{Vararg{Int32}}
    damp_R::Tuple{Vararg{Float32}}
    damp_C::Tuple{Vararg{Float32}}
    B1::Int32 # Batch size along t1
    B2::Int32 # Batch size along t2
    tile_size::Int32 # Number of sampled trajectories integrated per GPU tile
    nThreads::Tuple{Int32,Int32} # Number of threads used by GPU
    traj_stride::Int32 # Trajectory stride for χ integration
    threshold::Float32
    ϕ::Function
    ϕ_prime::Function
end

function CreateDMFTRateModel(N::Int,
    J0::Real, g::Real, τ::Tuple{Vararg{Float64}},
    Ttot::Real,
    nIte_list::Tuple{Vararg{Int}},
    nTraj_list::Tuple{Vararg{Int}},
    damp_R::Tuple{Vararg{Float64}},
    damp_C::Tuple{Vararg{Float64}};
    dt::Real=0.1,
    threshold::Real=1e-4,
    ϕ::Function=tanh,
    ϕ_prime::Function=x -> 1 - tanh(x)^2,
    B1::Int=256,
    B2::Int=64,
    tile_size::Int=4096, # Use a smaller value if out of memory. This won't have a big impact on speed.
    nThreads::Tuple{Int,Int}=(256, 4),
    traj_stride::Int=2,
    traj_init_μx::Real=0.0,
    traj_init_σx::Real=sqrt(0.1)
)
    if B1 <= 0 || B2 <= 0
        throw(ArgumentError("B1 and B2 must be positive. Provided B1=$B1, B2=$B2."))
    end
    if traj_stride <= 0
        throw(ArgumentError("traj_stride must be positive. Provided traj_stride=$traj_stride."))
    end
    if tile_size <= 0
        throw(ArgumentError("tile_size must be positive. Provided tile_size=$tile_size."))
    end
    if traj_init_σx < 0
        throw(ArgumentError("traj_init_σx must be nonnegative. Provided traj_init_σx=$traj_init_σx."))
    end

    dev = CUDA.device()
    max_threads_per_block = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_x = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    max_y = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)

    total_threads = nThreads[1] * nThreads[2]

    if total_threads > max_threads_per_block
        throw(ArgumentError("🚫 GPU limit exceeded: Total threads per block cannot exceed $max_threads_per_block. Provided: $total_threads."))
    end

    if nThreads[1] > max_x || nThreads[2] > max_y
        throw(ArgumentError("🚫 GPU dimension limit exceeded. Max X: $max_x, Max Y: $max_y. Provided: $(nThreads[1])x$(nThreads[2])."))
    end

    if total_threads % 32 != 0
        @warn "🚨 Performance Notice: For optimal GPU warp scheduling, the total number of threads ($total_threads) should ideally be a multiple of 32."
    end

    nTime_calc = round(Ttot / dt)
    max_nTraj = maximum(nTraj_list)
    max_n_use = cld(max_nTraj, traj_stride)
    tile_size_eff = min(tile_size, max_n_use)

    # 10 large N_T x N_T matrices
    mem_matrices = 10.0 * Float64(nTime_calc)^2 * 4.0
    # 5 trajectory matrices of size (max_nTraj, N_T)
    mem_traj = 5.0 * Float64(max_nTraj) * nTime_calc * 4.0
    # Workspace matrices (tiled χ integration workspace): V + H + temp
    mem_ws = (Float64(tile_size_eff) * B2 * nTime_calc +
              Float64(tile_size_eff) * B2 * B1 +
              Float64(tile_size_eff) * B2) * 4
    # Some other vectors
    mem_vectors = (6.0 * nTime_calc + 3.0 * max_nTraj) * 4.0

    # Add 10% safety buffer for miscellanea and overhead
    est_peak_bytes = (mem_matrices + mem_traj + mem_ws + mem_vectors) * 1.2

    GC.gc(true)
    CUDA.reclaim()

    free_mem_bytes, _ = CUDA.memory_info()

    est_peak_gb = round(est_peak_bytes / 1024^3, digits=2)
    free_mem_gb = round(free_mem_bytes / 1024^3, digits=2)

    if est_peak_bytes > free_mem_bytes
        error_msg = "🚫 Out of Memory Risk: Estimated peak VRAM is $(est_peak_gb)GB, but only $(free_mem_gb)GB is free. " *
                    "Reduce Ttot, increase dt, lower max(nTraj_list), reduce B2, or reduce tile_size."
        throw(ArgumentError(error_msg))
    end

    τchn, τrec, τcon, τdiv = τ_parser(τ) .|> Float32

    N = Int32(N)
    nTime = Int32(round(Ttot / dt))
    J0 = Float32(J0)
    g = Float32(g)
    Ttot = Float32(Ttot)
    traj_init_μx = Float32(traj_init_μx)
    traj_init_σx = Float32(traj_init_σx)
    nIte_list = Int32.(nIte_list)
    nTraj_list = Int32.(nTraj_list)
    dt = Float32(dt)
    damp_R = Float32.(damp_R)
    damp_C = Float32.(damp_C)
    threshold = Float32(threshold)
    B1 = Int32(B1)
    B2 = Int32(B2)
    tile_size = Int32(tile_size_eff)
    nThreads = Int32.(nThreads)
    traj_stride = Int32(traj_stride)

    return DMFTRateModel_CUDA(N, J0, g,
        τchn, τrec, τcon, τdiv,
        Ttot, dt, nTime, traj_init_μx, traj_init_σx,
        nTraj_list, nIte_list,
        damp_R, damp_C, B1, B2, tile_size, nThreads, traj_stride, threshold,
        ϕ, ϕ_prime)
end

# function _nearest_posdef!(A::CuMatrix{Float32}; ε::Float32=1f-5)

#     N = size(A, 1)
#     x0 = CUDA.rand(Float32, N)

#     # Use `eigen` may cause overflow
#     # Time complexity O(k N^2) 
#     # Memory O(k N), k ≪ N
#     vals, _, info = eigsolve(Symmetric(A), x0, 1, :SR;
#         issymmetric=true, tol=1f-3, krylovdim=200,
#         maxiter=300)

#     # Check for convergence
#     margin = 1.02f0
#     if info.converged < 1
#         @warn "KrylovKit failed to converge. Info: iterations = $(info.numiter), residual = $(info.normres[1])."
#         margin = 1.10f0
#     end

#     λ_min = real(vals[1])

#     # Shift diagonal if the smallest eigenvalue is less than epsilon
#     if λ_min < ε
#         shift = (ε - λ_min) * margin
#         A[diagind(A)] .+= shift
#     end
#     # Now A_sym must be positive-definite mathematically, but 
#     # Lanczos approximation and Float32 Roundoff can still be a problem in reality
#     return A
# end

function _add_diag_shift_kernel!(A::CuDeviceMatrix{Float32}, λ::Float32, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds A[i, i] += λ
    end
    return
end

function add_diag_shift!(A::CuMatrix{Float32}, λ::Float32)
    λ == 0f0 && return A
    n = Int32(size(A, 1))
    threads = 256
    blocks = cld(Int(n), threads)
    @cuda threads = threads blocks = blocks _add_diag_shift_kernel!(A, λ, n)
    return A
end

function sample_η!(η::CuMatrix{Float32},
    Cη::CuMatrix{Float32}, Cη_work::CuMatrix{Float32}, Z::CuMatrix{Float32},
    model::DMFTRateModel_CUDA, Cϕ::CuMatrix{Float32},
    mϕ::CuVector{Float32}; shift_rtol::Float32=1f-6,
    shift_growth::Float32=8f0, shift_max::Float32=1f-2
)

    A = 1f0 - model.τcon - model.τdiv

    copyto!(Cη, Cϕ)
    Cη .*= (A * model.g^2)

    if model.τcon != 0f0
        alpha = model.N * model.g^2 * model.τcon
        mul!(Cη, mϕ, transpose(mϕ), alpha, 1f0)
    end

    maxabs = mapreduce(abs, max, Cη)
    isfinite(maxabs) || error("Cη contains NaN or Inf.")

    base_shift = shift_rtol * max(1f0, maxabs)

    shift = 0f0
    while true
        copyto!(Cη_work, Cη)
        add_diag_shift!(Cη_work, shift)

        F = cholesky!(Symmetric(Cη_work), check=false)
        if issuccess(F)
            CUDA.randn!(Z)
            mul!(η, Z, transpose(F.L))
            return η # nTraj × nTime
        end

        if shift == shift_max
            error("Cholesky failed up to diagonal shift = $(shift_max)")
        end

        shift = (shift == 0f0) ? base_shift : min(shift * shift_growth, shift_max)
    end
end

function integ_xtraj!(
    x_traj::CuMatrix{Float32}, ϕx::CuMatrix{Float32},
    mx_t::CuVector{Float32}, mϕ_t::CuVector{Float32}, f::CuVector{Float32},
    temp::CuVector{Float32}, new_temp::CuVector{Float32}, model::DMFTRateModel_CUDA,
    χϕ::CuMatrix{Float32}, η::CuMatrix{Float32}
)
    nTraj, nTime = size(x_traj)
    inv_N = 1f0 / Float32(nTraj)

    CUDA.randn!(temp)
    @. temp = model.traj_init_σx * temp + model.traj_init_μx

    copyto!(@view(x_traj[:, 1]), temp)
    @. ϕx[:, 1] = model.ϕ(temp)

    # Use sum! into a 1-element view to prevent CPU synchronization
    sum!(@view(mx_t[1:1]), temp)
    sum!(@view(mϕ_t[1:1]), @view(ϕx[:, 1]))
    @. mx_t[1:1] *= inv_N
    @. mϕ_t[1:1] *= inv_N

    B_val = model.τrec - 2f0 * model.τchn
    C1 = model.dt * model.g^2 * B_val
    C2 = model.dt * model.N * model.τchn * model.g^2
    C3 = model.N * model.J0

    α = exp(-model.dt)
    β = 1f0 - α

    # Buffer to hold the scalar dot product entirely in VRAM
    scalar_buf = CuArray{Float32}(undef, 1)

    for t in 1:(nTime-1)
        @views copyto!(f, η[:, t])

        if t > 1
            w = @view χϕ[t, 1:(t-1)]
            Φ = @view ϕx[:, 1:(t-1)]

            # Batched Mean-Field Feedback: f += C1 * Φ * w
            mul!(f, Φ, w, C1, 1f0)

            if C2 != 0f0
                mϕ_past = @view mϕ_t[1:(t-1)]
                sum!(scalar_buf, w .* mϕ_past)

                # Broadcast the 1-element scalar_buf and 1-element mϕ_t[t:t] view directly to f
                @. f += C2 * scalar_buf + C3 * mϕ_t[t:t]
            else
                @. f += C3 * mϕ_t[t:t]
            end
        else
            @. f += C3 * mϕ_t[t:t]
        end

        # State Update
        @. new_temp = α * temp + β * f

        copyto!(@view(x_traj[:, t+1]), new_temp)
        @. ϕx[:, t+1] = model.ϕ(new_temp)

        # In-place GPU reductions for the means (bypass CPU)
        sum!(@view(mx_t[t+1:t+1]), new_temp)
        sum!(@view(mϕ_t[t+1:t+1]), @view(ϕx[:, t+1]))
        @. mx_t[t+1:t+1] *= inv_N
        @. mϕ_t[t+1:t+1] *= inv_N

        copyto!(temp, new_temp)
    end
end

# ============= Integrate χ in the time domain ================

struct ChiChiPhiGPUWorkspace
    V::CuArray{Float32,3}          # (n_use, B2, nTime)
    H::CuArray{Float32,3}          # (n_use, B2, B1)
    temp::CuMatrix{Float32}        # (n_use, B2)
end

function ChiChiPhiGPUWorkspace(n_use::Integer, nTime::Integer,
    B1::Integer, B2::Integer)
    return ChiChiPhiGPUWorkspace(
        CuArray{Float32}(undef, n_use, B2, nTime),
        CuArray{Float32}(undef, n_use, B2, B1),
        CuArray{Float32}(undef, n_use, B2)
    )
end

@inline _pow2_threads(n::Integer) = max(32, prevpow(2, min(256, Int(n))))

function kernel_init_batch_add!(
    V::CuDeviceArray{Float32,3},
    temp::CuDeviceMatrix{Float32},
    χ::CuDeviceMatrix{Float32},
    χϕ::CuDeviceMatrix{Float32},
    ϕ_prime_x::CuDeviceMatrix{Float32},
    t2_start::Int32,
    n_batch_active::Int32,
    off::Int32,
    stride::Int32,
    n_tile::Int32
)
    j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    b = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    tid = Int32(threadIdx().x)
    nthreads = Int32(blockDim().x)

    if b > n_batch_active
        return
    end

    t2 = t2_start + b - 1
    t2_next = t2 + 1
    v_init = 0f0

    @inbounds begin
        if j <= n_tile
            traj = off + (j - 1) * stride
            temp[j, b] = 1f0

            # Only the interval that may be touched by later GEMMs must be zero.
            for k in t2_start:t2
                V[j, b, k] = 0f0
            end

            v_init = ϕ_prime_x[traj, t2_next]
            V[j, b, t2_next] = v_init
        end
    end

    # Exactly one full addition for χ[t2+1, t2].
    if blockIdx().x == 1 && tid == 1
        @inbounds χ[t2_next, t2] += Float32(n_tile)
    end

    sh = CuDynamicSharedArray(Float32, Int(nthreads))
    @inbounds sh[tid] = v_init
    sync_threads()

    offset = nthreads >>> 1
    while offset > 0
        if tid <= offset
            @inbounds sh[tid] += sh[tid+offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        CUDA.@atomic χϕ[t2_next, t2] += sh[1]
    end

    return
end

function kernel_history_reduce!(
    χ::CuDeviceMatrix{Float32},
    χϕ::CuDeviceMatrix{Float32},
    V::CuDeviceArray{Float32,3},
    temp::CuDeviceMatrix{Float32},
    H::CuDeviceArray{Float32,3},
    ϕ_prime_x::CuDeviceMatrix{Float32},
    χϕ_prev::CuDeviceMatrix{Float32},
    t2_start::Int32,
    n_batch_active::Int32,
    t_start::Int32,
    t_end::Int32,
    off::Int32,
    stride::Int32,
    α::Float32,
    βC1::Float32,
    n_tile::Int32
)
    j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    b = Int32(blockIdx().y)
    tid = Int32(threadIdx().x)
    nthreads = Int32(blockDim().x)

    if b > n_batch_active
        return
    end

    t2 = t2_start + b - 1
    has_j = j <= n_tile

    traj = Int32(0)
    temp_jb = 0f0

    @inbounds begin
        if has_j
            traj = off + (j - 1) * stride
            temp_jb = temp[j, b]
        end
    end

    sh_χ = CuDynamicSharedArray(Float32, Int(nthreads))
    sh_χϕ = CuDynamicSharedArray(Float32, Int(nthreads), Int(nthreads * sizeof(Float32)))

    @inbounds for t1 in t_start:t_end
        partial_χ = 0f0
        partial_χϕ = 0f0

        if has_j && (t1 > t2)
            add = (t_start > t2) ? H[j, b, t1-t_start+1] : 0f0

            k0 = max(t_start, t2)
            for k in k0:t1
                add = fma(χϕ_prev[t1, k], V[j, b, k], add)
            end

            temp_jb = fma(α, temp_jb, βC1 * add)
            vnew = temp_jb * ϕ_prime_x[traj, t1+1]

            V[j, b, t1+1] = vnew
            partial_χ = temp_jb
            partial_χϕ = vnew
        end

        sh_χ[tid] = partial_χ
        sh_χϕ[tid] = partial_χϕ
        sync_threads()

        offset = nthreads >>> 1
        while offset > 0
            if tid <= offset
                sh_χ[tid] += sh_χ[tid+offset]
                sh_χϕ[tid] += sh_χϕ[tid+offset]
            end
            sync_threads()
            offset >>>= 1
        end

        if tid == 1 && (t1 > t2)
            CUDA.@atomic χ[t1+1, t2] += sh_χ[1]
            CUDA.@atomic χϕ[t1+1, t2] += sh_χϕ[1]
        end
    end

    @inbounds begin
        if has_j
            temp[j, b] = temp_jb
        end
    end

    return
end

function integ_χ_χϕ!(
    χ::CuMatrix{Float32},
    χϕ::CuMatrix{Float32},
    ϕ_prime_x::CuMatrix{Float32},
    ws::ChiChiPhiGPUWorkspace,
    model::DMFTRateModel_CUDA,
    x_traj::CuMatrix{Float32},
    χϕ_old::CuMatrix{Float32};
    traj_stride::Int32=Int32(2),
    traj_offset::Int32=Int32(1),
    traj_tile_size::Int32=Int32(size(ws.V, 1))
)
    t1_block_size = Int(model.B1)
    t2_batch_size = Int(model.B2)
    nTraj, nTime = size(x_traj)

    ws_n_use, ws_B2, ws_nTime = size(ws.V)
    ws_H_nuse, ws_H_B2, ws_B1 = size(ws.H)
    ws_temp_nuse, ws_temp_B2 = size(ws.temp)

    stride_i = Int(traj_stride)
    stride_i > 0 || throw(ArgumentError("traj_stride must be positive."))

    off = mod1(Int(traj_offset), stride_i)
    n_use = length(off:stride_i:nTraj)
    n_use > 0 || throw(ArgumentError("No trajectories selected by traj_offset/traj_stride."))

    tile_use = min(Int(traj_tile_size), ws_n_use)
    tile_use > 0 || throw(ArgumentError("traj_tile_size must be >= 1."))

    fill!(χ, 0f0)
    fill!(χϕ, 0f0)
    @. ϕ_prime_x = model.ϕ_prime(x_traj)

    C1 = model.dt * model.g^2 * (model.τrec - 2f0 * model.τchn)
    α = exp(-model.dt)
    βC1 = (1f0 - α) * C1

    V_flat = reshape(ws.V, ws_n_use * t2_batch_size, nTime)
    H_flat = reshape(ws.H, ws_n_use * t2_batch_size, t1_block_size)

    requested_threads = Int(model.nThreads[1]) * Int(model.nThreads[2])

    main_threads = _pow2_threads(requested_threads)
    init_threads = (main_threads, 1)

    init_shmem = main_threads * sizeof(Float32)
    main_shmem = 2 * main_threads * sizeof(Float32)

    for j_start in 1:tile_use:n_use
        n_tile = min(tile_use, n_use - j_start + 1)
        off_tile = off + (j_start - 1) * stride_i

        for t2_start in 1:t2_batch_size:(nTime-1)
            n_batch_active = min(t2_batch_size, nTime - t2_start)

            init_blocks = (cld(n_tile, main_threads), n_batch_active)
            @cuda threads = init_threads blocks = init_blocks shmem = init_shmem kernel_init_batch_add!(
                ws.V, ws.temp, χ, χϕ, ϕ_prime_x,
                Int32(t2_start), Int32(n_batch_active), Int32(off_tile), Int32(traj_stride), Int32(n_tile),
            )

            for t_start in (t2_start+1):t1_block_size:(nTime-1)
                t_end = min(t_start + t1_block_size - 1, nTime - 1)
                chunk_len = t_end - t_start + 1

                n_hist_batches = min(n_batch_active, t_start - t2_start)
                if n_hist_batches > 0
                    hist_rows = ws_n_use * n_hist_batches
                    @views mul!(
                        view(H_flat, 1:hist_rows, 1:chunk_len),
                        view(V_flat, 1:hist_rows, t2_start:t_start-1),
                        transpose(view(χϕ_old, t_start:t_end, t2_start:t_start-1)),
                    )
                end

                n_work_batches = min(n_batch_active, t_end - t2_start)
                if n_work_batches > 0
                    main_blocks = (cld(n_tile, main_threads), n_work_batches)
                    @cuda threads = main_threads blocks = main_blocks shmem = main_shmem kernel_history_reduce!(
                        χ, χϕ, ws.V, ws.temp, ws.H, ϕ_prime_x, χϕ_old,
                        Int32(t2_start), Int32(n_work_batches), Int32(t_start), Int32(t_end), Int32(off_tile),
                        Int32(traj_stride), α, βC1, Int32(n_tile),
                    )
                end
            end
        end
    end

    inv_use = 1f0 / Float32(n_use)
    χ .*= inv_use
    χϕ .*= inv_use

    return χ, χϕ
end

# function initialize_mC(model::DMFTRateModel_CUDA; N::Int32=model.N,
#     T::Float32=200f0, burn::Float32=50f0, n_samples::Int64=36)

#     if haskey(ENV, "SLURM_MEM_PER_NODE")
#         check_memory = true
#         available_gb = parse(Float64, ENV["SLURM_MEM_PER_NODE"]) / 1024.0
#     elseif haskey(ENV, "SLURM_MEM_PER_CPU") && haskey(ENV, "SLURM_CPUS_ON_NODE")
#         check_memory = true
#         available_gb = parse(Float64, ENV["SLURM_MEM_PER_CPU"]) * parse(Float64, ENV["SLURM_CPUS_ON_NODE"]) / 1024.0
#     else
#         # not on Slurm node, skip memory guard
#         check_memory = false
#     end

#     M = model.nTime

#     # Avoid OOM kill
#     if check_memory
#         N_f64 = Float64(N)
#         M_f64 = Float64(M)
#         n_samples_f64 = Float64(n_samples)
#         bytes_per_f32 = 4.0
#         bytes_J_array = n_samples_f64 * N_f64 * N_f64 * bytes_per_f32
#         bytes_ode_temp = n_samples_f64 * (12.0 * N_f64 * bytes_per_f32)
#         bytes_outputs = n_samples_f64 * (2.0 * M_f64 * bytes_per_f32 + 2.0 * M_f64 * M_f64 * bytes_per_f32)
#         bytes_base = (N_f64 * N_f64 + 4.0 * M_f64 * M_f64 + 4.0 * M_f64) * bytes_per_f32
#         estimated_gb = (bytes_J_array + bytes_ode_temp + bytes_outputs + bytes_base) / (1024.0^3)

#         if estimated_gb > 0.85 * available_gb
#             error("Insufficient Slurm job memory for initialization:\n" *
#                   "  Estimated required: $(round(estimated_gb, digits=2)) GB\n" *
#                   "  Available to job:   $(round(available_gb, digits=2)) GB\n")
#         end
#     end

#     τ = (Float64(model.τchn), Float64(model.τrec),
#         Float64(model.τcon), Float64(model.τdiv))

#     J_array = [Matrix{Float32}(CreateJ(Int64(N), Float64(model.J0),
#         Float64(model.g), τ)) for _ in 1:n_samples]

#     tspan = (0f0, T)

#     function prob_func(prob, i, repeat)
#         u0 = 0.5f0 .* randn(Float32, N)
#         cache = zeros(Float32, N) # Thread-local cache
#         return remake(prob, u0=u0, p=(J_array[i], cache))
#     end

#     function ODEFunc!(du, u, p, t)
#         J, ϕu = p
#         @. ϕu = model.ϕ(u)
#         mul!(du, J, ϕu)
#         du .-= u
#         return nothing
#     end

#     function output_func(sol, i)
#         times = sol.t
#         burn_idx = findfirst(>=(burn), times)
#         X = reduce(hcat, @view sol.u[burn_idx:burn_idx+M-1])
#         ϕX = model.ϕ.(X)
#         mx = vec(mean(X, dims=1))
#         mϕ = vec(mean(ϕX, dims=1))
#         Cx = zeros(Float32, M, M)
#         Cϕ = zeros(Float32, M, M)
#         mul!(Cx, transpose(X), X, 1f0 / N, 0f0)
#         mul!(Cϕ, transpose(ϕX), ϕX, 1f0 / N, 0f0)
#         return ((mx, mϕ, Cx, Cϕ), false)
#     end

#     prob = ODEProblem(ODEFunc!, zeros(Float32, N),
#         tspan, (zeros(Float32, N, N), zeros(Float32, N)))

#     ensemble_problem = EnsembleProblem(prob, prob_func=prob_func,
#         output_func=output_func)
#     sim = solve(ensemble_problem, Tsit5(), EnsembleThreads();
#         trajectories=n_samples, saveat=model.dt)

#     mx_mean = zeros(Float32, M)
#     mϕ_mean = zeros(Float32, M)
#     Cx_mean = zeros(Float32, M, M)
#     Cϕ_mean = zeros(Float32, M, M)

#     for i in 1:n_samples
#         mx, mϕ, Cx, Cϕ = sim.u[i]
#         mx_mean .+= mx
#         mϕ_mean .+= mϕ
#         Cx_mean .+= Cx
#         Cϕ_mean .+= Cϕ
#     end

#     mx_mean ./= n_samples
#     mϕ_mean ./= n_samples
#     Cx_mean ./= n_samples
#     Cϕ_mean ./= n_samples

#     mx_mean = CuArray{Float32}(mx_mean)
#     mϕ_mean = CuArray{Float32}(mϕ_mean)
#     Cx_mean = CuArray{Float32}(Cx_mean)
#     Cϕ_mean = CuArray{Float32}(Cϕ_mean)

#     # Enforce positive-definiteness
#     _nearest_posdef!(Cx_mean)
#     _nearest_posdef!(Cϕ_mean)

#     return mx_mean, mϕ_mean, Cx_mean, Cϕ_mean
# end

function DMFTMainloop(model::DMFTRateModel_CUDA; verbose::Bool=true,
    minimal_return::Bool=false)

    # Validity check based on model parameters
    if 1f0 - 2f0 * model.τcon - 2f0 * model.τdiv - abs(model.τrec) < 0f0
        return nothing, nothing, nothing, nothing, nothing, nothing
    end

    nT = model.nTime

    ## For debugging.
    # Tinit = min(model.Ttot * 3, 3000f0)
    # if verbose
    #     println("Simulating the network to initialize the correlations...")
    #     flush(stdout)
    #     start = now()
    # end
    # if model.τchn <= 0f0
    #     mx, mϕ, Cx, Cϕ = initialize_mC(model; N=Int32(2000),
    #         T=Tinit, burn=Tinit / 5, n_samples=24)
    # else
    #     mx, mϕ, Cx, Cϕ = initialize_mC(model; T=Tinit, burn=Tinit / 5, n_samples=48)
    # end
    # if verbose
    #     elapsed = now() - start
    #     dur = round(Dates.value(elapsed) / 1e3, digits=1)
    #     println("Initialization done. Elapsed = $(dur)s.")
    #     flush(stdout)
    # end

    # μx = Float32(mean(mx))
    # Δx = Float32(mean(diag(Cx)) - μx^2)

    mx = CUDA.zeros(Float32, nT)
    mϕ = CUDA.zeros(Float32, nT)
    Cx = 0.1 .* I(nT) |> CuMatrix{Float32}
    Cϕ = 0.1 .* I(nT) |> CuMatrix{Float32}

    χ = diagm(-1 => ones(nT - 1)) |> CuMatrix{Float32}
    χϕ = diagm(-1 => ones(nT - 1)) |> CuMatrix{Float32}
    χϕ .*= 0.2f0

    Ite_count = 0
    nSteps = length(model.nIte_list)

    new_mx = similar(mx)
    new_mϕ = similar(mϕ)
    new_Cx = similar(Cx)
    new_Cϕ = similar(Cϕ)
    new_χ = similar(χ)
    new_χϕ = similar(χϕ)

    Cη = CUDA.zeros(Float32, nT, nT)
    Cη_work = similar(Cη)

    # mx_t and mϕ_t are computed in-place directly inside integ_xtraj!
    mx_t = CuVector{Float32}(undef, nT)
    mϕ_t = CuVector{Float32}(undef, nT)

    for block in 1:nSteps
        nIte = model.nIte_list[block]
        nTraj = model.nTraj_list[block]
        damp_R = model.damp_R[block]
        damp_C = model.damp_C[block]

        # Block-specific GPU allocations
        η = CUDA.zeros(Float32, nTraj, nT)
        Z = similar(η)

        x_traj = CUDA.zeros(Float32, nTraj, nT)
        ϕx = similar(x_traj)
        ϕp = similar(x_traj)

        f = CuVector{Float32}(undef, nTraj)
        temp = similar(f)
        new_temp = similar(f)

        cov_factor = (1f0 - damp_C) / Float32(nTraj)

        max_n_use = cld(nTraj, model.traj_stride)
        tile_n_use = min(max_n_use, Int(model.tile_size))
        ws = ChiChiPhiGPUWorkspace(tile_n_use, nT, model.B1, model.B2)

        for _ in 1:nIte
            Ite_count += 1

            sample_η!(η, Cη, Cη_work, Z, model, Cϕ, mϕ)
            integ_xtraj!(x_traj, ϕx, mx_t, mϕ_t, f, temp, new_temp,
                model, χϕ, η)

            @. new_mx = damp_C * mx + (1f0 - damp_C) * mx_t
            @. new_mϕ = damp_C * mϕ + (1f0 - damp_C) * mϕ_t

            copyto!(new_Cx, Cx)
            rmul!(new_Cx, damp_C)
            mul!(new_Cx, transpose(x_traj), x_traj, cov_factor, 1f0)

            copyto!(new_Cϕ, Cϕ)
            rmul!(new_Cϕ, damp_C)
            mul!(new_Cϕ, transpose(ϕx), ϕx, cov_factor, 1f0)

            # We cyclically use a subset of the trajectories with stride `model.traj_stride`, 
            # because the convergence of χ is easier.
            # Setting `traj_stride=1` is equivalent to using all trajectories.
            if nTraj < 4096
                traj_stride = Int32(1)
            else
                traj_stride = model.traj_stride
            end
            traj_offset = Int32(mod1(Ite_count, traj_stride))
            integ_χ_χϕ!(new_χ, new_χϕ, ϕp, ws, model, x_traj, χϕ;
                traj_stride=traj_stride, traj_offset=traj_offset)

            @. new_χ = damp_R * χ + (1f0 - damp_R) * new_χ
            @. new_χϕ = damp_R * χϕ + (1f0 - damp_R) * new_χϕ

            q_old = view(Cϕ, diagind(Cϕ))
            q_new = view(new_Cϕ, diagind(new_Cϕ))

            diff_sq = mapreduce((n, o) -> abs2(n - o), +, q_new, q_old)

            diff_q = Float32(sqrt(diff_sq) / max(sqrt(sum(abs2, q_old)), eps(Float32)) / (1f0 - damp_C))

            if verbose
                println("Iteration $Ite_count, normalized |ΔC| = $(diff_q)")
                flush(stdout)
            end

            if diff_q < model.threshold
                if minimal_return
                    return Array(new_Cϕ), Array(new_χϕ)
                else
                    return Array(new_mx), Array(new_mϕ), Array(new_Cx), Array(new_Cϕ), Array(new_χ), Array(new_χϕ)
                end
            end

            copyto!(mx, new_mx)
            copyto!(mϕ, new_mϕ)
            copyto!(Cx, new_Cx)
            copyto!(Cϕ, new_Cϕ)
            copyto!(χ, new_χ)
            copyto!(χϕ, new_χϕ)

        end
    end

    if verbose
        @warn "DMFT loop didn't converge (this is normal for the chaotic phases)."
    end

    if minimal_return
        return Array(new_Cϕ), Array(new_χϕ)
    else
        return Array(new_mx), Array(new_mϕ), Array(new_Cx), Array(new_Cϕ), Array(new_χ), Array(new_χϕ)
    end
end

# ======================== Nonstationary DMFT end =====================

## This version is much faster if we only need the stationary statistics
# Do not use Float32 for the Novikov's theorem-based algorithm. 
# The Novikov's theorem-based computation is numerically unstable in Float32.

# For hardware, we recommend using NVIDIA A100/H100/H200/B100/B200, AMD Instinct MI300X, or 
# later models for best FP64 performance.

# Do not use RTX PRO 6000 Blackwell, L40S, or similar GPUs, because they have very poor FP64 performance.

# ========================= Stationary DMFT =====================================

struct DMFTStationaryRateModel_CUDA
    N::Int64
    J0::Float64
    g::Float64
    τchn::Float64
    τrec::Float64
    τcon::Float64
    τdiv::Float64
    Ttot::Float64
    dt::Float64
    nTime::Int64
    nTraj_list::Tuple{Vararg{Int64}}
    nIte_list::Tuple{Vararg{Int64}}
    damp_R::Tuple{Vararg{Float64}}
    damp_C::Tuple{Vararg{Float64}}
    threshold::Float64
    ϕ::Function
    ϕ_prime::Function
    remove_χϕ_ft_spike::Bool
    χϕ_ft_spike_ratio::Float64
    enforce_odd_symmetry::Bool # Enforce zero τchn mean-feedback for τchn<0 to prevent numerical instability
end


function CreateStationaryDMFTRateModel(N::Integer,
    J0::Real, g::Real, τ::Tuple{Vararg{Real}},
    Ttot::Real,
    nIte_list::Tuple{Vararg{Integer}},
    nTraj_list::Tuple{Vararg{Integer}},
    damp_R::Tuple{Vararg{Real}},
    damp_C::Tuple{Vararg{Real}};
    dt::Real=0.1,
    threshold::Real=1e-4,
    ϕ::Function=tanh,
    ϕ_prime::Function=(x -> 1 - tanh(x)^2),
    remove_χϕ_ft_spike::Bool=false,
    χϕ_ft_spike_ratio::Real=2.0,
    enforce_odd_symmetry::Bool=true
)

    τchn, τrec, τcon, τdiv = τ_parser(τ) .|> Float64

    nTime_calc = round(Int64, Ttot / dt)
    N = Int64(N)
    nTime = Int64(nTime_calc)
    J0 = Float64(J0)
    g = Float64(g)
    Ttot = Float64(Ttot)
    nIte_list = Int64.(nIte_list)
    nTraj_list = Int64.(nTraj_list)
    dt = Float64(dt)
    damp_R = Float64.(damp_R)
    damp_C = Float64.(damp_C)
    threshold = Float64(threshold)
    χϕ_ft_spike_ratio = Float64(χϕ_ft_spike_ratio)
    isfinite(χϕ_ft_spike_ratio) && χϕ_ft_spike_ratio > 1.0 ||
        throw(ArgumentError("χϕ_ft_spike_ratio must be finite and greater than 1."))

    return DMFTStationaryRateModel_CUDA(N, J0, g,
        τchn, τrec, τcon, τdiv,
        Ttot, dt, nTime,
        nTraj_list, nIte_list,
        damp_R, damp_C, threshold,
        ϕ, ϕ_prime,
        remove_χϕ_ft_spike,
        χϕ_ft_spike_ratio,
        enforce_odd_symmetry)
end

function _stationary_mainloop_memory_terms(model::DMFTStationaryRateModel_CUDA)
    nT = iseven(model.nTime) ? model.nTime + 1 : model.nTime
    nFreq = nT ÷ 2 + 1
    pad = round(Int64, Float64(nT) / 2.5)
    nTimePad = nT + pad
    max_nTraj = maximum(model.nTraj_list)

    b_f64 = 8.0
    b_c64 = 16.0

    mem_global = 6.0 * Float64(nT) * b_f64
    mem_eta = Float64(max_nTraj) * Float64(nT) * b_f64
    mem_sample_ws = Float64(nFreq) * b_f64 +
                    Float64(max_nTraj) * Float64(nFreq) * b_c64
    mem_integ_ws = (
        3.0 * Float64(max_nTraj) * Float64(nTimePad) +
        2.0 * Float64(max_nTraj) * Float64(nT) +
        2.0 * Float64(max_nTraj) +
        Float64(nTimePad) + 2.0 * Float64(nT) + 1.0
    ) * b_f64
    mem_fft_temp = Float64(max_nTraj) * Float64(nFreq) * b_c64 +
                   Float64(max_nTraj) * Float64(nT) * b_f64 +
                   Float64(nT) * b_f64
    mem_iter_cols = 2.0 * Float64(nT) * b_f64

    return (
        nT=nT,
        nFreq=nFreq,
        max_nTraj=max_nTraj,
        b_f64=b_f64,
        b_c64=b_c64,
        shared_bytes=mem_global + mem_eta + mem_sample_ws + mem_integ_ws +
                     mem_fft_temp + mem_iter_cols,
    )
end

function _estimate_stationary_mainloop_fn_memory(model::DMFTStationaryRateModel_CUDA)
    mem = _stationary_mainloop_memory_terms(model)
    mem_chiphi_ws = (
        3.0 * Float64(mem.max_nTraj) * Float64(mem.nFreq) * mem.b_c64 +
        (Float64(mem.nT) + Float64(mem.nFreq)) * mem.b_f64 +
        2.0 * Float64(mem.nFreq) * mem.b_c64
    )
    mem_chiphi_raw = Float64(mem.nT) * mem.b_f64
    return 1.2 * (mem.shared_bytes + mem_chiphi_ws + mem_chiphi_raw)
end

function _estimate_stationary_mainloop_timeinteg_memory(model::DMFTStationaryRateModel_CUDA)
    mem = _stationary_mainloop_memory_terms(model)
    mem_chiphi_ws = (
        Float64(mem.max_nTraj) * Float64(mem.nT) +
        2.0 * Float64(mem.max_nTraj) +
        Float64(mem.nT)
    ) * mem.b_f64
    return 1.2 * (mem.shared_bytes + mem_chiphi_ws)
end

function _check_stationary_mainloop_memory(est_peak_bytes::Float64;
    verbose::Bool=false)
    GC.gc(true)
    CUDA.reclaim()
    free_mem_bytes, _ = CUDA.memory_info()
    est_peak_gb = round(est_peak_bytes / 1024^3, digits=2)
    free_mem_gb = round(free_mem_bytes / 1024^3, digits=2)

    if verbose
        println("Estimated peak VRAM = $(est_peak_gb)GB; free VRAM = $(free_mem_gb)GB.")
        flush(stdout)
    end

    if est_peak_bytes > free_mem_bytes
        error_msg = "Out of Memory Risk: Estimated peak VRAM is $(est_peak_gb)GB, " *
                    "but only $(free_mem_gb)GB is free. Reduce Ttot, increase dt, or lower max(nTraj_list)."
        throw(ArgumentError(error_msg))
    end

    return nothing
end

@inline function _snapshot_float_token(x::Real)
    s = string(round(Float64(x), digits=6))
    return replace(replace(s, "-" => "m"), "." => "p")
end

function _save_stationary_iteration_snapshot(path::AbstractString,
    Cϕ::AbstractVector{<:Real},
    χϕ::AbstractVector{<:Real},
    dt::Float64)

    mkpath(dirname(path))
    t = range(0.0; step=dt, length=length(Cϕ))
    fig = Figure(size=(1000, 500))

    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="Cϕ")
    lines!(ax1, t, Cϕ, color=:dodgerblue3)

    ax2 = Axis(fig[1, 2], xlabel="t", ylabel="χϕ")
    lines!(ax2, t, χϕ, color=:firebrick3)

    save(path, fig)

    return nothing
end


struct SampleEtaFFTWorkspace
    λϕ::CuVector{Float64}
    ξ::CuMatrix{ComplexF64}
end

function SampleEtaFFTWorkspace(nTraj::Int64, nT::Int64)
    nFreq = nT ÷ 2 + 1
    return SampleEtaFFTWorkspace(
        CUDA.zeros(Float64, nFreq),
        CuArray{ComplexF64}(undef, nTraj, nFreq),
    )
end

function sample_η_fft!(η::CuMatrix{Float64}, ws::SampleEtaFFTWorkspace,
    model::DMFTStationaryRateModel_CUDA, Cϕ::CuVector{Float64}, mϕ::Float64
)
    nTraj, nT = size(η)
    nFreq = nT ÷ 2 + 1
    size(ws.λϕ, 1) == nFreq || throw(ArgumentError("ws.λϕ has wrong length."))
    size(ws.ξ) == (nTraj, nFreq) || throw(ArgumentError("ws.ξ has wrong size."))

    nT_f = Float64(nT)
    A = 1.0 - model.τcon - model.τdiv

    copyto!(ws.λϕ, real.(CUFFT.rfft(Cϕ))) # FFT of first column gives eigenvalues
    ws.λϕ .*= A * model.g^2
    ws.λϕ[1:1] .+= model.N * model.g^2 * model.τcon * nT_f * mϕ^2
    ws.λϕ .= max.(ws.λϕ, 0.0)

    CUDA.randn!(ws.ξ)
    ws.ξ[:, 1] .= real.(ws.ξ[:, 1]) .* sqrt(2.0)
    if iseven(nT)
        ws.ξ[:, end] .= real.(ws.ξ[:, end]) .* sqrt(2.0)
        # Important: * sqrt(2.0) to adjust for the loss of variance
    end

    ws.ξ .*= reshape(sqrt.(ws.λϕ), 1, nFreq) .* sqrt(nT_f)
    copyto!(η, CUFFT.irfft(ws.ξ, nT, 2))
    return η
end

struct ComputeChiPhiFNWorkspace
    Cη::CuVector{Float64}
    ϕ_ft::CuMatrix{ComplexF64}
    η_ft::CuMatrix{ComplexF64}
    cross_ft::CuMatrix{ComplexF64}
    Cϕη_ft::CuVector{ComplexF64}
    Cη_ft::CuVector{Float64}
    χϕ_ft::CuVector{ComplexF64}
end

function ComputeChiPhiFNWorkspace(nTraj::Int64, nTime::Int64)
    nFreq = nTime ÷ 2 + 1
    return ComputeChiPhiFNWorkspace(
        CUDA.zeros(Float64, nTime),
        CuArray{ComplexF64}(undef, nTraj, nFreq),
        CuArray{ComplexF64}(undef, nTraj, nFreq),
        CuArray{ComplexF64}(undef, nTraj, nFreq),
        CuArray{ComplexF64}(undef, nFreq),
        CUDA.zeros(Float64, nFreq),
        CuArray{ComplexF64}(undef, nFreq),
    )
end

function zero_χϕ_ft_spikes_after!(
    χϕ_ft::CuVector{ComplexF64};
    spike_ratio::Float64=2.0
)
    nFreq = length(χϕ_ft)

    spike_ratio = Float64(spike_ratio)
    isfinite(spike_ratio) && spike_ratio > 1.0 ||
        throw(ArgumentError("spike_ratio must be finite and greater than 1."))

    start_idx = nFreq ÷ 3 + 1
    tail = @view χϕ_ft[start_idx:nFreq]
    tail_power_mean = sum(abs2, tail) / Float64(length(tail))
    @. tail = ifelse(
        abs2(tail) > spike_ratio * tail_power_mean,
        ComplexF64(0.0),
        tail,
    )
    return χϕ_ft
end

function compute_χϕ_FN!(χϕ::CuVector{Float64}, ws::ComputeChiPhiFNWorkspace,
    model::DMFTStationaryRateModel_CUDA, ϕx::AbstractMatrix{Float64}, η::CuMatrix{Float64},
    mϕ::Float64, Cϕ::CuVector{Float64}, scratch::CuMatrix{Float64}
)
    nTraj, nTime = size(ϕx)

    A = 1.0 - model.τcon - model.τdiv
    dt = model.dt
    sqrt2π = sqrt(2.0 * pi)
    scale = dt / sqrt2π
    dω = 1.0 / (Float64(nTime) * dt)

    @. ws.Cη = (A * model.g^2) * Cϕ + (model.N * model.g^2 * model.τcon * mϕ^2)

    copyto!(scratch, ϕx)
    copyto!(ws.ϕ_ft, CUFFT.rfft(scratch, 2))
    copyto!(ws.η_ft, CUFFT.rfft(η, 2))
    @. ws.ϕ_ft *= scale
    @. ws.η_ft *= scale

    @. ws.cross_ft = ws.ϕ_ft * conj(ws.η_ft)
    ws.Cϕη_ft .= vec(sum(ws.cross_ft; dims=1))
    @. ws.Cϕη_ft *= (sqrt2π / Float64(nTraj))

    copyto!(ws.Cη_ft, real.(CUFFT.rfft(ws.Cη)))
    @. ws.Cη_ft *= scale

    @. ws.χϕ_ft = ifelse(abs(ws.Cη_ft) != 0.0, sqrt2π * ws.Cϕη_ft / ws.Cη_ft, ComplexF64(0.0))

    if model.remove_χϕ_ft_spike
        zero_χϕ_ft_spikes_after!(
            ws.χϕ_ft;
            spike_ratio=model.χϕ_ft_spike_ratio,
        )
    end

    copyto!(χϕ, CUFFT.brfft(ws.χϕ_ft, nTime))
    χϕ .*= (dω^2 / sqrt2π)
    return χϕ
end

function _compute_circulant_col(traj::AbstractMatrix{Float64}, scratch::CuMatrix{Float64})
    nTraj, nT = size(traj)

    copyto!(scratch, traj)
    X_fft = CUFFT.rfft(scratch, 2)
    @. X_fft = abs2(X_fft)
    auto = CUFFT.irfft(X_fft, nT, 2)

    col = vec(sum(auto; dims=1))
    @. col /= (Float64(nT) * Float64(nTraj))
    return col
end

struct IntegXtrajStationaryWorkspace
    pad::Int64
    nTimePad::Int64
    x_traj::CuMatrix{Float64}
    ϕx::CuMatrix{Float64}
    traj_out::CuMatrix{Float64}
    η_pad::CuMatrix{Float64}
    ϕ_mean::CuVector{Float64}
    x0::CuVector{Float64}
    f::CuVector{Float64}
    scalar_buf::CuVector{Float64}
    χ_rev::CuVector{Float64}
    w_work::CuVector{Float64}
end

function IntegXtrajStationaryWorkspace(nTraj::Int64, nTime::Int64)
    pad = round(Int64, Float64(nTime) / 2.5)
    nTimePad = nTime + pad
    return IntegXtrajStationaryWorkspace(
        pad,
        nTimePad,
        CUDA.zeros(Float64, nTraj, nTimePad),
        CUDA.zeros(Float64, nTraj, nTimePad),
        CUDA.zeros(Float64, nTraj, nTime),
        CUDA.zeros(Float64, nTraj, nTimePad),
        CUDA.zeros(Float64, nTimePad),
        CUDA.zeros(Float64, nTraj),
        CUDA.zeros(Float64, nTraj),
        CUDA.zeros(Float64, 1),
        CUDA.zeros(Float64, nTime),
        CUDA.zeros(Float64, nTime)
    )
end

function integ_xtraj_stationary!(ws::IntegXtrajStationaryWorkspace,
    model::DMFTStationaryRateModel_CUDA, χϕ::CuVector{Float64},
    η::CuMatrix{Float64}, μx::Float64, Δx::Float64
)
    nTraj, nTime = size(η)
    ws.nTimePad == nTime + ws.pad || throw(ArgumentError("workspace nTime mismatch."))
    size(ws.x_traj, 1) == nTraj || throw(ArgumentError("workspace nTraj mismatch."))

    B = model.τrec - 2.0 * model.τchn
    C1 = model.dt * model.g^2 * B
    if model.enforce_odd_symmetry
        # Enforce zero τchn mean-feedback for τchn<0 to prevent numerical instability
        C2 = model.τchn < 0 ? 0.0 : model.dt * model.N * model.τchn * model.g^2
    else
        C2 = model.dt * model.N * model.τchn * model.g^2
    end
    C3 = model.N * model.J0
    inv_nTraj = 1.0 / Float64(nTraj)

    CUDA.randn!(ws.x0)
    @. ws.x0 = sqrt(Δx) * ws.x0 + μx
    copyto!(@view(ws.x_traj[:, 1]), ws.x0)

    copyto!(@view(ws.η_pad[:, 1:ws.pad]), @view(η[:, (nTime-ws.pad+1):nTime]))
    copyto!(@view(ws.η_pad[:, (ws.pad+1):ws.nTimePad]), η)

    x1 = @view ws.x_traj[:, 1]
    ϕ1 = @view ws.ϕx[:, 1]
    @. ϕ1 = model.ϕ(x1)
    sum!(@view(ws.ϕ_mean[1:1]), ϕ1)
    ws.ϕ_mean[1:1] .*= inv_nTraj

    α = exp(-model.dt)
    β = 1.0 - α
    copyto!(ws.χ_rev, χϕ)
    reverse!(ws.χ_rev)

    @inbounds for t in 1:(ws.nTimePad-1)
        copyto!(ws.f, @view ws.η_pad[:, t])
        K = min(t, nTime)
        if K > 1
            w = @view ws.χ_rev[(nTime-K+1):nTime]
            wK = @view ws.w_work[1:K]
            copyto!(wK, w)
            wK[1:1] .*= 0.5
            wK[K:K] .*= 0.5

            Φ = @view ws.ϕx[:, (t-K+1):t]
            mul!(ws.f, Φ, wK, C1, 1.0)

            ϕ_vec = @view ws.ϕ_mean[(t-K+1):t]
            sum!(ws.scalar_buf, wK .* ϕ_vec)
            ϕ_t = @view ws.ϕ_mean[t:t]
            @. ws.f += C2 * ws.scalar_buf + C3 * ϕ_t
        else
            ϕ_t = @view ws.ϕ_mean[t:t]
            @. ws.f += C3 * ϕ_t
        end

        x_t = @view ws.x_traj[:, t]
        x_next = @view ws.x_traj[:, t+1]
        @. x_next = α * x_t + β * ws.f

        ϕ_next = @view ws.ϕx[:, t+1]
        @. ϕ_next = model.ϕ(x_next)
        mean_next = @view ws.ϕ_mean[(t+1):(t+1)]
        sum!(mean_next, ϕ_next)
        mean_next .*= inv_nTraj
    end

    return nothing
end


# For debug. Initialize using direct simulation.
# function initialize_mC_circular(model::DMFTStationaryRateModel_CUDA; N::Integer=model.N, 
#     T::Real=300.0, burn::Real=50.0, n_samples::Int64=48,
#     exclude_bimodal::Bool=false)

#     if haskey(ENV, "SLURM_MEM_PER_NODE")
#         check_memory = true
#         available_gb = parse(Float64, ENV["SLURM_MEM_PER_NODE"]) / 1024.0
#     elseif haskey(ENV, "SLURM_MEM_PER_CPU") && haskey(ENV, "SLURM_CPUS_ON_NODE")
#         check_memory = true
#         available_gb = parse(Float64, ENV["SLURM_MEM_PER_CPU"]) * parse(Float64, ENV["SLURM_CPUS_ON_NODE"]) / 1024.0
#     else
#         # not on Slurm node, skip memory guard
#         check_memory = false
#     end

#     nT = iseven(model.nTime) ? model.nTime + 1 : model.nTime
#     M = (nT + 1) ÷ 2
#     saveat = model.dt
#     tspan = (0.0, T)

#     # Pad length for linear convolution via FFT
#     K = nextpow(2, 2M - 1)

#     # Avoid OOM kill
#     if check_memory
#         Nf = Float64(N)
#         Mf = Float64(M)
#         Kf = Float64(K)
#         nf = Float64(n_samples)
#         b = 8.0 # 8 bytes for Float64
#         nSteps  = Float64(round(Int64, T / model.dt))
#         nthread = Float64(Threads.nthreads())

#         bytes_J = nf * Nf^2 * b
#         bytes_ode_sol = nthread * nSteps * Nf * b

#         bytes_X = 2.0 * Nf * Mf * b
#         bytes_padded = 2.0 * Nf * Kf * b 
#         bytes_fft = 2.0 * Nf * (Kf / 2.0 + 1.0) * 16.0  # ComplexF64 FFT outputs (16 bytes each)

#         bytes_out_tmp = nthread * (bytes_X + bytes_padded + bytes_fft)
#         bytes_results = nf * (2.0 + 2.0 * Mf) * b

#         estimated_gb = (bytes_J + bytes_ode_sol + bytes_out_tmp + bytes_results) / 1024.0^3

#         if estimated_gb > 0.85 * available_gb
#             error("Insufficient Slurm job memory:\n" *
#                   "  Estimated: $(round(estimated_gb, digits=2)) GB\n" *
#                   "  Available: $(round(available_gb, digits=2)) GB\n")
#         end
#     end

#     τ = (model.τchn, model.τrec, model.τcon, model.τdiv)

#     J_array = Vector{Matrix{Float64}}()
#     sizehint!(J_array, n_samples)

#     while length(J_array) < n_samples
#         J = CreateJ(Int64(N), model.J0, model.g, τ) |> Matrix{Float64}
#         if exclude_bimodal
#             p_value = bimodal_test(J; nInits=24, burn_in=20.0, T=300.0,
#                 saveat=saveat, parallel=true)
#             p_value > 0.05 || continue
#         end
#         push!(J_array, Matrix{Float64}(J))
#     end

#     fft_plan = plan_rfft(zeros(Float64, N, K), 2)

#     function prob_func(prob, i, repeat)
#         u0 = 0.5 .* randn(Float64, N)
#         cache = zeros(Float64, N) # Thread-local cache
#         return remake(prob, u0=u0, p=(J_array[i], cache))
#     end

#     function ODEFunc!(du, u, p, t)
#         J, ϕu = p
#         @. ϕu = model.ϕ(u)
#         mul!(du, J, ϕu)
#         du .-= u
#         return nothing
#     end

#     function output_func(sol, i)
#         times = sol.t
#         burn_idx = findfirst(>=(burn), times)
#         X = reduce(hcat, @view sol.u[burn_idx:burn_idx+M-1])   # N × M
#         ϕX = model.ϕ.(X)

#         mx = Float64(mean(X))
#         mϕ = Float64(mean(ϕX))

#         X_padded = zeros(Float64, N, K)
#         ϕX_padded = zeros(Float64, N, K)
#         X_padded[:, 1:M] .= X
#         ϕX_padded[:, 1:M] .= ϕX

#         X_fft = fft_plan * X_padded
#         ϕX_fft = fft_plan * ϕX_padded

#         X_ps_sum = dropdims(sum(abs2, X_fft, dims=1), dims=1)
#         ϕX_ps_sum = dropdims(sum(abs2, ϕX_fft, dims=1), dims=1)

#         Cx_raw = irfft(X_ps_sum, K)
#         Cϕ_raw = irfft(ϕX_ps_sum, K)

#         denom = Float64.(N .* (M:-1:1))
#         Cx = Cx_raw[1:M] ./ denom
#         Cϕ = Cϕ_raw[1:M] ./ denom

#         return ((mx, mϕ, Cx, Cϕ), false)
#     end

#     prob = ODEProblem(ODEFunc!, zeros(Float64, N), tspan,
#                       (zeros(Float64, N, N), zeros(Float64, N)))

#     ensemble_problem = EnsembleProblem(prob; prob_func=prob_func,
#                                        output_func=output_func)
#     sim = solve(ensemble_problem, Tsit5(), EnsembleThreads();
#                 trajectories=n_samples, saveat=saveat)

#     μx = 0.0
#     μϕ = 0.0
#     Cx_mean = zeros(Float64, M)
#     Cϕ_mean = zeros(Float64, M)

#     for i in 1:n_samples
#         mx, mϕ, Cx, Cϕ = sim.u[i]
#         μx += mx
#         μϕ += mϕ
#         Cx_mean .+= Cx
#         Cϕ_mean .+= Cϕ
#     end

#     μx /= n_samples
#     μϕ /= n_samples
#     Cx_mean ./= n_samples
#     Cϕ_mean ./= n_samples

#     Cx_col = vcat(Cx_mean, reverse(Cx_mean[2:end]))
#     Cϕ_col = vcat(Cϕ_mean, reverse(Cϕ_mean[2:end]))

#     return μx, μϕ, CuArray{Float64}(Cx_col), CuArray{Float64}(Cϕ_col)
# end

function initialize_mC_circular(model::DMFTStationaryRateModel_CUDA)
    nT = iseven(model.nTime) ? model.nTime + 1 : model.nTime
    d = collect(0:(nT-1))
    d = min.(d, nT .- d) .* model.dt

    C_seed = @. exp(-((d / 15.0)^2))
    C_seed[1] += 1e-4

    Cx_col = CuArray(C_seed)
    Cϕ_col = copy(Cx_col)
    if model.τchn > 0.0
        mx = 0.5
        mϕ = 0.2
    else
        mx = 0.0
        mϕ = 0.0
    end
    return mx, mϕ, Cx_col, Cϕ_col
end


function DMFTStationaryMainloop(model::DMFTStationaryRateModel_CUDA;
    verbose::Bool=true, return_history::Bool=false, minimal_return::Bool=false)

    if 1.0 - model.τcon - model.τdiv < 0.0
        return nothing, nothing, nothing, nothing, nothing, nothing
        @warn "Invalid parameters: τcon + τdiv must be less than 1.0."
    end

    _check_stationary_mainloop_memory(
        _estimate_stationary_mainloop_fn_memory(model),
        verbose=verbose
    )

    nT = iseven(model.nTime) ? model.nTime + 1 : model.nTime

    mx, mϕ, Cx, Cϕ = initialize_mC_circular(model)

    χϕ = CUDA.zeros(Float64, nT)

    new_Cx = similar(Cx)
    new_Cϕ = similar(Cϕ)
    new_χϕ = similar(χϕ)

    # ---- histories ----
    if return_history
        mx_history = Float64[]
        mϕ_history = Float64[]
        Cx_history = Vector{Float64}[]
        Cϕ_history = Vector{Float64}[]
        χϕ_history = Vector{Float64}[]
    end

    to_cpu_col(v::CuVector{Float64}) = Array(@view(v[1:model.nTime]))
    # snapshot_prefix = joinpath("results", "images",
    #     "DMFTStationaryMainloop_" *
    #     string(time_ns()))

    Ite_count = 0

    for block in eachindex(model.nIte_list)

        nIte = model.nIte_list[block]
        nTraj = model.nTraj_list[block]
        damp_R = model.damp_R[block]
        damp_C = model.damp_C[block]

        η = CUDA.zeros(Float64, nTraj, nT)
        sample_ws = SampleEtaFFTWorkspace(nTraj, nT)
        integ_ws = IntegXtrajStationaryWorkspace(nTraj, nT)
        χϕ_ws = ComputeChiPhiFNWorkspace(nTraj, nT)

        for _ in 1:nIte
            Ite_count += 1

            sample_η_fft!(η, sample_ws, model, Cϕ, mϕ)

            Cx0 = Array(@view(Cx[1:1]))[1]
            Δx = max(Cx0 - mx^2, 1e-4)

            integ_xtraj_stationary!(integ_ws, model, χϕ, η, mx, Δx)

            x_traj = @view integ_ws.x_traj[:, (integ_ws.pad+1):integ_ws.nTimePad]
            ϕx = @view integ_ws.ϕx[:, (integ_ws.pad+1):integ_ws.nTimePad]
            new_mx = damp_C * mx + (1.0 - damp_C) * Float64(mean(x_traj))
            new_mϕ = damp_C * mϕ + (1.0 - damp_C) * Float64(mean(ϕx))

            Cx_col = _compute_circulant_col(x_traj, integ_ws.traj_out)
            @. new_Cx = damp_C * Cx + (1.0 - damp_C) * Cx_col

            Cϕ_col = _compute_circulant_col(ϕx, integ_ws.traj_out)
            @. new_Cϕ = damp_C * Cϕ + (1.0 - damp_C) * Cϕ_col

            compute_χϕ_FN!(new_χϕ, χϕ_ws, model, ϕx, η, mϕ, Cϕ, integ_ws.traj_out)
            @. new_χϕ = damp_R * χϕ + (1.0 - damp_R) * new_χϕ

            # _save_stationary_iteration_snapshot(
            #     snapshot_prefix * "_iter" * string(Ite_count) * ".png",
            #     to_cpu_col(new_Cϕ),
            #     to_cpu_col(new_χϕ),
            #     model.dt
            # )

            diff_C = Float64(norm(new_Cϕ .- Cϕ) /
                             max(norm(Cϕ), eps(Float64)) /
                             (1.0 - damp_C))

            if verbose
                println("Iteration $Ite_count, normalized |ΔC| = $diff_C")
                flush(stdout)
            end

            mx = new_mx
            mϕ = new_mϕ
            copyto!(Cx, new_Cx)
            copyto!(Cϕ, new_Cϕ)
            copyto!(χϕ, new_χϕ)

            if return_history
                push!(mx_history, mx)
                push!(mϕ_history, mϕ)
                push!(Cx_history, to_cpu_col(Cx))
                push!(Cϕ_history, to_cpu_col(Cϕ))
                push!(χϕ_history, to_cpu_col(χϕ))
            end

            if diff_C < model.threshold
                Cx_out = to_cpu_col(Cx)
                Cϕ_out = to_cpu_col(Cϕ)
                χϕ_out = to_cpu_col(χϕ)

                if return_history
                    return mx, mϕ,
                    Cx_out,
                    Cϕ_out,
                    χϕ_out,
                    (
                        mx_history,
                        mϕ_history,
                        Cx_history,
                        Cϕ_history,
                        χϕ_history,
                    )
                elseif minimal_return
                    return Cϕ_out, χϕ_out
                else
                    return mx, mϕ,
                    Cx_out,
                    Cϕ_out,
                    χϕ_out
                end
            end
        end
    end

    if verbose
        @warn "DMFT loop didn't converge."
    end

    Cx_out = to_cpu_col(Cx)
    Cϕ_out = to_cpu_col(Cϕ)
    χϕ_out = to_cpu_col(χϕ)

    if return_history
        return mx, mϕ,
        Cx_out,
        Cϕ_out,
        χϕ_out,
        (
            mx_history,
            mϕ_history,
            Cx_history,
            Cϕ_history,
            χϕ_history,
        )
    elseif minimal_return
        return Cϕ_out, χϕ_out
    else
        return mx, mϕ,
        Cx_out,
        Cϕ_out,
        χϕ_out
    end

end


## Another version using time-domain integration for χϕ (similar to Zou and Huang 2024)

struct IntegChiPhiStationaryWorkspace
    χϕ_traj::CuMatrix{Float64}
    χ_prev::CuVector{Float64}
    χ_next::CuVector{Float64}
    χϕ_mean_rev::CuVector{Float64}
end

function IntegChiPhiStationaryWorkspace(nTraj::Int64, nTime::Int64)
    return IntegChiPhiStationaryWorkspace(
        CUDA.zeros(Float64, nTraj, nTime),
        CUDA.zeros(Float64, nTraj),
        CUDA.zeros(Float64, nTraj),
        CUDA.zeros(Float64, nTime),
    )
end


function integ_χϕ_stationary!(χϕ_mean::CuVector{Float64},
    ws::IntegChiPhiStationaryWorkspace,
    model::DMFTStationaryRateModel_CUDA,
    x_traj::AbstractMatrix{Float64}
)
    nTraj, nTime = size(x_traj)

    B = model.τrec - 2.0 * model.τchn
    C1 = model.dt * model.g^2 * B

    α = exp(-model.dt)
    β = 1.0 - α
    inv_nTraj = 1.0 / Float64(nTraj)

    fill!(ws.χϕ_traj, 0.0)
    fill!(χϕ_mean, 0.0)
    fill!(ws.χϕ_mean_rev, 0.0)

    # Ito convention: equal-time response vanishes, so the first nonzero
    # response is stored at the second time bin.
    χ_prev = ws.χ_prev
    χ_next = ws.χ_next
    fill!(χ_prev, 0.0)
    fill!(χ_next, 0.0)

    fill!(χ_next, 1.0)
    x1 = @view x_traj[:, 2]
    χϕ1 = @view ws.χϕ_traj[:, 2]
    @. χϕ1 = χ_next * model.ϕ_prime(x1)
    χϕ_mean1 = @view χϕ_mean[2:2]
    sum!(χϕ_mean1, χϕ1)
    χϕ_mean1 .*= inv_nTraj
    copyto!(@view(ws.χϕ_mean_rev[(nTime-1):(nTime-1)]), χϕ_mean1)
    χ_prev, χ_next = χ_next, χ_prev

    @inbounds for τ in 2:(nTime-1)
        χϕ_hist = @view ws.χϕ_traj[:, 1:τ]
        χϕ_first = @view ws.χϕ_traj[:, 1]
        χϕ_last = @view ws.χϕ_traj[:, τ]
        χϕ_next = @view ws.χϕ_traj[:, τ+1]
        x_next = @view x_traj[:, τ+1]
        mean_hist_rev = @view ws.χϕ_mean_rev[(nTime-τ+1):nTime]
        mean_first = @view χϕ_mean[1:1]
        mean_last = @view χϕ_mean[τ:τ]

        mul!(χ_next, χϕ_hist, mean_hist_rev)
        @. χ_next -= 0.5 * (χϕ_last * mean_first + χϕ_first * mean_last)
        @. χ_next = α * χ_prev + β * C1 * χ_next
        @. χϕ_next = χ_next * model.ϕ_prime(x_next)
        χϕ_mean_next = @view χϕ_mean[(τ+1):(τ+1)]
        sum!(χϕ_mean_next, χϕ_next)
        χϕ_mean_next .*= inv_nTraj
        copyto!(@view(ws.χϕ_mean_rev[(nTime-τ):(nTime-τ)]), χϕ_mean_next)

        χ_prev, χ_next = χ_next, χ_prev
    end

    return χϕ_mean
end


function DMFTStationaryMainloop_TimeInteg(model::DMFTStationaryRateModel_CUDA;
    verbose::Bool=true, return_history::Bool=false, minimal_return::Bool=false)

    if 1.0 - model.τcon - model.τdiv < 0.0
        return nothing, nothing, nothing, nothing, nothing, nothing
        @warn "Invalid parameters: τcon + τdiv must be less than 1.0."
    end

    _check_stationary_mainloop_memory(
        _estimate_stationary_mainloop_timeinteg_memory(model),
        verbose=verbose
    )

    nT = iseven(model.nTime) ? model.nTime + 1 : model.nTime

    mx, mϕ, Cx, Cϕ = initialize_mC_circular(model)

    χϕ = CUDA.zeros(Float64, nT)

    new_Cx = similar(Cx)
    new_Cϕ = similar(Cϕ)
    new_χϕ = similar(χϕ)

    if return_history
        mx_history = Float64[]
        mϕ_history = Float64[]
        Cx_history = Vector{Float64}[]
        Cϕ_history = Vector{Float64}[]
        χϕ_history = Vector{Float64}[]
    end

    to_cpu_col(v::CuVector{Float64}) = Array(@view(v[1:model.nTime]))

    Ite_count = 0

    for block in eachindex(model.nIte_list)

        nIte = model.nIte_list[block]
        nTraj = model.nTraj_list[block]
        damp_R = model.damp_R[block]
        damp_C = model.damp_C[block]

        η = CUDA.zeros(Float64, nTraj, nT)
        sample_ws = SampleEtaFFTWorkspace(nTraj, nT)
        integ_ws = IntegXtrajStationaryWorkspace(nTraj, nT)
        χϕ_ws = IntegChiPhiStationaryWorkspace(nTraj, nT)

        for _ in 1:nIte
            Ite_count += 1

            sample_η_fft!(η, sample_ws, model, Cϕ, mϕ)

            Cx0 = Array(@view(Cx[1:1]))[1]
            Δx = max(Cx0 - mx^2, 1e-4)

            integ_xtraj_stationary!(integ_ws, model, χϕ, η, mx, Δx)

            x_traj = @view integ_ws.x_traj[:, (integ_ws.pad+1):integ_ws.nTimePad]
            ϕx = @view integ_ws.ϕx[:, (integ_ws.pad+1):integ_ws.nTimePad]
            new_mx = damp_C * mx + (1.0 - damp_C) * Float64(mean(x_traj))
            new_mϕ = damp_C * mϕ + (1.0 - damp_C) * Float64(mean(ϕx))

            Cx_col = _compute_circulant_col(x_traj, integ_ws.traj_out)
            @. new_Cx = damp_C * Cx + (1.0 - damp_C) * Cx_col

            integ_χϕ_stationary!(new_χϕ, χϕ_ws, model, x_traj)
            @. new_χϕ = damp_R * χϕ + (1.0 - damp_R) * new_χϕ

            Cϕ_col = _compute_circulant_col(ϕx, integ_ws.traj_out)
            @. new_Cϕ = damp_C * Cϕ + (1.0 - damp_C) * Cϕ_col

            diff_C = Float64(norm(new_Cϕ .- Cϕ) /
                             max(norm(Cϕ), eps(Float64)) /
                             (1.0 - damp_C))

            if verbose
                println("Iteration $Ite_count, normalized |ΔC| = $diff_C")
                flush(stdout)
            end

            mx = new_mx
            mϕ = new_mϕ
            copyto!(Cx, new_Cx)
            copyto!(Cϕ, new_Cϕ)
            copyto!(χϕ, new_χϕ)

            if return_history
                push!(mx_history, mx)
                push!(mϕ_history, mϕ)
                push!(Cx_history, to_cpu_col(Cx))
                push!(Cϕ_history, to_cpu_col(Cϕ))
                push!(χϕ_history, to_cpu_col(χϕ))
            end

            if diff_C < model.threshold
                Cx_out = to_cpu_col(Cx)
                Cϕ_out = to_cpu_col(Cϕ)
                χϕ_out = to_cpu_col(χϕ)

                if return_history
                    return mx, mϕ,
                    Cx_out,
                    Cϕ_out,
                    χϕ_out,
                    (
                        mx_history,
                        mϕ_history,
                        Cx_history,
                        Cϕ_history,
                        χϕ_history,
                    )
                elseif minimal_return
                    return Cϕ_out, χϕ_out
                else
                    return mx, mϕ,
                    Cx_out,
                    Cϕ_out,
                    χϕ_out
                end
            end
        end
    end

    if verbose
        @warn "DMFT loop didn't converge."
    end

    Cx_out = to_cpu_col(Cx)
    Cϕ_out = to_cpu_col(Cϕ)
    χϕ_out = to_cpu_col(χϕ)

    if return_history
        return mx, mϕ,
        Cx_out,
        Cϕ_out,
        χϕ_out,
        (
            mx_history,
            mϕ_history,
            Cx_history,
            Cϕ_history,
            χϕ_history,
        )
    elseif minimal_return
        return Cϕ_out, χϕ_out
    else
        return mx, mϕ,
        Cx_out,
        Cϕ_out,
        χϕ_out
    end

end
# ======================== Stationary DMFT end ==================================


end
