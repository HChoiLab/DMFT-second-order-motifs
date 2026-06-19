module Utils
using Random
using LinearAlgebra
using Dates
using Distributed
using DSP, FFTW, LsqFit
using DifferentialEquations
using NonlinearSolve
using SciMLBase: successful_retcode
using StatsBase, RCall
using LoopVectorization
# To install disptest locally, use Julia command R"install.packages('diptest', repos='http://cran.us.r-project.org')"
# When running on a remote cluster, install R via anaconda and run `R.home()` to get
# the directory of R. Then set julia `ENV["R_HOME"]` to that directory and build RCall.
# To install diptest, use anaconda: conda install -c conda-forge r-diptest
R"""library(diptest)"""
export CreateJ, TheoryR, TheoryOutlier, bimodal_test, format_elapsed, extract_stationary_C, 
    find_fixed_point, GaussianTanh!, GaussianTanh_jac!, 
    τ_parser, autocov_fft, PhaseClassifier, ComputeMeanDistance, compute_g, ComputeDInf,
    ComputeLSPR, LSPR_params, tanhLSPR, KSEntropyKYDimension, NumericalAutocorrelation, 
    GaussianPositiveActivation!, ϕ_positive, ϕ_positive_p, NumericalPRD, TheoreticalPRϕ, parameter_token,
    FixedPointSearchConfig, FixedPointSolverCache, FixedPointCandidate, FixedPointSummary,
    FixedPointBatchSummary, FixedPointRepresentativeIndex, set_worker_J!,
    residual_rms, fixed_point_residual!, fixed_point_jacobian!, normalized_distance,
    merge_distinct_fixed_point!, merge_distinct_fixed_points!, distinct_fixed_points,
    fixed_point_stability, is_nonzero_fixed_point, add_symmetric_fixed_points,
    summarize_fixed_points, independent_thread_rngs, solve_fixed_point_batch,
    n_inits_for_batch, distributed_batch_ranges, solve_fixed_point_batch_range,
    restore_representative_index
    
const 🔒 = ReentrantLock()

## ================ Tanh dynamics ==============

function GaussianTanh!(du, u, J, t)
    ϕ = tanh.(u)
    mul!(du, J, ϕ)
    du .-= u
end

function GaussianTanh_jac!(Jac, u, J, t)
    v = 1.0 .- tanh.(u).^2
    Jac .= J .* v'
    Jac[diagind(Jac)] .-= 1.0
end

## ================== General helper functions ===================

function bimodal_test(J::Matrix{Float64}; nInits::Int64=20, 
    burn_in::Float64 = 50.0, T::Float64 = 300.0, saveat::Float64 = 1.0, 
    parallel::Bool=true)
    # It should be noted that the dip test is no perfect for detecting the SC phase.
    # For some oscillatory trajectories that spend longer time at the extremes, the distribution of
    # mean activity across time can also be bimodal.
    
    N = size(J)[1]
    function prob_func(prob, i, repeat)
        remake(prob, u0 = randn(N))
    end

    tspan = (0.0, burn_in + T)
    prob = ODEProblem(GaussianTanh!, randn(N), tspan, J)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

    if parallel
        sim = solve(ensemble_prob, Tsit5();
            trajectories=nInits, saveat=saveat)
    else
        sim = solve(ensemble_prob, Tsit5(), EnsembleSerial();
            trajectories=nInits, saveat=saveat)
    end
    len_traj = length(sim[1].t) - findfirst(t -> t >= burn_in, sim[1].t) + 1
    burn_idx = findfirst(t -> t >= burn_in, sim[1].t)
    X_tensor = zeros(N, len_traj, nInits)
    if parallel
        Threads.@threads for a in 1:nInits
            X_tensor[:, :, a] .= sim[a][:, burn_idx:end]
        end
    else
        for a in 1:nInits
            X_tensor[:, :, a] .= sim[a][:, burn_idx:end]
        end
    end

    mean_x = dropdims(mean(X_tensor; dims=1), dims=1) |> vec

    lock(🔒)
    p_value = try
        result = rcopy(R"suppressWarnings(dip.test($mean_x))")
        result[:p_value]
    finally
        unlock(🔒)
    end

    return p_value
end

"""
    Compute unbiased autocovariance via FFT for a single real series (length M) 
"""
function autocov_fft(x::AbstractVector{T}) where {T<:Real}
    Tf = float(T)
    M = length(x)
    xf = Tf.(x)
    ac = xcorr(xf)
    ac = @view ac[M:end]          
    denom = Tf.(M .- (0:(M-1)))   
    return ac ./ denom            # unbiased estimator
end

function τ_parser(τ::Tuple{Vararg{Real}})
    τ_chn = τ[1]
    τ_rec = τ[2]
    if length(τ) == 2
        τ_con = abs(τ_chn)
        τ_div = abs(τ_chn)
    elseif length(τ) == 4
        τ_con = τ[3]
        τ_div = τ[4]
    end
    return τ_chn, τ_rec, τ_con, τ_div
end

function format_elapsed(start_time)
    elapsed_seconds = Dates.value(now() - start_time) ÷ 1000
    hours = elapsed_seconds ÷ 3600
    minutes = (elapsed_seconds % 3600) ÷ 60
    seconds = elapsed_seconds % 60
    lpad(hours, 2, "0") * ":" * lpad(minutes, 2, "0") * ":" * lpad(seconds, 2, "0")
end

"""
    Find the nonzero fixed point for strong positive chain motifs.
"""
function find_fixed_point(J::Matrix{Float64}; tspan::Tuple{Float64,Float64}=(0.0, 50.0), tol::Float64=1e-6,
    ϕ::Function=tanh, verbose::Bool=true, n_tail::Int=5,
    tail_window::Float64=2.0)

    n_tail ≥ 2 || throw(ArgumentError("n_tail must be at least 2."))
    tail_window > 0 || throw(ArgumentError("tail_window must be positive."))

    N = size(J, 1)
    t0, t1 = Float64.(tspan)

    u0 = 0.4 * randn(N)
    ode_ϕ_buf = zeros(N)

    function ODEFunc!(du, u, J, t)
        @. ode_ϕ_buf = ϕ(u)
        mul!(du, J, ode_ϕ_buf)
        @. du = du - u
    end

    check_start = max(t0, t1 - tail_window)
    tail_times = collect(range(check_start, t1; length=n_tail))
    tail_hist = Matrix{Float64}(undef, N, n_tail)
    tail_idx = Ref(0)

    cb = FunctionCallingCallback(
        (u, t, integrator) -> begin
            tail_idx[] += 1
            tail_hist[:, tail_idx[]] .= u
        end;
        funcat = tail_times,
        func_everystep = false
    )

    prob = ODEProblem(ODEFunc!, u0, (t0, t1), J)
    solve(prob, Tsit5();
        reltol=1e-6, abstol=1e-6,
        callback=cb, save_everystep=false,
        saveat=Float64[])

    tail_idx[] == n_tail ||
        throw(ErrorException("Expected $n_tail tail samples, got $(tail_idx[])."))

    ϕ_buf = zeros(N)
    residual_buf = zeros(N)
    diff_buf = zeros(N)

    function residual_rms(x)
        @. ϕ_buf = ϕ(x)
        mul!(residual_buf, J, ϕ_buf)
        @. residual_buf = residual_buf - x
        return norm(residual_buf) / sqrt(N)
    end

    x_last = copy(@view tail_hist[:, end])
    max_residual = 0.0
    max_drift = 0.0
    for j in 1:n_tail
        xj = @view tail_hist[:, j]
        max_residual = max(max_residual, residual_rms(xj))
        if j < n_tail
            @. diff_buf = xj - x_last
            max_drift = max(max_drift, norm(diff_buf) / sqrt(N))
        end
    end

    converged = (max_residual ≤ tol) && (max_drift ≤ tol)
    if !converged && verbose
        @warn "Didn't converge to a fixed point." max_residual max_drift tol
    end

    ϕ_last = similar(x_last)
    @. ϕ_last = ϕ(x_last)

    if verbose && mean(abs.(x_last)) < 1e-6
        @warn "The fixed point is very close to zero."
    end

    # normalize sign (so mean(x) ≥ 0)
    if mean(x_last) < 0
        return -x_last, -ϕ_last, converged
    else
        return x_last, ϕ_last, converged
    end
end

function compute_g(g_eff::Float64, τ::Tuple{Vararg{Float64}})
    τchn, τrec, τcon, τdiv = τ_parser(τ)
    A = 1.0 - τcon - τdiv
    B = τrec - 2τchn
    g = g_eff * sqrt(A) / (A + B)
    return isinf(g) ? 100.0 : g
end

"""
Extract the stationary part of the correlation matrix C after discarding the burn-in period for
the nonstationary DMFT iteration.
"""
function extract_stationary_C(C::AbstractMatrix{<:Real}; burn::Real=50.0, dt::Real=0.1, discard_tail::Real=30.0)
    nRow = size(C, 1)
    Idx = round(Int, burn / dt) + 1

    if Idx > nRow
        throw(ArgumentError("Burn-in time $burn exceeds total time $(dt * (nRow - 1))."))
    end

    # Total possible lags after discarding burn-in
    nLags_total = nRow - Idx + 1
    
    n_discard = round(Int, discard_tail / dt)
    
    # Number of elements we actually want to compute and keep
    nLags_keep = nLags_total - n_discard
    
    if nLags_keep <= 0
        throw(ArgumentError("Discard tail time $discard_tail is too large for the 
        available stationary time $((nLags_total - 1) * dt)."))
    end

    times = (0:(nLags_keep - 1)) .* dt
    C_stationary = zeros(Float32, nLags_keep)

    diag_stride = nRow + 1

    Threads.@threads for k in 0:(nLags_keep - 1)
        n_avg = nLags_total - k
        
        # Calculate the starting linear index for the k-th diagonal
        # Equivalent to Cartesian C[Idx + k, Idx]
        start_idx = (Idx + k) + (Idx - 1) * nRow
        
        s = 0.0f0
        # !!! Can't use @turbo here
        @inbounds @simd for j in 0:(n_avg - 1)
            s += C[start_idx + j * diag_stride]
        end
        C_stationary[k + 1] = s / n_avg
    end

    return times, C_stationary
end


## ================ Matrix generation functions =========================

function τ_estimator(J::Matrix{Float64}; parallel::Bool=false)
    N = size(J, 1)
    J0 = mean(J)
    SD = std(J)
    Jt = (J .- J0) ./ SD

    τrec = (sum(Jt .* Jt') - sum(abs2, diag(Jt))) / (N * (N - 1))

    con_sum = 0.0
    row_sums = vec(sum(Jt, dims=2))
    row_sum_sq = vec(sum(abs2, Jt, dims=2))
    τcon = (sum(abs2, row_sums) - sum(row_sum_sq)) / (N^2 * (N - 1))

    col_sums = vec(sum(Jt, dims=1))
    col_sum_sq = vec(sum(abs2, Jt, dims=1))
    τdiv = (sum(abs2, col_sums) - sum(col_sum_sq)) / (N^2 * (N - 1))

    chn_sum = 0.0
    if parallel
        diag_vec = diag(Jt)
        @tturbo for j in 1:N
            s_col = col_sums[j] - Jt[j, j]
            s_row = row_sums[j] - Jt[j, j]
            dot_prod = 0.0
            for i = 1:N
                dot_prod += Jt[i, j] * Jt[j, i]
            end
            chn_sum += s_col * s_row - dot_prod + diag_vec[j]^2
        end
    else
        @inbounds for j in 1:N
            s_col = col_sums[j] - Jt[j, j]
            s_row = row_sums[j] - Jt[j, j]
            @views chn_sum += s_col * s_row - dot(Jt[:, j], Jt[j, :]) + Jt[j, j]^2
        end
    end

    τchn = chn_sum / (N * (N - 1) * (N - 2))
    return τchn, τrec, τcon, τdiv
end

function _createJ_method1(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    return_deviations::Bool=true, parallel::Bool=false)
    τc, τr, τcon, τdiv = τ_parser(τ)

    ηi = repeat(randn(N), 1, N)
    μij = randn(N, N)
    νij = randn(N, N)
    ζij = randn(N, N)

    J = J0 .+ g / sqrt(N) * (
        sign(τc) * sqrt(τcon) * ηi +
        sqrt(τdiv) * ηi' -
        sign(τc) * sqrt(τcon) * μij +
        sqrt(τdiv) * μij' +
        sign(τr) * sqrt(abs(τr) / 2) * νij +
        sqrt(abs(τr) / 2) * νij' +
        sqrt(1 - 2τcon - 2τdiv - abs(τr)) * ζij
    )

    if return_deviations
        τc_hat, τr_hat, τcon_hat, τdiv_hat = τ_estimator(J; parallel=parallel)
        deviations = [τc_hat - τc, τr_hat - τr, τcon_hat - τcon, τdiv_hat - τdiv]
        return J, deviations
    end
    return J
end


function _createJ_method2(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    return_deviations::Bool=true, parallel::Bool=false)
    τc, τr, τcon, τdiv = τ_parser(τ)
    A = 1 - τcon - τdiv
    B = τr - 2τc

    if τcon * τdiv > 0 && τc != 0
        Σ_row = (g^2 / N) * [
            τcon τc
            τc τdiv
        ]

        Lrow = cholesky(Symmetric(Σ_row)).L
        Z1 = randn(N)
        Z2 = randn(N)
        η = Lrow[1, 1] .* Z1
        μ = Lrow[2, 1] .* Z1 .+ Lrow[2, 2] .* Z2
    else
        # det(Σ_row) = 0
        η = sqrt(τcon * g^2 / N) .* randn(N)
        μ = sqrt(τdiv * g^2 / N) .* randn(N)

        # Reduce leakage for large τcon and τdiv
        # but this may create a very small negative bias in τchn and τrec
        if norm(μ) > 1e-12
            η .-= (dot(η, μ) / dot(μ, μ)) .* μ
        elseif norm(η) > 1e-12
            μ .-= (dot(η, μ) / dot(η, η)) .* η
        end
    end

    J = J0 .+ η .+ μ'

    if B != 0
        ν = zeros(Float64, N, N)
        Σ_pair = (g^2 / N) * [
            A B
            B A
        ]
        Lp = cholesky(Symmetric(Σ_pair)).L
        if parallel
            Threads.@threads for i in 1:N
                @inbounds for j in i+1:N
                    z1 = randn()
                    z2 = randn()
                    νij = Lp[1, 1] * z1
                    νji = Lp[2, 1] * z1 + Lp[2, 2] * z2
                    J[i, j] += νij
                    J[j, i] += νji
                end
            end
        else
            @inbounds for i in 1:N, j in i+1:N
                z1 = randn()
                z2 = randn()
                νij = Lp[1, 1] * z1
                νji = Lp[2, 1] * z1 + Lp[2, 2] * z2
                J[i, j] += νij
                J[j, i] += νji
            end
        end
        J[diagind(J)] .+= (g / sqrt(N) * sqrt(A)) .* randn(N)
    else
        J .+= (g / sqrt(N) * sqrt(A)) .* randn(N, N)
    end

    if return_deviations
        τc_hat, τr_hat, τcon_hat, τdiv_hat = τ_estimator(J; parallel=parallel)
        deviations = [τc_hat - τc, τr_hat - τr, τcon_hat - τcon, τdiv_hat - τdiv]
        return J, deviations
    end
    return J
end


"""
Create an N by N connectivity matrix with mean J0 and variance g^2/N. The motifs should be specified 
in τ according to the pattern defined in τ_parser. When only τchn and τrec are specified, the other two 
are default to abs(τchn). Enable quality_control is recommended for N ~ 1000.
"""
function CreateJ(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    verbose=true, quality_control::Bool=true, max_iter::Int64=50,
    parallel::Bool=false)
    τc, τr, τcon, τdiv = τ_parser(τ)

    if abs(τc) > sqrt(τcon * τdiv) || τcon < 0 || τdiv < 0
        if verbose
            @warn "Invalid motif strengths combination."
        end
        return nothing
    end

    if τc == 0 && τr == 0 && τcon == 0 && τdiv == 0
        return J0 .+ g / sqrt(N) .* randn(N, N)
    end

    # Method 1 from Shao et al., 2025
    method_1 =
        isapprox(abs(τc), sqrt(τcon * τdiv); atol=1e-12) &&
        (1 - 2τcon - 2τdiv - abs(τr) ≥ 0) && τcon * τdiv > 0

    if method_1
        if quality_control
            deviations = [1.0, 1.0, 1.0, 1.0]
            nIters = 0
            while any(x -> abs(x) > 3 / N, deviations) && nIters <= max_iter
                nIters += 1
                J, deviations = _createJ_method1(N, J0, g, τ; parallel=parallel)
            end
            return J
        else
            J = _createJ_method1(N, J0, g, τ; return_deviations=false, parallel=parallel)
            return J
        end
    else
        # Method 2 from Castedo et al., 2025
        A = 1 - τcon - τdiv
        B = τr - 2τc
        # Positive-definiteness conditions
        if 1 - 2 * (τcon + τdiv) ≤ 0 || A ≤ abs(B)
            if verbose
                @warn "Invalid motif strengths combination."
            end
            return nothing
        else
            if quality_control
                deviations = [1.0, 1.0, 1.0, 1.0]
                nIters = 0
                while any(x -> abs(x) > 3 / N, deviations) && nIters <= max_iter
                    nIters += 1
                    J, deviations = _createJ_method2(N, J0, g, τ; parallel=parallel)
                end
                if verbose && nIters > max_iter
                    @warn "Failed to generate a quality matrix within $max_iter iterations. Max deviations: $(maximum(abs.(deviations)))."
                end
                return J
            else
                J = _createJ_method2(N, J0, g, τ; return_deviations=false, parallel=parallel)
                return J
            end
        end
    end
end

function parameter_token(x::Real)
    s = string(round(Float64(x), digits=2))
    return replace(s, "." => "p")
end

## =================== Theoretical eigenspectrum for J =======================

function TheoryR(g::Float64, τ::Tuple{Vararg{Float64}})
    τc, τr, τcon, τdiv = τ_parser(τ)
    if abs(τc) > sqrt(τcon * τdiv) || τcon < 0 || τdiv < 0
        throw(ArgumentError("Invalid motif correlation combinations."))
    end
    return g * (1.0 .- τdiv .- τcon .+ τr .- 2τc) ./ sqrt.(1.0 .- τdiv .- τcon)
end

function TheoryOutlier(N::Integer, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}})
    τc, τr, τcon, τdiv = τ_parser(τ)
    if abs(τc) > sqrt(τcon * τdiv) || τcon < 0 || τdiv < 0
        throw(ArgumentError("Invalid motif correlation combinations."))
    end
    λ1 = 1 / 2 * (J0 * N .- sqrt.(Complex(J0^2 * N^2 .+ 4g^2 * ((N - 1) .* τc + τr))))
    λ2 = 1 / 2 * (J0 * N .+ sqrt.(Complex(J0^2 * N^2 .+ 4g^2 * ((N - 1) .* τc + τr))))
    return λ1, λ2
end


## ============= Positive activation functions ===============
ϕ_positive(x) = 1/2 * (tanh(x) + 1)
ϕ_positive_p(x) = 1/2 * (sech(x))^2

function GaussianPositiveActivation!(du, u, J, t)
    ϕ = ϕ_positive.(u)
    mul!(du, J, ϕ)
    du .-= u
end

## ================ Phase diagram ==========================

function _converged_to_fp(X; window::Int64=200, tol::Float64=1e-4)
    # A helper function for PhaseClaissifier to determine convergence
    # X is N × T array of trajectory
    T = size(X, 2)

    T ≤ window + 5 && throw(ArgumentError("tspan is too short."))

    Xwin = view(X, :, T-window:T)

    # Temporal std of neuron means
    mean_traj = mean(Xwin, dims=1) |> vec
    temp_std = std(mean_traj)

    # Spatial drift between earliest and latest window snapshots
    @views drift = norm(Xwin[:, end] - Xwin[:, 1]) / size(Xwin, 1)
    return (temp_std < tol) && (drift < tol)
end

"""
    This is a phase classifier based on simulated trajectories and the eigenspectrum.
    This function only considers real outliers.
"""
function PhaseClassifier(τ::Tuple{Vararg{Float64}}, g::Float64, J0::Float64, N::Int64;
    n_samples::Int64=48, saveat::Float64 = 1.0,
    tspan::Tuple{Float64,Float64}=(0.0, 500.0), burn_in::Float64=50.0)

    τc, τr, τcon, τdiv = τ_parser(τ)

    if  abs(τr) > 0.1
        throw(ArgumentError("This classifier only works for τrec ≈ 0."))
    end

    R = TheoryR(g, τ)
    _, λ2 = TheoryOutlier(N, J0, g, τ)
    if imag(λ2) != 0
        return missing
    elseif R <= 1.0 && real(λ2) <= 1.0
        return 0 # Trivial fixed point
    elseif R > 1.0 && real(λ2) <= R
        return 1 # Homogeneous chaos
    elseif R <= 1.0 && real(λ2) > 1.0
        return 2 # Nontrivial fixed point
    end

    J_array = [CreateJ(N, J0, g, τ; parallel=true, verbose=false) for _ in 1:n_samples]

    if J_array[1] === nothing
        return -1 # invalid motif combination
    end

    function prob_func(prob, i, repeat)
        remake(prob; u0 = 0.5 .* randn(N), p = J_array[i])
    end


    prob = ODEProblem(GaussianTanh!, randn(N), tspan, zeros(N, N))

    if BLAS.get_num_threads() > 1
        BLAS.set_num_threads(1)
    end    

    ensemble_problem = EnsembleProblem(prob, prob_func=prob_func)
    sim = solve(ensemble_problem, Tsit5(); trajectories=n_samples,
        saveat=saveat)
    times = sim[1].t

    is_fp = Vector{Bool}(undef, n_samples)
    burn_idx = findfirst(t -> t >= burn_in, times)
    Threads.@threads for j in 1:n_samples
        X = sim[j][:, burn_idx:end]
        is_fp[j] = _converged_to_fp(X; window = 100)
    end

    if sum(is_fp) > n_samples * 0.5
        return 2 # Nonzero fixed point
    else
        len_traj = length(times) - findfirst(t -> t >= burn_in, times) + 1
        X_tensor = zeros(N, len_traj, n_samples)
        Threads.@threads for i in 1:n_samples
            X_tensor[:, :, i] .= sim[i][:, times .>= burn_in]
        end
        mean_x = dropdims(mean(X_tensor; dims=1), dims=1) |> vec
        lock(🔒)
        result = try
            rcopy(R"suppressWarnings(dip.test($mean_x))")
        finally
            unlock(🔒)
        end
        if result[:p_value] > 0.05
            return 1 # Homogeneous chaos
        else
            return 3 # Structured chaos
        end
    end
end


function NumericalAutocorrelation(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    T::Float64=300.0, burn::Float64=50.0, dt::Float64=0.5, n_samples::Int64=50, ϕ::Function=tanh,
    exclude_bimodal::Bool=false, subtract_mean::Bool=false)

    _, τrec, τcon, τdiv = τ_parser(τ)
    if 1 - 2τcon - 2τdiv - abs(τrec) < 0
        return nothing, nothing, nothing, nothing 
    end

    function ODEFunc!(du, u, J, t)
        du .= -u .+ J * ϕ.(u)
    end

    J_array = Vector{Matrix{Float64}}()
    sizehint!(J_array, n_samples)

    while length(J_array) < n_samples
        J = CreateJ(N, J0, g, τ; parallel=true)
        if exclude_bimodal
            p_value = bimodal_test(J; parallel=true)
            p_value > 0.05 || continue   # skip bimodal networks
        end
        push!(J_array, J)
    end

    function prob_func(prob, i, repeat)
        # remake avoids mutating shared state across threads
        remake(prob; u0=randn(N), p=J_array[i])
    end

    tspan = (0.0, T)
    prob = ODEProblem(ODEFunc!, zeros(N), tspan, zeros(N, N))
    sim = solve(EnsembleProblem(prob; prob_func=prob_func), Tsit5();
                   trajectories=n_samples, saveat=dt)

    times = sim[1].t
    burn_idx = findfirst(>=(burn), times)
    M = length(times) - burn_idx + 1

    n_chunks = min(n_samples, Threads.nthreads())
    chunk_size = cld(n_samples, n_chunks)
    sample_chunks = [first_idx:min(first_idx + chunk_size - 1, n_samples) for first_idx in 1:chunk_size:n_samples]
    X_buf = [zeros(Float64, N, M) for _ in eachindex(sample_chunks)]
    ϕX_buf = [zeros(Float64, N, M) for _ in eachindex(sample_chunks)]

    mx_rec = zeros(Float64, n_samples, M)
    Cx_rec = zeros(Float64, n_samples, M)
    Cϕ_rec = zeros(Float64, n_samples, M)

    Threads.@threads :static for chunk_idx in eachindex(sample_chunks)
        X = X_buf[chunk_idx]
        ϕX = ϕX_buf[chunk_idx]

        for j in sample_chunks[chunk_idx]
            X .= @view Array(sim[j])[:, burn_idx:end]
            subtract_mean && (X .-= mean(X; dims=2))
            ϕX .= ϕ.(X)

            mx_rec[j, :] .= mean(X; dims=1) |> vec

            @inbounds for i in 1:N
                Cx_rec[j, :] .+= autocov_fft(view(X,  i, :)) ./ N
                Cϕ_rec[j, :] .+= autocov_fft(view(ϕX, i, :)) ./ N
            end
        end
    end

    mx_mean = vec(median(mx_rec; dims=1))
    Cx_mean = vec(median(Cx_rec; dims=1))
    Cx_std  = vec(std(Cx_rec;  dims=1))
    Cϕ_mean = vec(median(Cϕ_rec; dims=1))
    Cϕ_std  = vec(std(Cϕ_rec;  dims=1))

    return times[burn_idx:end] .- burn, mx_mean, Cx_mean, Cx_std, Cϕ_mean, Cϕ_std
end

## ================= Ergodicity test ===================
function ComputeMeanDistance(
    N::Int64, J0::Float64, g::Float64,
    τ::Tuple{Vararg{Float64}};
    Tvals::StepRangeLen=1.0:1200.0,
    burn_in::Float64=100.0,
    nNets::Int64=10,
    nInits::Int64=48,
    μ0::Float64=0.0,
    σ0::Float64=1.0,
    reltol::Float64=1e-3,
    abstol::Float64=1e-6,
    verbose::Bool=false,
    log_channel::Union{Nothing,RemoteChannel}=nothing
)
    # Check the initialization
    τchn, τrec, τcon, τdiv = τ_parser(τ)
    Δ = τrec - 2τchn
    if Δ > 0.3 && μ0 != 0.0
        @warn "Initializing with a bias for motif combination τchn = $τchn, τrec = $τrec, 
        τcon = $τcon, τdiv = $τdiv is not recommended."
        flush(stdout)
    elseif τchn > 1e-5 && (abs(μ0) ≤ 1.0 || 2σ0 ≥ abs(μ0))
        @warn "Initializing with a larger bias for motif combination τchn = $τchn,
        τrec = $τrec, τcon = $τcon, τdiv = $τdiv is recommended."
        flush(stdout)
    end

    # Store D(T) for all valid networks
    dt = Float64(step(Tvals))
    D_all = zeros(length(Tvals), nNets)

    function prob_func(prob, i, repeat)
        remake(prob, u0=σ0 .* randn(N) .+ μ0)
    end

    tspan = (0.0, burn_in + Tvals[end])

    valid_nets = 0

    function log_progress!(valid_nets::Int64)
        if verbose && log_channel !== nothing
            percent_done = floor(Int, 100 * valid_nets / nNets)
            put!(
                log_channel,
                "Worker $(myid()) finished valid network $valid_nets/$nNets " *
                "($(percent_done)% complete)"
            )
        end
        return nothing
    end
    
    # Keep going until we have exactly nNets valid networks
    while valid_nets < nNets
        J = CreateJ(N, J0, g, τ; parallel=true, verbose=false)

        if J === nothing
            return nothing, nothing, nothing # The motif combination is invalid
        end

        if τcon + τdiv ≥ 0.02 && Δ ≤ 0.3
            p_val = bimodal_test(J; nInits=20, burn_in=burn_in, T=500.0,
                 parallel=true)
            if p_val < 0.05
                continue # Network is bimodal, try generating a new one
            end
        end

        prob = ODEProblem(GaussianTanh!, randn(N), tspan, J)
        ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

        sim = solve(ensemble_prob, Vern7(), EnsembleThreads();
            trajectories=nInits, saveat=dt, reltol=reltol, abstol=abstol, maxiters=Inf)

        times = sim[1].t
        burn_idx = findfirst(t -> t >= burn_in, times)
        burn_idx === nothing && error("No saved time reaches burn_in=$burn_in.")
        elapsed_times = times[burn_idx:end] .- burn_in
        len_traj = length(elapsed_times)
        counts = reshape(1:len_traj, 1, len_traj, 1)
        step_indices = [searchsortedlast(elapsed_times, T) for T in Tvals]
        any(iszero, step_indices) && error(
            "Tvals contains values before the first saved post-burn time."
        )

        X_tensor = zeros(N, len_traj, nInits)
        for a in 1:nInits
            X_tensor[:, :, a] .= sim[a][:, burn_idx:end]
        end

        X_cumsum = cumsum(X_tensor, dims=2)
        M_tensor = X_cumsum ./ counts

        D_net = zeros(length(Tvals))

        Threads.@threads for Ti in eachindex(step_indices)
            M = @view M_tensor[:, step_indices[Ti], :]
            Gram = M' * M
            norms = diag(Gram)
            Dmat = norms .+ norms' .- 2Gram
            n = size(Dmat, 1)
            D_net[Ti] = sum(triu(Dmat, 1)) / (n * (n - 1) / 2) / N
        end
        
        valid_nets += 1
        D_all[:, valid_nets] .= D_net
        log_progress!(valid_nets)
    end

    D_mean = mean(D_all; dims=2) |> vec
    D_sem = std(D_all; dims=2) ./ sqrt(nNets) |> vec
    
    return Tvals, D_mean, D_sem
end


function ComputeDInf(Tvals::StepRangeLen, D_mean::Vector{Float64})
    model(T, p) = p[1] .* exp.(-T ./ p[2]) .+
               p[3] .* exp.(-T ./ p[4]) .+ p[5]
    p0 = [1.0, 200.0, 1.0, 1000.0, 0.1]
    p_lower = [-Inf, 1e-5, -Inf, 1e-5, 0.0]
    p_upper = [100, 1.0e5, 100.0, 1.0e5, 20.0]
    fit = curve_fit(model, Tvals, D_mean, p0; lower=p_lower, upper = p_upper)
    if (fit.param[5] == 20.0 || isapprox(fit.param[2], p_lower[2]) 
        || isapprox(fit.param[4], p_lower[4]) || isapprox(fit.param[2], p_upper[2])
         || isapprox(fit.param[4], p_upper[4]))
        return(mean(D_mean[end-1000:end]))
    else
        return fit.param[5]
    end
end
## ===================== Lyapunov Spectrum Start ============================

struct LSPR_params
    nLE::Int64
    tONS::Float64
    burn_in::Float64
    T::Float64 # The time interval for measuring the Lyapunov spectrum
    return_traj::Bool
    exclude_bimodal::Bool
end

function tanhLSPR(J::Matrix{Float64}, params::LSPR_params)

    N = size(J, 1)
    if params.exclude_bimodal
        # Check if there is a bimodal structure
        p = bimodal_test(J)
        # Exclude networks with bimodal μ_x
        if p < 0.05
            return (LS=NaN, PR=NaN)
        end
    end

    # Time interval for re-orthonormalization
    tau_ons = params.tONS

    function lspr_dynamics!(du, u, p, t)
        J_mat = p

        x = @view u[1:N]
        Q = @view u[N+1:end]

        dx = @view du[1:N]
        dQ = @view du[N+1:end]

        Q_mat = reshape(Q, (N, params.nLE))
        dQ_mat = reshape(dQ, (N, params.nLE))

        tanh_x = tanh.(x)

        # Dynamics for x: dx = -x + J * tanh(x)
        mul!(dx, J_mat, tanh_x)
        @. dx = dx - x

        # Dynamics for Q: dQ = -Q + J * (sech^2(x) .* Q)
        # sech^2(x) = 1 - tanh^2(x)
        sech2 = 1.0 .- tanh_x .^ 2
        temp_Q = sech2 .* Q_mat

        mul!(dQ_mat, J_mat, temp_Q)
        @. dQ_mat = dQ_mat - Q_mat
    end

    x0 = randn(N)
    Q0_mat = Matrix(qr(randn(N, params.nLE)).Q)
    u0 = vcat(x0, vec(Q0_mat))

    tspan_burn = (0.0, params.burn_in)
    prob_burn = ODEProblem(lspr_dynamics!, u0, tspan_burn, J)

    sol_burn = solve(prob_burn, Tsit5();
        save_everystep=false, save_start=false, save_end=true)

    u_measure_start = copy(sol_burn.u[end])

    # After burn-in, also orthonormalize Q a few times to align with Lyapunov directions
    function equilibrate_Q!(integrator)
        u_curr = integrator.u
        Q_vec = @view u_curr[N+1:end]
        Q_curr = reshape(Q_vec, (N, params.nLE))
        Q_fact = qr(Q_curr)
        Q_new = Matrix(Q_fact.Q)
        Q_vec .= vec(Q_new)
    end

    cb_equilibrate = PeriodicCallback(equilibrate_Q!, tau_ons; save_positions=(false, false))
    tspan_equilibrate = (0.0, 10 * tau_ons)  # 10 orthonormalizations
    prob_equilibrate = ODEProblem(lspr_dynamics!, u_measure_start, tspan_equilibrate, J)
    sol_equilibrate = solve(prob_equilibrate, Tsit5();
        callback=cb_equilibrate, save_everystep=false,
        save_start=false, save_end=true)

    u_measure_start = copy(sol_equilibrate.u[end])

    if params.return_traj
        tTraj = Float64[]
        L1Traj = Float64[]
        PRTraj = Float64[]
    end

    LS_accum = zeros(params.nLE)
    PR_accum = Ref(0.0)
    counter = Ref(0)

    function orthonormalize_Q!(integrator)
        u_curr = integrator.u
        Q_vec = @view u_curr[N+1:end]
        Q_curr = reshape(Q_vec, (N, params.nLE))

        Q_fact = qr(Q_curr)

        # Update Q in the integrator state to the orthonormalized Q
        Q_new = Matrix(Q_fact.Q)
        Q_vec .= vec(Q_new)

        R_diag = abs.(diag(Q_fact.R))
        local_LEs = log.(R_diag) ./ tau_ons
        LS_accum .+= local_LEs

        pr_val = 1.0 / sum(x -> x^4, @view Q_new[:, 1])
        PR_accum[] += pr_val

        counter[] += 1

        if params.return_traj
            push!(tTraj, integrator.t)
            push!(L1Traj, local_LEs[1])
            push!(PRTraj, pr_val)
        end
    end

    cb = PeriodicCallback(orthonormalize_Q!, tau_ons; save_positions=(false, false))

    tspan_measure = (0.0, params.T)
    prob_measure = ODEProblem(lspr_dynamics!, u_measure_start, tspan_measure, J)
    solve(prob_measure, Tsit5(), callback=cb, save_everystep=false, save_start=false, save_end=false)

    n_measurements = counter[]
    if n_measurements == 0
        @warn "No measurements were taken. T might be too close to burn_in."
        return (LS=NaN, PR=NaN)
    end

    LS_avg = LS_accum ./ n_measurements
    PR_avg = PR_accum[] / n_measurements

    if params.return_traj
        return (t=tTraj, LS=LS_avg, PR=PR_avg, L1Traj=L1Traj, PRTraj=PRTraj)
    else
        return (LS=LS_avg, PR=PR_avg)
    end
end

function KSEntropyKYDimension(LS::AbstractVecOrMat{Float64})
    if LS isa Vector
        LS = reshape(LS, :, 1)
    end

    rows, cols = size(LS)
    rows > 0 || throw(ArgumentError("Input must not be empty."))

    H_values = Vector{Float64}(undef, cols)
    KYD_values = Vector{Float64}(undef, cols)

    for (i, spectrum) in enumerate(eachcol(LS))
        LS = sort(spectrum, rev=true)

        # KS Entropy
        H_values[i] = sum(LS[LS.>0])

        # KY Dimension
        cs = cumsum(LS)
        j = findlast(cs .>= 0)

        if j === nothing
            KYD_values[i] = 0.0
        elseif j == rows
            KYD_values[i] = Float64(rows)
            if cols == 1
                @warn "All Lyapunov exponents are positive. KYD set to embedding dimension."
            end
        else
            KYD_values[i] = j - cs[j] / LS[j+1]
        end
    end

    return mean(H_values), std(H_values), mean(KYD_values), std(KYD_values)
end

function ComputeLSPR(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    burn_in::Float64=100.0, T::Float64=200.0, nLE::Int64=1,
    n_samples::Int64=10, tONS::Float64=0.1, seed::Union{Int64,Nothing}=nothing, verbose=true,
    return_traj::Bool=false, chaos_only=false, return_KSEandKYD::Bool=false, 
    exclude_bimodal::Bool=false)

    nLE ≤ N || throw(ArgumentError("nLE has to be smaller or equal to N"))

    if !isnothing(seed)
        Random.seed!(seed)
    end

    J_array = [CreateJ(N, J0, g, τ; parallel=true, verbose=false) for _ in 1:n_samples]

    if isnothing(J_array[1])
        if verbose
            @warn "Invalid motif combination."
        end
        return nothing
    end

    params = LSPR_params(nLE, tONS, burn_in, T, return_traj, exclude_bimodal)

    LS_all = Matrix{Float64}(undef, nLE, n_samples)
    PR_all = Vector{Float64}(undef, n_samples)

    for i in 1:n_samples
        res = tanhLSPR(J_array[i], params)
        LS_all[:, i] .= res.LS
        PR_all[i] = res.PR
        if return_traj
            if i == 1
                # Initialize based on first result dimensions
                len = length(res.t)
                t_ref = res.t
                L1Traj_all = Matrix{Float64}(undef, len, n_samples)
                PRTraj_all = Matrix{Float64}(undef, len, n_samples)
            end
            L1Traj_all[:, i] .= res.L1Traj
            PRTraj_all[:, i] .= res.PRTraj
        end
    end

    chaotic_idx = @views LS_all[1, :] .> 0 .&& .!isnan.(LS_all[1, :])
    if chaos_only
        if sum(chaotic_idx) > 0
            @views begin
                LS_mean = dropdims(mean(LS_all[:, chaotic_idx], dims=2), dims=2)
                LS_std = dropdims(std(LS_all[:, chaotic_idx], dims=2), dims=2)
                PR_mean = mean(PR_all[chaotic_idx])
                PR_std = std(PR_all[chaotic_idx])
                if return_KSEandKYD
                    H_mean, H_std, KYD_mean, KYD_std = KSEntropyKYDimension(LS_all[:, chaotic_idx])
                end
            end
        else
            LS_mean = NaN
            LS_std = NaN
            PR_mean = NaN
            PR_std = NaN
            if return_KSEandKYD
                H_mean = NaN
                H_std = NaN
                KYD_mean = NaN
                KYD_std = NaN
            end
            @warn "All networks are nonchaotic or shows bimodal structure for τ = $τ. Consider increasing `n_samples` or setting
            `chaos_only` to false."
        end
    else
        valid_idx = .!isnan.(LS_all[1, :])
        if sum(valid_idx) == 0
            @warn "All networks returned NaN Lyapunov exponents."
            LS_mean = NaN
            LS_std = NaN
            PR_mean = NaN
            PR_std = NaN
            if return_KSEandKYD
                H_mean = NaN
                H_std = NaN
                KYD_mean = NaN
                KYD_std = NaN
            end
        else
            LS_mean = dropdims(mean(LS_all[:, valid_idx], dims=2), dims=2)
            LS_std = dropdims(std(LS_all[:, valid_idx], dims=2), dims=2)
            PR_mean = mean(PR_all[valid_idx])
            PR_std = std(PR_all[valid_idx])
            if return_KSEandKYD
                H_mean, H_std, KYD_mean, KYD_std = KSEntropyKYDimension(LS_all[:, valid_idx])
            end
        end
    end

    if return_traj && return_KSEandKYD
        return (
            LS_mean=LS_mean, LS_std=LS_std,
            PR_mean=PR_mean, PR_std=PR_std,
            t=t_ref, L1Traj=L1Traj_all, PRTraj=PRTraj_all,
            H_mean=H_mean, H_std=H_std,
            KYD_mean=KYD_mean, KYD_std=KYD_std
        )
    elseif return_KSEandKYD
        return (
            LS_mean=LS_mean, LS_std=LS_std,
            PR_mean=PR_mean, PR_std=PR_std,
            H_mean=H_mean, H_std=H_std,
            KYD_mean=KYD_mean, KYD_std=KYD_std
        )
    elseif return_traj
        return (
            LS_mean=LS_mean, LS_std=LS_std,
            PR_mean=PR_mean, PR_std=PR_std,
            t=t_ref, L1Traj=L1Traj_all, PRTraj=PRTraj_all
        )
    else
        return (
            LS_mean=LS_mean, LS_std=LS_std,
            PR_mean=PR_mean, PR_std=PR_std
        )
    end
end

## ====================== PR dimension =========================
function NumericalPRD(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}};
    burn_in::Float64 = 50.0,
    n_samples::Int64 = 128,
    T::Union{Float64, Nothing} = nothing,
    save_step::Float64 = 1.0,
    exclude_bimodal::Bool = false)

    T_val::Float64 = isnothing(T) ? 50.0 * N : T

    Dϕ_samples = zeros(Float64, n_samples)
    C4_samples  = zeros(Float64, n_samples)
    C2_samples  = zeros(Float64, n_samples)   # variance of a single neuron

    Threads.@threads for k in 1:n_samples
        valid_sample = false
        
        while !valid_sample
            J = CreateJ(N, J0, g, τ)
            
            if exclude_bimodal
                p = bimodal_test(J; parallel = false)
                if p < 0.05
                    continue
                end
            end
            
            C_sum = zeros(N, N)
            n_samp = Ref(0)
            ϕ_buf = zeros(N)

            cb = FunctionCallingCallback(
                (u, t, integrator) -> begin
                    @. ϕ_buf = tanh(u) 
                    BLAS.ger!(1.0, ϕ_buf, ϕ_buf, C_sum)
                    n_samp[] += 1
                end;
                funcat = burn_in:save_step:T_val,
                func_everystep = false
            )

            u0 = 0.5 .* randn(N)
            prob = ODEProblem(GaussianTanh!, u0, (0.0, T_val), J)
            
            solve(prob, Tsit5(), callback = cb,
                  save_everystep = false,
                  saveat = Float64[])

            C = C_sum ./ n_samp[]
            
            tr_C       = tr(C)
            frob2      = sum(abs2, C)
            diag_frob2 = sum(abs2, diag(C))
            
            Dϕ_samples[k] = tr_C^2 / (N * frob2)
            C4_samples[k]  = (frob2 - diag_frob2) / (N - 1)
            C2_samples[k]  = tr_C / N

            valid_sample = true
        end
    end

    return mean(Dϕ_samples), std(Dϕ_samples),
           mean(C2_samples),  std(C2_samples),
           mean(C4_samples),  std(C4_samples)

end

"""
Compute the four-point function 𝒞ϕ(ω₁, ω₂) in the frequency domain.
"""
function FourPointC(N::Int64, J0::Float64, g::Float64, τ::Tuple{Vararg{Float64}},
    Cϕh::Vector{ComplexF64}, χϕh::Vector{ComplexF64}, dt::Float64)
    @assert length(Cϕh) == length(χϕh) "Cϕ and χϕ must have the same length."

    Cϕh = real.(Cϕh)
    τchn, τrec, τcon, τdiv = τ_parser(τ)

    χ1 = χϕh # χϕ(ω₁),    (T,)
    χ2 = transpose(χϕh)  # χϕ(ω₂),    (1,T)
    χ1c = conj.(χϕh) # χϕ*(ω₁),   (T,)
    χ2c = χϕh'     # χϕ*(ω₂), (1,T)
    C1 = Cϕh # Cϕ(ω₁),    (T,)
    C2 = transpose(Cϕh)  # Cϕ(ω₂),    (1,T)
    absχ1sq = abs2.(χϕh) # |χϕ(ω₁)|², (T,)
    absχ2sq = transpose(abs2.(χϕh))  # |χ^φ(ω₂)|², (1,T)
    g2 = g^2
    g4 = g^4

    # Γ_{F00'F00'}(ω₁,ω₂)
    χ1χ2 = χ1 * χ2 # χ(ω₁)χ(ω₂), (T,T)
    ΓFF = (g2 / (2π)) .*
          ((1 .+ 2π * g2 * (1 + N * (τchn^2 + τcon*τdiv)) .* χ1χ2) ./
           (1 .- 4π^2 * g4 .* χ1χ2 .^ 2))

    # Γ_{F*00'F0'0}(ω₁,ω₂) 
    χ1cχ2 = χ1c .* χ2 # χ*(ω₁)χ(ω₂), (T,T)
    ΓF̄F = (g2 / (2π)) .*
           (τrec .+ 2π * g2 * (τrec^2 + 2N * τchn^2) .* χ1cχ2) ./
           (1 .- 4π^2 * τrec^2 * g4 .* χ1cχ2 .^ 2)

    # Γ_{F*00'C00'}(ω₁,ω₂)
    inner = (τrec + N * τchn * (τcon + τdiv)) .+
            τrec * (2π)^2 .* χ1c .*
            (χ2c .* conj.(ΓFF) .+ χ2 .* ΓF̄F)

    ΓF̄C = g4 .* χ1c .* C2 .* inner ./
           (1 .- τrec * g4 * (2π)^2 .* χ1c .^ 2 .* absχ2sq)

    # Γ_{C00'F*00'}(ω₁,ω₂)
    ΓCF̄ = transpose(ΓF̄C)

    H = C1 * C2 .* (χ1 * χ2 .* ΓFF .+ χ1c * χ2 .* ΓF̄F) .+
        absχ1sq * χ2c .* C2 .* ΓCF̄ .+ χ1c * absχ2sq .* C1 .* ΓF̄C

    denom = 1 .- 4π^2 * g4 .* absχ1sq * absχ2sq
    Ψ = (4π^2 ./ denom) .*
        (g4 * (1 + N * (τcon^2 + τdiv^2)) .* C1 * C2 .* absχ1sq .* absχ2sq .+
         H .+ conj.(H))

    return Ψ
end

"""
Compute the scalar sum of the four-point function 𝒞ϕ(ω₁, ω₂) directly.
O(1) memory, O(T^2) time.
"""
function SumFourPointC(N::Int64, g::Float64, τ::Tuple{Vararg{Float64}},
    Cϕh::Vector{ComplexF64}, χϕh::Vector{ComplexF64})
    
    @assert length(Cϕh) == length(χϕh) "Cϕh and χϕh must have the same length."

    Cϕh = real.(Cϕh)
    τchn, τrec, τcon, τdiv = τ_parser(τ)

    T = length(Cϕh)
    g2 = g^2
    g4 = g^4

    c_FF1 = 2π * g2 * (1 + N * (τchn^2 + τcon*τdiv))
    c_FF2 = 4π^2 * g4
    c_FbF1 = 2π * g2 * (τrec^2 + 2N * τchn^2)
    c_FbF2 = 4π^2 * τrec^2 * g4
    c_in1 = τrec + N * τchn * (τcon + τdiv)
    c_in2 = 4π^2 * τrec
    c_denom_FC = τrec * g4 * 4π^2
    c_denom = 4π^2 * g4
    c_Psi1 = g4 * (1 + N * (τcon^2 + τdiv^2))
    pref_FF = g2 / 2π

    abs2_χ = abs2.(χϕh)
    c_χ = conj.(χϕh)

    n_chunks = Threads.nthreads() * 4
    chunk_size = cld(T, n_chunks)
    chunks = Iterators.partition(1:T, chunk_size)

    tasks = map(chunks) do chunk
        Threads.@spawn begin
            local_sum = 0.0
            @inbounds for i in chunk
                chi1 = χϕh[i]
                chi1c = c_χ[i]
                abs1sq = abs2_χ[i]
                C1 = Cϕh[i]

                for j in 1:T
                    chi2   = χϕh[j]
                    chi2c  = c_χ[j]
                    abs2sq = abs2_χ[j]
                    C2     = Cϕh[j]

                    # Γ_{F00'F00'}(ω₁,ω₂)
                    chi1chi2 = chi1 * chi2
                    ΓFF = pref_FF * (1 + c_FF1 * chi1chi2) / (1 - c_FF2 * chi1chi2^2)

                    # Γ_{F*00'F0'0}(ω₁,ω₂) and its swapped version for ΓCF
                    chi1c_chi2 = chi1c * chi2
                    ΓFbF = pref_FF * (τrec + c_FbF1 * chi1c_chi2) / (1 - c_FbF2 * chi1c_chi2^2)
                    
                    chi2c_chi1 = chi2c * chi1
                    ΓFbF_ji = pref_FF * (τrec + c_FbF1 * chi2c_chi1) / (1 - c_FbF2 * chi2c_chi1^2)

                    # Inner block and transposed inner block
                    inner_ij = c_in1 + c_in2 * chi1c * (chi2c * conj(ΓFF) + chi2 * ΓFbF)
                    inner_ji = c_in1 + c_in2 * chi2c * (chi1c * conj(ΓFF) + chi1 * ΓFbF_ji)

                    # Γ_{F*00'C00'}(ω₁,ω₂)
                    denom_FC_ij = 1 - c_denom_FC * chi1c^2 * abs2sq
                    ΓFbC = g4 * chi1c * C2 * inner_ij / denom_FC_ij

                    # Γ_{C00'F*00'}(ω₁,ω₂)
                    denom_FC_ji = 1 - c_denom_FC * chi2c^2 * abs1sq
                    ΓCF = g4 * chi2c * C1 * inner_ji / denom_FC_ji

                    # H matrix element
                    H = C1 * C2 * (chi1chi2 * ΓFF + chi1c_chi2 * ΓFbF) + 
                        abs1sq * chi2c * C2 * ΓCF + 
                        chi1c * abs2sq * C1 * ΓFbC

                    denom = 1 - c_denom * abs1sq * abs2sq
                    term1 = c_Psi1 * C1 * C2 * abs1sq * abs2sq
                    
                    # Ψ_ij = (4π^2 / denom) * (term1 + H + conj(H))
                    val = (4π^2 / denom) * (real(term1) + 2 * real(H))
                    
                    local_sum += val
                end
            end
            return local_sum
        end
    end
    return sum(fetch.(tasks))
end

function TheoreticalPRϕ(N::Int64, g::Float64, τ::Tuple{Vararg{Float64}},
    Cϕ::Vector, χϕ::Vector, dt::Float64; ft_done::Bool=false)
    # Set ft_done = true if Cϕ and χϕ are already
    # Fourier transformed using fft and scaled by dt/sqrt(2π).

    T = length(Cϕ)

    if !ft_done
        C2 = Cϕ[1]
        Cϕ = dt/sqrt(2π) .* fft(Cϕ)
        χϕ = dt/sqrt(2π) .* fft(χϕ)
    else
        C2 = sqrt(2π) / (T * dt) * sum(Cϕ)
    end
    
    sum_Ψ = SumFourPointC(N, g, τ, Cϕ, χϕ)
    C4 = 1 / T^2 * (2π / dt^2) * sum_Ψ

    DPR = C2^2 / (C2^2 + C4)

    return DPR, C2, C4
end

## =================== Fixed point search helpers =====================

struct FixedPointSearchConfig
    seed::Int
    N::Int
    g_eff::Float64
    J0::Float64
    n_inits::Int
    batchsize::Int
    mu0_sigma::Float64
    sigma0::Float64
    residual_tol::Float64
    fp_distance_tol::Float64
    nl_reltol::Float64
    nl_abstol::Float64
    max_nl_iters::Int
    stability_tol::Float64
    distributed_task_batches::Int
end

struct FixedPointSolverCache{M}
    J::M
    activation::Vector{Float64}
    derivative::Vector{Float64}
    residual::Vector{Float64}
end

function FixedPointSolverCache(J::AbstractMatrix{Float64})
    N = size(J, 1)
    return FixedPointSolverCache(J, zeros(N), zeros(N), zeros(N))
end

struct FixedPointCandidate
    x::Vector{Float64}
    residual::Float64
end

struct FixedPointSummary
    x::Vector{Float64}
    residual::Float64
    stable::Bool
    max_real_eig::Float64
end

struct FixedPointBatchSummary
    fixed_points::Vector{FixedPointCandidate}
    residual_sum::Float64
    residual_max::Float64
    n_converged::Int
    n_inits::Int
end

const _WORKER_J_REF = Ref{Matrix{Float64}}()
const _EMPTY_REPRESENTATIVE_INDICES = Int[]

function set_worker_J!(J::AbstractMatrix{Float64})
    _WORKER_J_REF[] = J isa Matrix{Float64} ? J : Matrix{Float64}(J)
    return nothing
end

function _worker_J()
    isassigned(_WORKER_J_REF) || error("Worker $(myid()) has no assigned J matrix.")
    return _WORKER_J_REF[]
end

mutable struct FixedPointRepresentativeIndex
    representatives::Vector{FixedPointCandidate}
    bins::Dict{NTuple{3,Int},Vector{Int}}
    tol::Float64
    radius::Float64
    coordinate_indices::NTuple{3,Int}
end

function FixedPointRepresentativeIndex(config::FixedPointSearchConfig; tol::Float64=config.fp_distance_tol)
    coordinate_indices = (1, max(1, cld(config.N, 2)), config.N)
    radius = tol * sqrt(config.N)
    return FixedPointRepresentativeIndex(
        FixedPointCandidate[],
        Dict{NTuple{3,Int},Vector{Int}}(),
        tol,
        radius,
        coordinate_indices,
    )
end

function residual_rms(x::AbstractVector{Float64}, cache::FixedPointSolverCache)
    ϕ = cache.activation
    r = cache.residual
    @inbounds for i in eachindex(x, ϕ)
        ϕ[i] = tanh(x[i])
    end
    mul!(r, cache.J, ϕ)
    @inbounds for i in eachindex(r, x)
        r[i] -= x[i]
    end
    return norm(r) / sqrt(length(x))
end

function fixed_point_residual!(F, x, cache::FixedPointSolverCache)
    ϕ = cache.activation
    @inbounds for i in eachindex(x, ϕ)
        ϕ[i] = tanh(x[i])
    end
    mul!(F, cache.J, ϕ)
    @inbounds for i in eachindex(F, x)
        F[i] -= x[i]
    end
    return nothing
end

function fixed_point_jacobian!(Jac, x, cache::FixedPointSolverCache)
    J = cache.J
    derivative = cache.derivative
    @inbounds for i in eachindex(x, derivative)
        ϕ = tanh(x[i])
        derivative[i] = 1.0 - ϕ * ϕ
    end
    @inbounds for j in axes(J, 2)
        scale = derivative[j]
        for i in axes(J, 1)
            Jac[i, j] = J[i, j] * scale
        end
    end
    @inbounds for i in axes(Jac, 1)
        Jac[i, i] -= 1.0
    end
    return nothing
end

function normalized_distance(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    sumsq = 0.0
    @inbounds for i in eachindex(x, y)
        δ = x[i] - y[i]
        sumsq += δ * δ
    end
    return sqrt(sumsq / length(x))
end

function _representative_bin(x::AbstractVector{Float64}, index::FixedPointRepresentativeIndex)
    return ntuple(i -> floor(Int, x[index.coordinate_indices[i]] / index.radius), 3)
end

function _index_representative!(index::FixedPointRepresentativeIndex, representative_idx::Int)
    key = _representative_bin(index.representatives[representative_idx].x, index)
    push!(get!(Vector{Int}, index.bins, key), representative_idx)
    return nothing
end

function _nearby_representative_index(
    index::FixedPointRepresentativeIndex,
    result::FixedPointCandidate,
)
    center = _representative_bin(result.x, index)
    for Δ1 in -1:1, Δ2 in -1:1, Δ3 in -1:1
        key = (center[1] + Δ1, center[2] + Δ2, center[3] + Δ3)
        for representative_idx in get(index.bins, key, _EMPTY_REPRESENTATIVE_INDICES)
            representative = index.representatives[representative_idx]
            if normalized_distance(result.x, representative.x) ≤ index.tol
                return representative_idx
            end
        end
    end
    return nothing
end

function merge_distinct_fixed_point!(
    index::FixedPointRepresentativeIndex,
    result::FixedPointCandidate
)
    representative_idx = _nearby_representative_index(index, result)
    if representative_idx === nothing
        push!(index.representatives, result)
        _index_representative!(index, length(index.representatives))
    elseif result.residual < index.representatives[representative_idx].residual
        old_key = _representative_bin(index.representatives[representative_idx].x, index)
        index.representatives[representative_idx] = result
        if _representative_bin(result.x, index) != old_key
            old_bin = index.bins[old_key]
            deleteat!(old_bin, findfirst(==(representative_idx), old_bin))
            _index_representative!(index, representative_idx)
        end
    end
    return index
end

function fixed_point_stability(
    x::AbstractVector{Float64},
    jacobian_cache::FixedPointSolverCache,
    config::FixedPointSearchConfig
)
    J = jacobian_cache.J
    Jac = similar(J)
    fixed_point_jacobian!(Jac, x, jacobian_cache)
    max_real_eig = maximum(real, eigvals(Jac))
    return max_real_eig < -config.stability_tol, max_real_eig
end

function is_nonzero_fixed_point(x::AbstractVector{Float64}; tol::Float64)
    return norm(x) / sqrt(length(x)) > tol
end

function add_symmetric_fixed_points(results, config::FixedPointSearchConfig; tol::Float64=config.fp_distance_tol)
    symmetric_results = FixedPointCandidate[]
    sizehint!(symmetric_results, 2 * length(results))
    for result in results
        push!(symmetric_results, result)
        if is_nonzero_fixed_point(result.x; tol=tol)
            push!(symmetric_results, FixedPointCandidate(-result.x, result.residual))
        end
    end
    return symmetric_results
end

function distinct_fixed_points(results, config::FixedPointSearchConfig; tol::Float64=config.fp_distance_tol)
    index = FixedPointRepresentativeIndex(config; tol=tol)
    for result in results
        merge_distinct_fixed_point!(index, result)
    end
    return index.representatives
end

function merge_distinct_fixed_points!(
    index::FixedPointRepresentativeIndex,
    results
)
    for result in results
        merge_distinct_fixed_point!(index, result)
    end
    return index
end

function _summarize_fixed_point_chunk(fixed_points::Vector{FixedPointCandidate}, config::FixedPointSearchConfig)
    summaries = Vector{FixedPointSummary}(undef, length(fixed_points))
    jacobian_cache = FixedPointSolverCache(_worker_J())
    for i in eachindex(fixed_points)
        fp = fixed_points[i]
        stable, max_real_eig = fixed_point_stability(fp.x, jacobian_cache, config)
        summaries[i] = FixedPointSummary(fp.x, fp.residual, stable, max_real_eig)
    end
    return summaries
end

function _summarize_fixed_points_local(fixed_points, J::AbstractMatrix{Float64}, config::FixedPointSearchConfig)
    summaries = Vector{FixedPointSummary}(undef, length(fixed_points))
    jacobian_caches = [FixedPointSolverCache(J) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for i in eachindex(fixed_points)
        fp = fixed_points[i]
        jacobian_cache = jacobian_caches[Threads.threadid()]
        stable, max_real_eig = fixed_point_stability(fp.x, jacobian_cache, config)
        summaries[i] = FixedPointSummary(fp.x, fp.residual, stable, max_real_eig)
    end
    return summaries
end

function summarize_fixed_points(fixed_points, J::AbstractMatrix{Float64}, config::FixedPointSearchConfig)
    isempty(fixed_points) && return FixedPointSummary[]
    if nprocs() == 1
        return _summarize_fixed_points_local(fixed_points, J, config)
    end

    chunk_size = max(16, cld(length(fixed_points), 4 * nworkers()))
    chunks = [
        fixed_points[first:min(first + chunk_size - 1, length(fixed_points))]
        for first in 1:chunk_size:length(fixed_points)
    ]
    chunk_summaries = pmap(chunk -> _summarize_fixed_point_chunk(chunk, config), chunks; batch_size=1)

    summaries = FixedPointSummary[]
    sizehint!(summaries, length(fixed_points))
    for chunk_summary in chunk_summaries
        append!(summaries, chunk_summary)
    end
    return summaries
end

function independent_thread_rngs(n_threads::Int)
    device = RandomDevice()
    return [MersenneTwister(rand(device, UInt32)) for _ in 1:n_threads]
end

function solve_fixed_point_batch(J::AbstractMatrix{Float64}, n_inits::Int, config::FixedPointSearchConfig)
    nl_func = NonlinearFunction(fixed_point_residual!; jac=fixed_point_jacobian!)
    n_threads = Threads.nthreads()
    rngs = independent_thread_rngs(n_threads)
    fixed_point_results_by_thread = [FixedPointCandidate[] for _ in 1:n_threads]
    residual_sum_by_thread = zeros(Float64, n_threads)
    residual_max_by_thread = fill(NaN, n_threads)
    n_converged_by_thread = zeros(Int, n_threads)

    Threads.@threads :static for thread_idx in 1:n_threads
        rng = rngs[thread_idx]
        cache = FixedPointSolverCache(J)
        u0 = zeros(Float64, config.N)
        fixed_point_results = fixed_point_results_by_thread[thread_idx]
        sizehint!(fixed_point_results, cld(n_inits, n_threads))
        residual_sum = 0.0
        residual_max = NaN
        n_converged = 0

        for _ in thread_idx:n_threads:n_inits
            μ0 = config.mu0_sigma * randn(rng)
            randn!(rng, u0)
            @inbounds for i in eachindex(u0)
                u0[i] = config.sigma0 * u0[i] + μ0
            end
            nl_prob = NonlinearProblem(nl_func, u0, cache)
            sol = solve(nl_prob, LevenbergMarquardt();
                reltol=config.nl_reltol,
                abstol=config.nl_abstol,
                maxiters=config.max_nl_iters)

            residual = residual_rms(sol.u, cache)
            converged = successful_retcode(sol) && isfinite(residual) && residual ≤ config.residual_tol
            if converged
                push!(fixed_point_results, FixedPointCandidate(Vector{Float64}(sol.u), residual))
                residual_sum += residual
                residual_max = n_converged == 0 ? residual : max(residual_max, residual)
                n_converged += 1
            end
        end

        residual_sum_by_thread[thread_idx] = residual_sum
        residual_max_by_thread[thread_idx] = residual_max
        n_converged_by_thread[thread_idx] = n_converged
    end

    n_fixed_points = sum(length, fixed_point_results_by_thread)
    fixed_points = FixedPointCandidate[]
    sizehint!(fixed_points, n_fixed_points)
    for thread_results in fixed_point_results_by_thread
        append!(fixed_points, thread_results)
    end

    residual_sum = sum(residual_sum_by_thread)
    residual_max = NaN
    for thread_max in residual_max_by_thread
        if !isnan(thread_max)
            residual_max = isnan(residual_max) ? thread_max : max(residual_max, thread_max)
        end
    end
    n_converged = sum(n_converged_by_thread)

    fixed_points_with_symmetry = add_symmetric_fixed_points(fixed_points, config)
    return FixedPointBatchSummary(
        distinct_fixed_points(fixed_points_with_symmetry, config),
        residual_sum,
        residual_max,
        n_converged,
        n_inits)
end

function n_inits_for_batch(batch_idx::Int, config::FixedPointSearchConfig)
    completed_inits = (batch_idx - 1) * config.batchsize
    return min(config.batchsize, config.n_inits - completed_inits)
end

function distributed_batch_ranges(batch_indices::AbstractVector{Int}, config::FixedPointSearchConfig)
    batches_per_task = max(1, config.distributed_task_batches)
    return [
        batch_indices[first:min(first + batches_per_task - 1, length(batch_indices))]
        for first in 1:batches_per_task:length(batch_indices)
    ]
end

function solve_fixed_point_batch_range(
    batch_indices,
    result_channel::RemoteChannel,
    config::FixedPointSearchConfig
)
    J = _worker_J()
    for batch_idx in batch_indices
        batch_result = solve_fixed_point_batch(
            J, n_inits_for_batch(batch_idx, config), config)
        put!(result_channel, (batch_idx, batch_result))
    end
    return nothing
end

function restore_representative_index(fixed_points, config::FixedPointSearchConfig)
    representative_index = FixedPointRepresentativeIndex(config)
    merge_distinct_fixed_points!(representative_index, fixed_points)
    return representative_index
end

end
