using LinearAlgebra
using Random
using JLD2
include("Utils.jl")
using .Utils
include("DMFT.jl")
using .DMFT
BLAS.set_num_threads(32)

N = 4000
n_samples = 10
τrec_vec = [-0.7, 0.5]
τchn = 10.0 / N
J0 = -5.0 / N
T_vec = [50.0, 800.0]
g = 3.0
n_theta = 200
seed = 1234
x_grid = -30.0:0.05:30.0


function sample_eigenvalues(N::Int64, J0::Float64, τ::Tuple{Vararg{Float64}}, g::Float64, n_samples::Int64;
    seed::Int64=1234, ϕ::Function=tanh, ϕ_prime::Function=x -> 1 - tanh(x)^2,
    fp_tspan::Tuple{Float64,Float64}=(0.0, 50.0))

    n_samples >= 1 || error("n_samples must be at least 1.")

    τ_tuple, _, _ = DMFT.boundary_AB(τ)
    values_by_sample = Vector{Vector{ComplexF64}}(undef, n_samples)
    fp_by_sample = Vector{Vector{Float64}}(undef, n_samples)

    sample_idx = 1
    attempt_idx = 0
    while sample_idx <= n_samples
        attempt_idx += 1
        Random.seed!(seed + attempt_idx - 1)
        J = CreateJ(N, J0, g, τ_tuple; parallel=true)
        J === nothing && error("CreateJ failed for τ = $(τ_tuple).")
        x_fp, _, converged = find_fixed_point(J; tspan=fp_tspan, tol=5e-6, ϕ=ϕ, verbose=false)
        converged || continue
        x_fp = Float64.(x_fp)
        d = ϕ_prime.(x_fp)
        M = J .* reshape(d, 1, :)
        values_by_sample[sample_idx] = eigvals(M)
        fp_by_sample[sample_idx] = x_fp
        sample_idx += 1
    end
    return (eigenvalues=vcat(values_by_sample...), x_fp=vcat(fp_by_sample...))
end

eigenvalues_by_τrec = Vector{Vector{ComplexF64}}(undef, length(τrec_vec))
x_fp_by_τrec = Vector{Vector{Float64}}(undef, length(τrec_vec))
boundary_x_by_τrec = Vector{Vector{Float64}}(undef, length(τrec_vec))
boundary_y_by_τrec = Vector{Vector{Float64}}(undef, length(τrec_vec))
theory_p_fp_by_τrec = Vector{Vector{Float64}}(undef, length(τrec_vec))

for (i, τrec) in enumerate(τrec_vec)
    T = T_vec[i]
    τ = (τchn, τrec)

    sample_result = sample_eigenvalues(N, J0, τ, g, n_samples;
        seed=seed, fp_tspan=(0.0, T))
    xb, yb = BoundaryCurve(N, J0, τ, g;
        n_theta=n_theta, boundary_method=:generic, x_grid=x_grid)
    _, theory_p = DMFT_FP_Generic(J0, g, N, τ; x_grid=x_grid)

    eigenvalues_by_τrec[i] = sample_result.eigenvalues
    x_fp_by_τrec[i] = sample_result.x_fp
    boundary_x_by_τrec[i] = xb
    boundary_y_by_τrec[i] = yb
    theory_p_fp_by_τrec[i] = theory_p
end

jldsave("./results/FPStatisticsJacobianBoundary.jld2";
    N, J0, τchn, τrec_vec, T_vec, g, n_samples, n_theta, x_grid,
    eigenvalues_by_τrec, x_fp_by_τrec, boundary_x_by_τrec, boundary_y_by_τrec,
    theory_p_fp_by_τrec)
