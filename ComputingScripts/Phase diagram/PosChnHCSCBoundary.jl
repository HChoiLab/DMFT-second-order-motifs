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
J0 = -5.0 / N
num_points = 15
g_vec = LinRange(4.0, 1.05, num_points)
τchn = 2.40 / N
Δτchn = 0.025 / N

T = 120.0
dt = 0.05
dmft_max_iter = 500
dmft_damp = 0.7
dmft_tol = 5e-5
dmft_μ0 = 0.0
Δx_seed = 1.0

phi_prime(x) = sech(x)^2
τchn_boundary = fill(NaN, num_points)

start = now()

for (i, g) in enumerate(g_vec)
    global τchn, Δx_seed
    while N * τchn < 6.3
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

        if isfinite(sol.Δx)
            global Δx_seed = max(sol.Δx, 0.0)
        end

        ϕ_prime_mean = gauss_expectation(phi_prime, sol.mx, max(sol.Δx, 0.0))
        residual = 1.0 - ϕ_prime_mean * (N * J0 + g^2 * (N * τchn) * ϕ_prime_mean)

        if residual < 0.0
            τchn_boundary[i] = τchn
            break
        end

        global τchn += Δτchn
    end

    elapsed = now() - start
    println("Finished g = $(round(g, digits=6)) ($(i)/$(length(g_vec))); " *
        "N*τchn = $(round(N * τchn_boundary[i], digits=4)); " *
        "elapsed = $(round(Dates.value(elapsed) / 1e3, digits=1)) seconds")
    flush(stdout)
end

file_name = "HCSCBoundaryN$(N)J0$(replace(string(round(N * J0, digits=1)), "." => "p"))B0.jld2"
output_file = joinpath(ROOT_DIR, "results", file_name)
jldsave(output_file;
    N, J0, num_points, g_vec, T,
    dt, dmft_max_iter, dmft_damp, dmft_tol, dmft_μ0, τchn_boundary)


# ## For nonnegligible B
# using CUDA, .DMFT_CUDA, Trapz, CairoMakie, LaTeXStrings
# CUDA.allowscalar(false)
# BLAS.set_num_threads(1)
# const IMAGE_DIR = joinpath(ROOT_DIR, "results", "images")

# const N = 2000
# const J0 = -5.0 / N
# const num_points = 10
# const g_vec = LinRange(4.0, 1.55, num_points)
# const τchn_init = 2.25 / N
# const τrec = -0.3
# const Δτchn = 0.025 / N

# const T = 100.0
# const dt = 0.032
# const burn = 40.0
# const discard_tail = 30.0

# nIter_list = (30, 20, 30, 30)
# nTraj_list = (8192, 12800, 16000, 32000)
# const damp_R = (0.1, 0.25, 0.6, 0.9)
# const damp_C = (0.1, 0.35, 0.6, 0.9)


# function save_Cϕ_χϕ_plot(Cϕ, χϕ, g, τchn, N, dt)

#     t_Cϕ = collect(range(0.0; step=dt, length=length(Cϕ)))
#     t_χϕ = collect(range(0.0; step=dt, length=length(χϕ)))
#     fig = Figure(size=(600, 300))

#     ax_Cϕ = Axis(fig[1, 1],
#         title="g = $(round(g, digits=4)), Nτchn = $(round(N*τchn, digits=4))",
#         xlabel=L"t",
#         ylabel=L"C^\phi")
#     lines!(ax_Cϕ, t_Cϕ, Cϕ, color=:dodgerblue3)
#     ylims!(ax_Cϕ, -0.01, maximum(Cϕ))

#     ax_χϕ = Axis(fig[1, 2],
#         xlabel=L"t",
#         ylabel=L"χ^ϕ")
#     lines!(ax_χϕ, t_χϕ, χϕ, color=:firebrick3)

#     file_name = "CandResponse_g$(parameter_token(g))_Ntauchn$(parameter_token(N*τchn)).png"
#     save(joinpath(IMAGE_DIR, file_name), fig)

#     return nothing
# end

# function find_hcsc_boundary_b_nonzero(g_vec, N, J0, τchn_init, τrec, Δτchn,
#     T, dt, burn, nIter_list, nTraj_list, damp_R, damp_C)

#     τchn_boundary = fill(NaN, length(g_vec))
#     τchn = τchn_init

#     start = now()

#     for (i, g) in enumerate(g_vec)
#         residual_old = NaN
#         while N * τchn < 3.6
#             τ = (τchn, τrec)

#             model = DMFT_CUDA.CreateDMFTRateModel(N, J0, g, τ, T,
#                 nIter_list, nTraj_list, damp_R, damp_C;
#                 dt=dt, tile_size=16000, B2=128)
#             Cϕ_full, χϕ_full = DMFT_CUDA.DMFTMainloop(model; verbose=false,
#                 return_history=false, minimal_return=true)
#             _, Cϕ = extract_stationary_C(Cϕ_full; burn=burn, dt=dt, discard_tail=discard_tail)
#             times, χϕ = extract_stationary_C(χϕ_full; burn=burn, dt=dt, discard_tail=discard_tail)
#             save_Cϕ_χϕ_plot(Cϕ, χϕ, g, τchn, N, dt)
#             χint = trapz(times, χϕ)
#             residual_new = 1.0 - χint * (N * J0 + g^2 * (N * τchn) * χint)
#             println("g = $(round(g, digits=4)), N*τchn = $(round(N * τchn, digits=4)),
#              residual = $(round(residual_new, digits=4)).")
#             flush(stdout)
#             if residual_new < 0.0 || abs(residual_new)<1e-2 || residual_new > residual_old
#                 τchn_boundary[i] = τchn
#                 break
#             end
#             residual_old = residual_new
#             τchn += Δτchn
#         end

#         elapsed = now() - start
#         println("Finished g = $(round(g, digits=4)) ($(i)/$(length(g_vec))); " *
#             "N*τchn = $(round(N * τchn_boundary[i], digits=4)); " *
#             "elapsed = $(round(Dates.value(elapsed) / 1e3, digits=1)) seconds")
#         flush(stdout)
#     end

#     return τchn_boundary
# end

# τchn_boundary = find_hcsc_boundary_b_nonzero(g_vec, N, J0, τchn_init, τrec, Δτchn,
#     T, dt, burn, nIter_list, nTraj_list, damp_R, damp_C)

# file_name = "PosChnHCSCBoundaryN$(N)J0$(replace(string(round(N * J0, digits=1)), "." => "p"))B0p-3.jld2"
# output_file = joinpath(ROOT_DIR, "results", file_name)
# jldsave(output_file;
#     N, J0, num_points, g_vec, T, burn,
    # dt, τchn_boundary)
