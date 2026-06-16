using Random, StatsBase, JLD2, Dates, LinearAlgebra
include("Utils.jl")
using .Utils
include("DMFT.jl")
using .DMFT
BLAS.set_num_threads(1)

N = 2000
τchn_vec = [6.0, 8.0, 10.0] ./ N
J0 = -5.0 / N
num_points_dmft = 50
g_vec_dmft = range(0.5, 2.0, num_points_dmft)
μ_dmft = zeros(length(τchn_vec), num_points_dmft)
Δ_dmft = zeros(length(τchn_vec), num_points_dmft)
for (j, τchn) in enumerate(τchn_vec)
    Threads.@threads for i in eachindex(g_vec_dmft)
        g = g_vec_dmft[i]
        sol = DMFT_FP_Gaussian(J0, g, N, (τchn, 0.0); μ0=0.2, Δ0=1.0, max_iter=10000, verbose=true)
        if sol.converged
            μ_dmft[j, i] = sol.μ
            Δ_dmft[j, i] = sol.Δ
        else
            μ_dmft[j, i] = NaN
            Δ_dmft[j, i] = NaN
        end
    end
end

num_points_num = 15
g_vec_num = range(first(g_vec_dmft), last(g_vec_dmft), num_points_num)
num_samples = 10
μ_num = zeros(length(τchn_vec), num_points_num)
Δ_num = zeros(length(τchn_vec), num_points_num)
μ_num_std = zeros(length(τchn_vec), num_points_num)
Δ_num_std = zeros(length(τchn_vec), num_points_num)
println("Numerical integration begins")
flush(stdout)
start = now()
for (j, τchn) in enumerate(τchn_vec)
    Threads.@threads for i in eachindex(g_vec_num)
        g = g_vec_num[i]
        mean_tmp = Float64[]
        var_tmp = Float64[]
        while length(mean_tmp) < num_samples
            J = CreateJ(N, J0, g, (τchn, 0.0); parallel=false)
            x_fixed, _, converged = find_fixed_point(J; tspan=(0.0, 250.0),
                    verbose=false, tol=5e-5, tail_window=5.0, n_tail=10)
            if converged
                push!(mean_tmp, mean(x_fixed))
                push!(var_tmp, var(x_fixed))
            else
                @warn "Didn't find a fixed point for g = $(round(g, digits=4)), N*τchn = $(round(N * τchn, digits=4))."
            end
        end
        μ_num[j, i] = mean(mean_tmp)
        Δ_num[j, i] = mean(var_tmp)
        μ_num_std[j, i] = std(mean_tmp)
        Δ_num_std[j, i] = std(var_tmp)
    end
    println("Nτchn = $(round(N*τchn, digits=4)) finished.")
    println("Time elapsed is $(format_elapsed(start))")
    flush(stdout)
end
jldsave("./results/FP2FPgSweep.jld2"; μ_num, Δ_num, μ_num_std,
    Δ_num_std, μ_dmft, Δ_dmft, τchn_vec, g_vec_num, g_vec_dmft, N, J0)
