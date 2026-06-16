using JLD2
using LinearAlgebra, FFTW, Dates
include("Utils.jl")
include("DMFT.jl")
using .DMFT
using .Utils

BLAS.set_num_threads(1)
FFTW.set_num_threads(10)

# This should take about 30 seconds.
const N = 2000
const J0 = 0.0
const g_eff_vec = [2.0, 4.0]
const num_points = 10
const dt = 0.025
const T = 100.0
const n_quad = 128
const τ_con_vec = LinRange(0.0, 60.0/N, num_points)
const τ_div_vec = LinRange(0.0, 60.0/N, num_points)


function compute_wrapper(g_eff_vec, τ_vecs)

    ng = length(g_eff_vec)
    nτ = length(τ_vecs)

    Dϕ = Matrix{Float64}(undef, ng, nτ)
    C2 = Matrix{Float64}(undef, ng, nτ)
    C4 = Matrix{Float64}(undef, ng, nτ)
    χϕ_h = Vector{Vector{ComplexF64}}(undef, ng)

    real_numerical_value(x) = abs(imag(x)) <= 1e-12 * max(1.0, abs(real(x))) ? real(x) : error("Expected real value, got $x.")

    function extract_τvec(τ::Tuple)
        idx = findall(x -> !iszero(x), τ)
        if isempty(idx)
            return 0.0
        elseif length(idx) == 1
            return Float64(τ[idx[1]])
        else
            error("Cannot build a 1D τ axis from tuple $τ: more than one nonzero entry.")
        end
    end

    τ_axis = [extract_τvec(τ) for τ in τ_vecs]

    # Zero motif tuple with same length as the input τ tuples
    τ0 = ntuple(_ -> 0.0, length(first(τ_vecs)))

    for (ig, g_eff) in enumerate(g_eff_vec)

        g0 = compute_g(g_eff, τ0)
        sol = DMFT_Stationary_Solver_B0(
            N, J0, g0, τ0;
            dt=dt, T=T, verbose=false, minimal_return=true, n_quad=n_quad
        )
        Cϕh = sol[:Cϕ_h]
        χϕh = sol[:χϕ_h]
        χϕ_h[ig] = χϕh

        for (iτ, τ_tuple) in enumerate(τ_vecs)
            g = compute_g(g_eff, τ_tuple)
            Dϕ_val, C2_val, C4_val = TheoreticalPRϕ(
                N, g, τ_tuple, Cϕh, χϕh, dt; ft_done=true
            )
            Dϕ[ig, iτ] = real_numerical_value(Dϕ_val)
            C2[ig, iτ] = real_numerical_value(C2_val)
            C4[ig, iτ] = real_numerical_value(C4_val)
        end
    end

    return (
        τ = τ_axis,
        Dϕ = Dϕ,
        C2 = C2,
        C4 = C4,
        χϕ_h = χϕ_h,
    )
end


con_params = [
    (0.0, 0.0, τ_con, 0.0)
    for τ_con in τ_con_vec
]
start = now()
result_con = compute_wrapper(g_eff_vec, con_params)
dur = round(Dates.value(now() - start) / 1e3, digits=1)
println("Convergent motif done. Elapsed = $(format_elapsed(start))")

flush(stdout)

div_params = [
    (0.0, 0.0, 0.0, τ_div)
    for τ_div in τ_div_vec
]
start = now()
result_div = compute_wrapper(g_eff_vec, div_params)
println("Divergent motif done. Elapsed = $(format_elapsed(start))")
flush(stdout)

file_name = joinpath(@__DIR__, "results/TheoreticalPRDcondiv.jld2")

jldsave(file_name;
    N, g_eff_vec, dt, T,
    result_con,
    result_div
)
