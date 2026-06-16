using LinearAlgebra
using JLD2
using Dates
using CUDA
include("Utils.jl")
using .Utils
include("DMFT.jl")
using .DMFT_CUDA
BLAS.set_num_threads(1)
CUDA.allowscalar(false)

N = 1000
J0 = 0.0
g = 4.0
τchn = 0.0
τrec = 0.2
τ = (τchn, τrec)
T = 500.0
dt = 0.05
nIter_list = (100, 50, 50, 20)
nTraj_list = (4096, 8192, 16384, 16384)
damp_R = (0.05, 0.2, 0.4, 0.8)
damp_C = (0.05, 0.2, 0.4, 0.8)

model = CreateStationaryDMFTRateModel(N, J0, g, τ, T,
    nIter_list, nTraj_list, damp_R, damp_C; dt=dt)

model_data = (
    N = model.N,
    J0 = model.J0,
    g = model.g,
    τ = (model.τchn, model.τrec, model.τcon, model.τdiv),
    Ttot = model.Ttot,
    dt = model.dt,
    nTime = model.nTime,
    nTraj_list = model.nTraj_list,
    nIte_list = model.nIte_list,
    damp_R = model.damp_R,
    damp_C = model.damp_C
)

Cϕ, χϕ = DMFTStationaryMainloop_TimeInteg(model; verbose = false, return_history=false, minimal_return=true)
CϕFN, χϕFN = DMFTStationaryMainloop(model; verbose = false, return_history=false, minimal_return=true)

file_name = joinpath(@__DIR__ ,"results/DMFTChaosAlgoComp.jld2")
jldsave(file_name; model_data, Cϕ, χϕ, CϕFN, χϕFN)