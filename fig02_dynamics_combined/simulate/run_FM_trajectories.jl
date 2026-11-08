#!/usr/bin/env julia
# run_FM_trajectories.jl — produce the K=2 FM trajectory used by panels A, B
# of fig:dynamics. Calls the shared MSR/DMFT solver and the finite-N Langevin
# driver in `_lib/msr/`, both at γ=0.4, η=10, β=1, ν=0.7, c=(1.7, 0.3),
# s₀=0.1.
#
# Outputs into ../data/:
#   finN_N3000_seed42_eig_nu0p7.jld2
#   msr_FM_nu0p7.jld2

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

const HERE = @__DIR__
const DATA_DIR = joinpath(HERE, "..", "data")
mkpath(DATA_DIR)

const MSR_SOLVER = joinpath(HERE, "..", "..", "_lib", "msr", "msr_solver.jl")
const FINITEN    = joinpath(HERE, "..", "..", "_lib", "msr", "finiteN.jl")

# ── DMFT trajectory (K=2 FM) ────────────────────────────────────────────
msr_out = joinpath(DATA_DIR, "msr_FM_nu0p7.jld2")
run(`julia $MSR_SOLVER
    --gamma=0.4 --eta=10.0 --beta=1.0 --nu=0.7
    --c=1.7,0.3 --s0=0.1 --Tmax=1000 --nsave=300
    --outfile=$msr_out`)

# ── Finite-N Langevin trajectory (N=3000, single seed, neigen on) ──────
finN_out = joinpath(DATA_DIR, "finN_N3000_seed42_eig_nu0p7.jld2")
run(`julia $FINITEN
    --gamma=0.4 --eta=10.0 --beta=1.0 --nu=0.7
    --c=1.7,0.3 --s0=0.1 --N=3000 --Tmax=1000 --nsave=300
    --neigen=2 --seed=42 --outfile=$finN_out`)

println("\nWrote:")
println("  $msr_out")
println("  $finN_out")
