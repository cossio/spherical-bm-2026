#!/usr/bin/env julia
# grokking_note_kl_nu_sweep_long.jl
#
# Long-time variant of grokking_note_kl_nu_sweep.jl: same physics, but T_max
# extended from 30 to 1000 with logarithmically-spaced checkpoints, so the
# resulting D_rev(t)/N curves can be displayed on a log-t axis covering
# ≈4.5 decades. All other parameters identical.
#
# Output:
#   ../data/grokking_note_kl_nu_sweep_long.jld2

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using JLD2: jldsave
using LinearAlgebra: BLAS, Symmetric, dot, eigen, mul!, norm
using Printf: @sprintf
using Random: Xoshiro, randn, randn!
using Statistics: mean, std

const SCRIPT_DIR = @__DIR__
const DATA_OUT = joinpath(SCRIPT_DIR, "..", "data")
mkpath(DATA_OUT)

const LOG_PATH = joinpath(DATA_OUT, "grokking_note_kl_nu_sweep_long.log")
open(LOG_PATH, "w") do io end
const LOG_LOCK = ReentrantLock()
function log_msg(s::String)
    lock(LOG_LOCK) do
        open(LOG_PATH, "a") do io; println(io, s); end
    end
end

BLAS.set_num_threads(1)
log_msg(@sprintf("BLAS=%d, Julia=%d", BLAS.get_num_threads(), Threads.nthreads()))

# ─────────────────────────────────────────────────────────────
#  Note formulas (same as grokking_note_kl_nu_sweep.jl)
# ─────────────────────────────────────────────────────────────

function G_sc(z, c_rmt)
    edge = 2/sqrt(c_rmt)
    z ≤ edge ? sqrt(c_rmt) : (c_rmt/2) * (z - sqrt(z^2 - 4/c_rmt))
end
phi_fn(g1, γη) = g1 ≥ 1 ? 1/(4*γη) :
    0.5*(g1/γη + 1/g1 - g1^2/(2γη) + log(g1) - 1)
mu_fn(g1, γη, λ1) = g1 ≥ 1 ? 1 + 1/γη : λ1

function kl_instant(λ1, qv1w, γ, η, ω)
    # qv1w ≡ (v_1(t)·w*)^2, the squared overlap of W(t)'s top eigenvector
    # with the unit teacher direction.  Earlier versions used
    #   q1 = u_{11}^2 · (1 − 1/(2ω−1))
    # as an approximation under the assumption that v_1 is purely along e_1.
    # That approximation has an O(1) bias because the K=2 data covariance
    # also drives e_2; the correct form below uses the directly-computed
    # (v_1·w*)^2.  See `2026-04-26-KL-rev-typ-dynamics/verify_alpha_factorization.jl`
    # for the empirical audit.
    γη = γ*η
    g1 = G_sc(λ1, γη)
    m2 = max(0.0, 1 - g1)
    μ  = mu_fn(g1, γη, λ1)
    φ  = phi_fn(g1, γη)
    q1 = qv1w
    κ  = λ1 * q1
    α  = m2 * q1
    D_for = φ + 0.5*log(ω) - 0.5*(1-1/ω)*κ
    D_rev = (ω - 1 - log(ω))/2 - φ + (μ - 1)/2 - (ω/2)*α
    return (; D_for, D_rev, g1, m2, μ, φ, α, q1)
end
pm_plateau(ω, γ, η) = (ω - 1 - log(ω))/2 + 1/(4*γ*η)

# ─────────────────────────────────────────────────────────────
#  Langevin (upper-triangle, minimal)
# ─────────────────────────────────────────────────────────────

function goe_noise_upper!(rng, W, N, scale)
    @inbounds for j in 1:N
        W[j,j] += scale*sqrt(2)*randn(rng)
        for i in 1:j-1
            W[i,j] += scale*randn(rng)
        end
    end
end
function make_initial_W(rng, N, γ, η)
    σ_off = sqrt(1/(γ*η*N))
    W = zeros(N, N)
    @inbounds for j in 1:N
        W[j,j] = σ_off*sqrt(2)*randn(rng)
        for i in 1:j-1
            W[i,j] = σ_off*randn(rng)
        end
    end
    W
end
function make_initial_x(rng, N, K, s₀)
    x = zeros(N)
    s²_sum = sum(abs2, s₀)
    for a in 1:K
        x[a] = sqrt(N)*s₀[a]
    end
    if N > K
        perp = randn(rng, N-K)
        perp .*= sqrt(N*(1-s²_sum))/norm(perp)
        x[K+1:end] .= perp
    end
    x
end
function run_langevin!(rng, W, x, γ, η, β, ν, c_vec, N, dt, n_steps;
                       save_step_set::Set{Int}=Set{Int}(),
                       on_save=(step,t,W,x)->nothing)
    K = length(c_vec)
    Wx = zeros(N); ξ = zeros(N); x_pred = zeros(N); Wx_pred = zeros(N)
    w_decay = exp(-γ*dt/2); w_drive = (1-w_decay)/(γ/2)
    noise_W_scale = sqrt((1-w_decay^2)/(γ*η*N))
    noise_x_scale = sqrt(2ν*dt/β)
    sampler_coeff = -w_drive*K/(2N)
    half_dt_ν = (dt/2)*ν
    Wsym = Symmetric(W, :U)
    if 0 in save_step_set; on_save(0, 0.0, W, x); end
    for step in 1:n_steps
        rmul!(W, w_decay)
        @inbounds for a in 1:K
            W[a,a] += w_drive*c_vec[a]/2
        end
        BLAS.syr!('U', sampler_coeff, x, W)
        goe_noise_upper!(rng, W, N, noise_W_scale)
        mul!(Wx, Wsym, x)
        U₁ = dot(x, Wx)/N; μ₁ = U₁ + (N-1)/(N*β)
        inv₁ = 1/(1+dt*ν*μ₁)
        @inbounds for i in 1:N
            x_pred[i] = (x[i] + dt*ν*Wx[i])*inv₁
        end
        randn!(rng, ξ)
        x_dot_ξ = dot(x, ξ)/N
        @inbounds for i in 1:N
            ξ[i] = noise_x_scale*(ξ[i] - x_dot_ξ*x[i])
            x_pred[i] += ξ[i]
        end
        x_pred .*= sqrt(N)/norm(x_pred)
        mul!(Wx_pred, Wsym, x_pred)
        U₂ = dot(x_pred, Wx_pred)/N; μ₂ = U₂ + (N-1)/(N*β)
        μ_avg = (μ₁+μ₂)/2
        inv_avg = 1/(1 + dt*ν*μ_avg)
        @inbounds for i in 1:N
            x[i] = (x[i] + half_dt_ν*(Wx[i]+Wx_pred[i]))*inv_avg + ξ[i]
        end
        x .*= sqrt(N)/norm(x)
        if step in save_step_set
            on_save(step, step*dt, W, x)
        end
    end
end

# ─────────────────────────────────────────────────────────────
#  Config (long-T variant)
# ─────────────────────────────────────────────────────────────

const ω_star = 2.5
const c_vec  = [2 - 1/ω_star, 1/ω_star]     # (1.6, 0.4)
const K      = length(c_vec)
const γ_val  = 0.4
const η_val  = 10.0
const β_val  = 1.0
const s0_vec = [0.1, 0.1]
const N      = 1500
const dt     = 0.02
const T_max  = 1000.0
const n_save = 80         # log-spaced
const n_seeds = 4         # per ν
const seed_base = 3000

const ν_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70, 0.85,
                1.00, 1.30, 1.70, 2.20, 3.00]
const n_ν = length(ν_grid)

# Unit teacher direction in the data eigenbasis.  For the rank-one teacher
# in the K=2 setup, w* lives in the (e_1, e_2) plane with
# (e_1·w*)^2 = 1 − 1/(2ω*−1)  and  (e_2·w*)^2 = 1/(2ω*−1).
const ρ²     = 1 - 1/(2*ω_star - 1)
const w_star = let v = zeros(N); v[1] = sqrt(ρ²); v[2] = sqrt(1-ρ²); v end

log_msg(@sprintf("=== ν sweep, long-T variant (log-spaced checkpoints) ==="))
log_msg(@sprintf("Fixed: ω*=%.2f c=(%.2f, %.2f) γ=%.2f η=%.1f β=%.1f",
    ω_star, c_vec[1], c_vec[2], γ_val, η_val, β_val))
log_msg(@sprintf("ν grid: %s", string(ν_grid)))
log_msg(@sprintf("N=%d n_save=%d n_seeds=%d T_max=%.1f (total %d seeds)",
    N, n_save, n_seeds, T_max, n_ν*n_seeds))
log_msg(@sprintf("PM-plateau D_rev/N = %.6f (constant in ν)",
    pm_plateau(ω_star, γ_val, η_val)))

# ─────────────────────────────────────────────────────────────
#  Build log-spaced checkpoint list
# ─────────────────────────────────────────────────────────────

n_steps = round(Int, T_max/dt)
# Logarithmic save grid: t=0 plus (n_save-1) log-spaced points from step=1 to n_steps.
log_steps = round.(Int, exp.(range(log(1), log(n_steps); length=n_save-1)))
save_steps = sort!(unique!(vcat([0], log_steps)))
const N_SAVE = length(save_steps)
save_step_set = Set(save_steps)
t_grid = Float64[s*dt for s in save_steps]

log_msg(@sprintf("n_steps=%d, n_save=%d, t∈[%.3f, %.1f]",
    n_steps, N_SAVE, t_grid[2], t_grid[end]))

# Map step → checkpoint index (so concurrent threads can safely write)
step_to_ckpt = Dict{Int,Int}()
for (k, s) in enumerate(save_steps)
    step_to_ckpt[s] = k
end

# Flat job list
jobs = [(ν_idx, s_idx) for ν_idx in 1:n_ν for s_idx in 1:n_seeds]
n_jobs = length(jobs)

# Outputs: indexed by (checkpoint, ν_idx, seed_idx)
λ1_mat     = zeros(N_SAVE, n_ν, n_seeds)
u1sq_mat   = zeros(N_SAVE, n_ν, n_seeds)   # (v_1·e_1)^2 — kept for figure panel C
qv1w_mat   = zeros(N_SAVE, n_ν, n_seeds)   # (v_1·w*)^2 — used inside kl_instant
Drev_mat   = zeros(N_SAVE, n_ν, n_seeds)
Dfor_mat   = zeros(N_SAVE, n_ν, n_seeds)
g1_mat     = zeros(N_SAVE, n_ν, n_seeds)
m2_mat     = zeros(N_SAVE, n_ν, n_seeds)

tic0 = time()
Threads.@threads for job_idx in 1:n_jobs
    local (ν_idx, s_idx) = jobs[job_idx]
    local ν_val = ν_grid[ν_idx]
    local local_tic = time()
    local rng = Xoshiro(seed_base + 100*ν_idx + s_idx)
    local W = make_initial_W(rng, N, γ_val, η_val)
    local x = make_initial_x(rng, N, K, s0_vec)

    function on_save(step, t, W, x)
        k = step_to_ckpt[step]
        F = eigen(Symmetric(W, :U))
        idx_top = argmax(F.values)
        λ1 = F.values[idx_top]
        v1 = F.vectors[:, idx_top]
        u1sq = v1[1]^2
        qv1w = dot(v1, w_star)^2
        K_inst = kl_instant(λ1, qv1w, γ_val, η_val, ω_star)
        λ1_mat[k, ν_idx, s_idx]   = λ1
        u1sq_mat[k, ν_idx, s_idx] = u1sq
        qv1w_mat[k, ν_idx, s_idx] = qv1w
        Drev_mat[k, ν_idx, s_idx] = K_inst.D_rev
        Dfor_mat[k, ν_idx, s_idx] = K_inst.D_for
        g1_mat[k, ν_idx, s_idx]   = K_inst.g1
        m2_mat[k, ν_idx, s_idx]   = K_inst.m2
    end

    run_langevin!(rng, W, x, γ_val, η_val, β_val, ν_val, c_vec, N, dt, n_steps;
                  save_step_set, on_save)

    log_msg(@sprintf("[ν=%.3f seed=%d done in %.1fs: D_rev_end=%+.4f g1_end=%.4f]",
        ν_val, s_idx, time()-local_tic,
        Drev_mat[end, ν_idx, s_idx], g1_mat[end, ν_idx, s_idx]))
end
log_msg(@sprintf("Total wall: %.1fs", time()-tic0))

# Seed-averages
Drev_mean = zeros(N_SAVE, n_ν)
Drev_se   = zeros(N_SAVE, n_ν)
Dfor_mean = zeros(N_SAVE, n_ν)
λ1_mean   = zeros(N_SAVE, n_ν)
m2_mean   = zeros(N_SAVE, n_ν)
qv1w_mean = zeros(N_SAVE, n_ν)
for k in 1:N_SAVE, j in 1:n_ν
    Drev_mean[k,j] = mean(@view Drev_mat[k,j,:])
    Drev_se[k,j]   = std(@view Drev_mat[k,j,:]) / sqrt(n_seeds)
    Dfor_mean[k,j] = mean(@view Dfor_mat[k,j,:])
    λ1_mean[k,j]   = mean(@view λ1_mat[k,j,:])
    m2_mean[k,j]   = mean(@view m2_mat[k,j,:])
    qv1w_mean[k,j] = mean(@view qv1w_mat[k,j,:])
end

# Dip diagnostics per ν
dip_depth = zeros(n_ν); dip_time = zeros(n_ν)
kl_asymp  = zeros(n_ν); kl_min   = zeros(n_ν)
for j in 1:n_ν
    kl = @view Drev_mean[:,j]
    idx_min = argmin(kl)
    kl_asymp[j] = mean(kl[end-5:end])
    kl_min[j]   = kl[idx_min]
    dip_depth[j] = kl_asymp[j] - kl_min[j]
    dip_time[j]  = t_grid[idx_min]
end

log_msg("\nν       D_rev(0)   D_rev(min)   D_rev(∞)   dip   t_dip   g1(end)")
pm_val = pm_plateau(ω_star, γ_val, η_val)
for j in 1:n_ν
    g1_end = mean(@view g1_mat[end,j,:])
    line = @sprintf("%-6.3f  %.4f    %.4f       %.4f    %+.4f  %5.2f   %.3f",
        ν_grid[j], Drev_mean[1,j], kl_min[j], kl_asymp[j],
        dip_depth[j], dip_time[j], g1_end)
    log_msg(line)
end
log_msg(@sprintf("\n(PM plateau = %.4f; dip = D_rev(∞) − D_rev(min); positive = overshoot)", pm_val))

# ─────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────

jldsave(joinpath(DATA_OUT, "grokking_note_kl_nu_sweep_long.jld2");
    t=t_grid, ν_grid=collect(ν_grid),
    λ1=λ1_mat, u1sq=u1sq_mat, qv1w=qv1w_mat, Drev=Drev_mat, Dfor=Dfor_mat,
    g1=g1_mat, m2=m2_mat,
    Drev_mean=Drev_mean, Drev_se=Drev_se,
    Dfor_mean=Dfor_mean, λ1_mean=λ1_mean, m2_mean=m2_mean, qv1w_mean=qv1w_mean,
    dip_depth=dip_depth, dip_time=dip_time,
    kl_asymp=kl_asymp, kl_min=kl_min,
    pm_plateau=pm_val,
    ω=ω_star, c_vec=c_vec, γ=γ_val, η=η_val, β=β_val, s0=s0_vec,
    N=N, n_seeds=n_seeds, T_max=T_max)
log_msg("Saved → grokking_note_kl_nu_sweep_long.jld2")
