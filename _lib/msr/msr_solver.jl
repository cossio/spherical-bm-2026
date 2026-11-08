# Arbitrary-K MSR/DMFT causal time-marching solver. Solves the closed
# two-time system in (s_1..s_K, Q, R, κ) by an exponential integrator
# plus trapezoidal quadrature on the Volterra memory kernels.
#
# Usage: julia msr_solver.jl [--gamma=0.5 --eta=3 --beta=1 --nu=1
#                             --c=c1,c2,..,cK --s0=0.05 --Tmax=auto
#                             --outfile=auto]

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using LinearAlgebra, JLD2

# ── Parse command-line arguments ─────────────────────────────────────
function parse_args(args)
    scalar_defaults = Dict(
        "gamma" => 0.5,
        "eta"   => 3.0,
        "beta"  => 1.0,
        "nu"    => 1.0,
        "dt"    => NaN,
        "Tmax"  => NaN,
        "nsave" => 0.0,
        # Optional probe DMFT subsystem (M=1 probe; valid for K∈{1,2}).
        # Set --nprobes=1 to enable, 0 (default) to disable. --nu_probe and
        # --beta_probe default to 10·ν and 10·β (fast/cold probe).
        "nprobes"    => 0.0,
        "nu_probe"   => NaN,
        "beta_probe" => NaN,
        # Optional finite-N noise-floor proxy: clamp |s_a(t)| ≥ s_floor
        # at every step. Mimics the N^{-1/2} thermal floor that prevents
        # finite-N s from decaying below O(N^{-1/2}). Default 0 = off.
        "s_floor"    => 0.0,
        # Save full Q[i,j] and R[i,j] two-time matrices (on save_idx grid)
        # in the output JLD2.  Default 0 = off; set 1 to enable.  Used
        # downstream by Schur-complement spectral closures.
        "save_full_QR" => 0.0,
    )
    c_str  = "1.0"
    s0_str = "0.05"
    sp0_str = "0.0"      # probe initial overlaps with c_a (symmetry breaking)
    outfile = ""
    for arg in args
        if startswith(arg, "--")
            kv = split(arg[3:end], '='; limit=2)
            length(kv) == 2 || error("Bad argument: $arg (expected --key=value)")
            key, val = kv
            if key == "outfile"
                outfile = val
            elseif key == "c"
                c_str = val
            elseif key == "s0"
                s0_str = val
            elseif key == "sp0"
                sp0_str = val
            elseif haskey(scalar_defaults, key)
                scalar_defaults[key] = parse(Float64, val)
            else
                error("Unknown argument: --$key")
            end
        else
            error("Unexpected positional argument: $arg (use --key=value)")
        end
    end
    return scalar_defaults, c_str, s0_str, sp0_str, outfile
end

parsed, c_str, s0_str, sp0_str, outfile_arg = parse_args(ARGS)

# ── Parameters ───────────────────────────────────────────────────────
const γ_val = parsed["gamma"]
const η_val = parsed["eta"]
const β_val = parsed["beta"]
const ν_val = parsed["nu"]

const c_vec = parse.(Float64, split(c_str, ','))
const K = length(c_vec)

# s0: scalar repeated K times, or CSV of length K
s0_vec = let toks = split(s0_str, ',')
    if length(toks) == 1
        fill(parse(Float64, toks[1]), K)
    elseif length(toks) == K
        parse.(Float64, toks)
    else
        error("--s0 must be a scalar or CSV of length K=$K (got $(length(toks)))")
    end
end
sum(abs2, s0_vec) < 1 || error("Σ s0_a² = $(sum(abs2, s0_vec)) must be < 1")

const σ² = 1 / (γ_val * η_val)
const edge_val = 2sqrt(σ²)

# Kernel coefficients
const c_Q = -K * ν_val / 2                     # M(t,u) coefficient of Q
const c_R = ν_val^2 / (η_val * γ_val)          # M coefficient of R = D_reg coeff

# Time discretization
const Δt = isnan(parsed["dt"]) ?
    clamp(sqrt(η_val * γ_val) / (5 * max(ν_val, 0.01)), 0.001, 0.5) : parsed["dt"]
const T_max = isnan(parsed["Tmax"]) ?
    clamp(2/γ_val + 30.0/ν_val, 8.0, 500.0) : parsed["Tmax"]
const N_t = round(Int, T_max / Δt) + 1

const ε_mem = 1e-10
const N_mem = min(N_t, ceil(Int, -2log(ε_mem) / (γ_val * Δt)))

println("MSR solver: K=$K, c=$c_vec, γ=$γ_val, η=$η_val, β=$β_val, ν=$ν_val")
println("  σ² = $(round(σ²; digits=4)), edge = $(round(edge_val; digits=4))")
println("  Δt = $(round(Δt; digits=5)), T_max = $T_max, N_t = $N_t, N_mem = $N_mem")
println("  s₀ = $s0_vec")
println("  Method: exponential integrator + trapezoidal quadrature")

# ── Probe DMFT parameters ─────────────────────────────────────────────
const M_probes   = round(Int, parsed["nprobes"])
const use_probe  = M_probes > 0
M_probes ∈ (0, 1) || error("--nprobes must be 0 or 1 (M=1 probe implementation only; K∈{1,2})")
use_probe && (K ∈ (1, 2) || error("probe DMFT implemented only for K∈{1,2}, got K=$K"))
const ν_p = use_probe ? (isnan(parsed["nu_probe"])   ? 10 * ν_val : parsed["nu_probe"])   : NaN
const β_p = use_probe ? (isnan(parsed["beta_probe"]) ? 10 * β_val : parsed["beta_probe"]) : NaN
if use_probe
    println("  probe DMFT: M=$M_probes, ν_p=$ν_p, β_p=$β_p")
end
const s_floor = parsed["s_floor"]
s_floor >= 0 || error("--s_floor must be ≥ 0")
if s_floor > 0
    println("  finite-N noise-floor proxy: |s_a(t)| ≥ $s_floor enforced at each step")
end

let mem_GB = 2.0 * N_t^2 * sizeof(Float64) / 1e9
    mem_GB > 1 && @warn "Q,R matrices require $(round(mem_GB; digits=1)) GB"
    if mem_GB > 30
        error("N_t=$N_t would need $(round(mem_GB; digits=1)) GB for Q,R. " *
              "Reduce --Tmax or increase --dt.")
    end
end

# ── φ₁(z) = (eᶻ − 1)/z, numerically stable ─────────────────────────
@inline function φ₁(z::Float64)
    if abs(z) < 1e-4
        return @evalpoly(z, 1.0, 0.5, 1/6, 1/24, 1/120)
    else
        return expm1(z) / z
    end
end

# ── Trapezoidal weight ───────────────────────────────────────────────
@inline function trapw(k::Int, lo::Int, hi::Int)
    lo >= hi && return 0.0
    (k == lo || k == hi) ? 0.5 : 1.0
end

# ── Precompute ───────────────────────────────────────────────────────
θ_arr = [(1 - exp(-γ_val * (i-1) * Δt / 2)) / γ_val for i in 1:N_t]
exp_cache = [exp(-γ_val * d * Δt / 2) for d in 0:N_t-1]

# ── Allocate ─────────────────────────────────────────────────────────
Q = zeros(N_t, N_t)
R = zeros(N_t, N_t)
s_arr = zeros(K, N_t)             # s_a(t_n) — integrated directly (linear)
κ_arr = zeros(N_t)

# ── Initial conditions ───────────────────────────────────────────────
for a in 1:K
    s_arr[a, 1] = s0_vec[a]
end
Q[1, 1] = 1.0
R[1, 1] = 1.0

@inline Qval(i, j) = i >= j ? Q[i, j] : Q[j, i]

ΣR_buf = zeros(N_t)

# ── Probe allocations (optional, M=1 probe) ──────────────────────────
# Storage layout:
#   s_p[a, t]:  probe overlap with c_a, K × N_t
#   Λ_arr[t]:   probe Lagrange multiplier (scalar per time)
#   Rp[i, j]:   probe-probe causal response,   i ≥ j; Rp[i,i] = 1
#   Qp[i, j]:   probe-probe correlator,        symmetric, Qp[i,i] = 1
#   B[i, j]:    probe(i)-training(j) correlator, NOT symmetric — full matrix
#   G[i, j]:    probe(i) response to training noise at j, causal; G[i,j]=0 for i≤j
#
# Kernels (M=1):
#   Mpx(t,u) = exp(-γ(t-u)/2) [ -K·ν_p/2 · B(t,u) + ν·ν_p/(ηγ) · G(t,u) ]
#   Mpp(t,u) = exp(-γ(t-u)/2) · ν_p^2/(ηγ) · Rp(t,u)
#   Dpp(t,u) = 2ν_p/β_p δ(t-u) + ν_p^2/(ηγ) · exp(-γ|t-u|/2) · Qp(t,u)
#   Dpx(t,u) = ν·ν_p/(ηγ) · exp(-γ|t-u|/2) · B(t,u)
#
# Kernel coefficients
const cB_Mpx  = use_probe ? (-K * ν_p / 2)              : 0.0
const cG_Mpx  = use_probe ? (ν_val * ν_p / (η_val * γ_val)) : 0.0
const cR_Mpp  = use_probe ? (ν_p^2 / (η_val * γ_val))   : 0.0
const cQ_Dpp  = cR_Mpp                                  # same prefactor as Mpp (on Qp)
const cB_Dpx  = use_probe ? (ν_val * ν_p / (η_val * γ_val)) : 0.0

if use_probe
    # Parse probe initial overlaps. --sp0 = scalar repeated K times, or
    # CSV of length K. With sp0 ≡ 0 (default) the replica-symmetric DMFT
    # has s_p ≡ 0 as a fixed point and the probe never breaks symmetry.
    # Set sp0 > 0 to seed the probe along the data directions; in the
    # gapped/cold/fast limit the asymptote is independent of |sp0| as
    # long as it is small enough.
    sp0_vec = let toks = split(sp0_str, ',')
        if length(toks) == 1
            fill(parse(Float64, toks[1]), K)
        elseif length(toks) == K
            parse.(Float64, toks)
        else
            error("--sp0 must be a scalar or CSV of length K=$K (got $(length(toks)))")
        end
    end
    sum(abs2, sp0_vec) < 1 || error("Σ sp0_a² = $(sum(abs2, sp0_vec)) must be < 1")
    println("  probe init s_p(0) = $sp0_vec  (symmetry-breaking seed)")

    s_p   = zeros(K, N_t)
    Λ_arr = zeros(N_t)
    Rp    = zeros(N_t, N_t)
    Qp    = zeros(N_t, N_t)
    B     = zeros(N_t, N_t)
    G     = zeros(N_t, N_t)
    Rp[1, 1] = 1.0
    Qp[1, 1] = 1.0
    # B(0,0) = ⟨y(0)·x(0)⟩/N. Independent random initial probe → 0 at
    # large N. We keep this at 0 even with sp0 ≠ 0, since sp0 only
    # selects the projection on the c_a directions and does not
    # constrain the bulk overlap with x(0).
    B[1, 1]  = 0.0
    G[1, 1]  = 0.0
    for a in 1:K
        s_p[a, 1] = sp0_vec[a]
    end
else
    s_p   = zeros(K, 0)
    Λ_arr = zeros(0)
    Rp    = zeros(0, 0)
    Qp    = zeros(0, 0)
    B     = zeros(0, 0)
    G     = zeros(0, 0)
end

# Symmetric access to Qp
@inline Qpval(i, j) = i >= j ? Qp[i, j] : Qp[j, i]

Mpx_buf = use_probe ? zeros(N_t) : zeros(0)    # cache e^{-γ(np-ki)/2}·(...) * Δt etc at step np
Mpp_buf = use_probe ? zeros(N_t) : zeros(0)

# ── Main time-marching loop ──────────────────────────────────────────
println("\nRunning MSR time-marching...")
t_start = time()

for ni in 2:N_t
    np = ni - 1
    ki_min = max(1, np - N_mem)

    # ── Phase 1: M kernel cache, κ[np] ──────────────────────────────
    # Σ_a c_a s_a(t_np)²
    sum_cs2 = 0.0
    for a in 1:K
        sum_cs2 += c_vec[a] * s_arr[a, np]^2
    end

    int_κ = 0.0
    for ki in ki_min:np
        ef = exp_cache[np - ki + 1]
        M_ki = (c_Q * Q[np, ki] + c_R * R[np, ki]) * ef
        ΣR_buf[ki] = M_ki
        Dreg_ki = c_R * ef * Q[np, ki]
        w = trapw(ki, ki_min, np)
        int_κ += w * (M_ki * Q[np, ki] + Dreg_ki * R[np, ki])
    end
    κ_np = ν_val * θ_arr[np] * sum_cs2 + ν_val / β_val + Δt * int_κ
    κ_arr[np] = κ_np

    exp_κ = exp(-κ_np * Δt)
    φ₁_κ = φ₁(-κ_np * Δt)

    # ── Phase 2: Advance R(ni, ·) ────────────────────────────────────
    R[ni, ni] = 1.0
    for mi in 1:np
        k_lo = max(mi, ki_min)
        sum_MR = 0.0
        for ki in k_lo:np
            w = trapw(ki, k_lo, np)
            sum_MR += w * ΣR_buf[ki] * R[ki, mi]
        end
        R[ni, mi] = exp_κ * R[np, mi] + φ₁_κ * Δt * (Δt * sum_MR)
    end

    # ── Phase 3: Advance each s_a (linear, exponential integrator) ──────
    # ṡ_a = α_a s_a + g_a(t), g_a(t) = ∫₀^t M(t,u) s_a(u) du.
    # Exponential integrator:  s_a(t+Δt) = exp(α_a Δt) s_a(t) + φ₁(α_a Δt) Δt g_a(t).
    for a in 1:K
        α_a = ν_val * c_vec[a] * θ_arr[np] - κ_np
        g_a = 0.0
        for ki in ki_min:np
            w = trapw(ki, ki_min, np)
            g_a += w * ΣR_buf[ki] * s_arr[a, ki]
        end
        g_a *= Δt                          # g_a(t_np) ≈ ∫ M·s du
        exp_α = exp(α_a * Δt)
        φ₁_α = φ₁(α_a * Δt)
        s_arr[a, ni] = exp_α * s_arr[a, np] + φ₁_α * Δt * g_a
        # Finite-N noise-floor proxy: prevent |s_a| from decaying below s_floor.
        # Preserves sign so the signed dynamics is consistent.
        if s_floor > 0 && abs(s_arr[a, ni]) < s_floor
            sgn = s_arr[a, ni] >= 0 ? 1.0 : -1.0
            s_arr[a, ni] = sgn * s_floor
        end
    end

    # ── Phase 4: Advance Q(ni, ·) ────────────────────────────────────
    Q[ni, ni] = 1.0
    for mi in 1:np
        sum_MQ = 0.0
        for ki in ki_min:np
            w = trapw(ki, ki_min, np)
            sum_MQ += w * ΣR_buf[ki] * Qval(ki, mi)
        end

        ki_lo_d = max(1, np - N_mem)
        sum_DR = 0.0
        for ki in ki_lo_d:mi
            ef = exp_cache[np - ki + 1]
            Dreg_ki = c_R * ef * Q[np, ki]
            w = trapw(ki, ki_lo_d, mi)
            sum_DR += w * Dreg_ki * R[mi, ki]
        end

        # Source: νθ Σ_a c_a s_a(t) s_a(t')
        source = 0.0
        for a in 1:K
            source += c_vec[a] * s_arr[a, np] * s_arr[a, mi]
        end
        source *= ν_val * θ_arr[np]

        Q[ni, mi] = exp_κ * Q[np, mi] + φ₁_κ * Δt * (source + Δt * sum_MQ + Δt * sum_DR)
    end

    # ── Phase 5 (probe DMFT): build probe kernel caches at np ──────
    # Only runs if use_probe. The probe subsystem sees the training bath
    # (Q, R, s_a, κ) as an input and updates its own two-time objects.
    if use_probe
        # Mpx_buf[ki] = e^{-γ(np-ki)/2} [ cB_Mpx·B(np,ki) + cG_Mpx·G(np,ki) ]
        # Mpp_buf[ki] = e^{-γ(np-ki)/2} ·  cR_Mpp · Rp(np, ki)
        for ki in ki_min:np
            ef = exp_cache[np - ki + 1]
            Mpx_buf[ki] = ef * (cB_Mpx * B[np, ki] + cG_Mpx * G[np, ki])
            Mpp_buf[ki] = ef * cR_Mpp * Rp[np, ki]
        end

        # Compute Λ(np) from the equal-time Q_p(t,t)=1 constraint:
        # Λ = (ν_p/γ)(1-e^{-γt/2}) Σ_a c_a s_p_a² + ν_p/β_p
        #     + ∫₀^t Mpx(t,u) B(u,t) du
        #     + ∫₀^t Mpp(t,u) Qp(u,t) du
        #     + ∫₀^t Dpp_reg(t,u) Rp(t,u) du
        #     + ∫₀^t Dpx(t,u) G(t,u) du
        sum_sp2 = 0.0
        for a in 1:K
            sum_sp2 += c_vec[a] * s_p[a, np]^2
        end
        int_Λ = 0.0
        for ki in ki_min:np
            ef = exp_cache[np - ki + 1]
            w = trapw(ki, ki_min, np)
            # ∫ Mpx · B(u,t)|_{t=np, u=ki} = Mpx(np,ki) · B(ki, np) (column of B)
            int_Λ += w * Mpx_buf[ki] * B[ki, np]
            # ∫ Mpp · Qp(u,t)|_{t=np} = Mpp(np,ki) · Qp(ki, np)  (Qp symmetric)
            int_Λ += w * Mpp_buf[ki] * Qpval(ki, np)
            # ∫ Dpp_reg(np,ki) · Rp(np,ki):  (regular part; delta already pulled out)
            int_Λ += w * cQ_Dpp * ef * Qpval(np, ki) * Rp[np, ki]
            # ∫ Dpx(np,ki) · G(np,ki):
            int_Λ += w * cB_Dpx * ef * B[np, ki] * G[np, ki]
        end
        # (ν_p/γ)·(1-e^{-γt/2}) = ν_p · θ_arr[np] (recall θ(t) = (1-e^{-γt/2})/γ).
        Λ_np = ν_p * θ_arr[np] * sum_sp2 + ν_p / β_p + Δt * int_Λ
        Λ_arr[np] = Λ_np

        exp_Λ = exp(-Λ_np * Δt)
        φ₁_Λ = φ₁(-Λ_np * Δt)

        # ── Phase 5a: Advance Rp(ni, mi) for mi ∈ [1, np] ─────────
        # ∂_t Rp = -Λ Rp + ∫_{t'}^t Mpp Rp du,  Rp(t,t) = 1.
        Rp[ni, ni] = 1.0
        for mi in 1:np
            k_lo = max(mi, ki_min)
            s_mr = 0.0
            for ki in k_lo:np
                w = trapw(ki, k_lo, np)
                s_mr += w * Mpp_buf[ki] * Rp[ki, mi]
            end
            Rp[ni, mi] = exp_Λ * Rp[np, mi] + φ₁_Λ * Δt * (Δt * s_mr)
        end

        # ── Phase 5b: Advance G(ni, mi) for mi ∈ [1, np] ──────────
        # ∂_t G = -Λ G + ∫_{t'}^t Mpx(t,u) R(u,t') du
        #             + ∫_{t'}^t Mpp(t,u) G(u,t') du
        # G(t,t') = 0 for t ≤ t' (causal).
        G[ni, ni] = 0.0
        for mi in 1:np
            k_lo = max(mi, ki_min)
            s_MR = 0.0
            s_MG = 0.0
            for ki in k_lo:np
                w = trapw(ki, k_lo, np)
                s_MR += w * Mpx_buf[ki] * R[ki, mi]
                s_MG += w * Mpp_buf[ki] * G[ki, mi]
            end
            G[ni, mi] = exp_Λ * G[np, mi] + φ₁_Λ * Δt * (Δt * s_MR + Δt * s_MG)
        end

        # ── Phase 5c: Advance Qp(ni, mi) for mi ∈ [1, np] ─────────
        # ∂_t Qp = -Λ Qp + (ν_p/γ)(1-e^{-γt/2}) Σ_a c_a s_p_a(t) s_p_a(t')
        #              + ∫₀^t Mpx(t,u) B(u,t') du
        #              + ∫₀^t Mpp(t,u) Qp(u,t') du
        #              + ∫₀^{t'} Dpp_reg(t,u) Rp(t',u) du   [delta at u=t outside [0,t'] if t>t']
        #              + ∫₀^{t'} Dpx(t,u) G(t',u) du
        Qp[ni, ni] = 1.0
        for mi in 1:np
            s_MB = 0.0
            s_MQ = 0.0
            for ki in ki_min:np
                w = trapw(ki, ki_min, np)
                s_MB += w * Mpx_buf[ki] * B[ki, mi]
                s_MQ += w * Mpp_buf[ki] * Qpval(ki, mi)
            end
            ki_lo_d = max(1, np - N_mem)
            s_DR = 0.0
            s_DG = 0.0
            for ki in ki_lo_d:mi
                ef = exp_cache[np - ki + 1]
                w = trapw(ki, ki_lo_d, mi)
                s_DR += w * cQ_Dpp * ef * Qpval(np, ki) * Rp[mi, ki]
                s_DG += w * cB_Dpx * ef * B[np, ki] * G[mi, ki]
            end
            src_p = 0.0
            for a in 1:K
                src_p += c_vec[a] * s_p[a, np] * s_p[a, mi]
            end
            src_p *= ν_p * θ_arr[np]
            Qp[ni, mi] = exp_Λ * Qp[np, mi] +
                          φ₁_Λ * Δt * (src_p + Δt * s_MB + Δt * s_MQ + Δt * s_DR + Δt * s_DG)
        end

        # ── Phase 5d: Advance B ROW B(ni, mi) for mi ∈ [1, np] ────
        # ∂_t B = -Λ B + (ν_p/γ)(1-e^{-γt/2}) Σ_a c_a s_p_a(t) s_a(t')
        #              + ∫₀^t Mpx(t,u) Q(u,t') du
        #              + ∫₀^t Mpp(t,u) B(u,t') du
        #              + ∫₀^{t'} Dpx(t,u) R(t',u) du
        for mi in 1:np
            s_MQ = 0.0
            s_MB = 0.0
            for ki in ki_min:np
                w = trapw(ki, ki_min, np)
                s_MQ += w * Mpx_buf[ki] * Qval(ki, mi)
                s_MB += w * Mpp_buf[ki] * B[ki, mi]
            end
            ki_lo_d = max(1, np - N_mem)
            s_DR = 0.0
            for ki in ki_lo_d:mi
                ef = exp_cache[np - ki + 1]
                w = trapw(ki, ki_lo_d, mi)
                s_DR += w * cB_Dpx * ef * B[np, ki] * R[mi, ki]
            end
            src_B = 0.0
            for a in 1:K
                src_B += c_vec[a] * s_p[a, np] * s_arr[a, mi]
            end
            src_B *= ν_p * θ_arr[np]
            B[ni, mi] = exp_Λ * B[np, mi] +
                         φ₁_Λ * Δt * (src_B + Δt * s_MQ + Δt * s_MB + Δt * s_DR)
        end

        # ── Phase 5e: Advance B COLUMN B(mi, ni) for mi ∈ [1, np] ──
        # By symmetry with the training effective process:
        # ∂_{t'} B(t,t') = -κ(t') B(t,t') + (ν/γ)(1-e^{-γt'/2}) Σ_a c_a s_a(t') s_p_a(t)
        #                  + ∫₀^{t'} M_training(t',u) B(t,u) du
        #                  + ∫₀^t Dpx(u,t') Rp(t,u) du
        # We evolve the t'-direction from t_np to t_ni = t_{np+1}, holding t=mi fixed.
        # Use the exponential integrator with the training κ(np), since t'=np is the
        # "current" training time.  Training kernels at np already cached in ΣR_buf.
        for mi in 1:np
            s_MB_col = 0.0
            for ki in ki_min:np
                w = trapw(ki, ki_min, np)
                # ∫₀^{t'=np} M_training(t',u=ki) · B(t=mi, u=ki) du
                s_MB_col += w * ΣR_buf[ki] * B[mi, ki]
            end
            k_lo = max(1, np - N_mem)
            s_DR_col = 0.0
            for ki in k_lo:mi
                ef = exp_cache[np - ki + 1]
                w = trapw(ki, k_lo, mi)
                # ∫₀^{t=mi} Dpx(u=ki, t'=np) · Rp(t=mi, u=ki) du
                # Dpx(u,t') = ν·ν_p/(ηγ) e^{-γ(t'-u)/2} B(u,t')   for t' ≥ u
                # Here u=ki, t'=np → e^{-γ(np-ki)/2}·B(ki, np)
                s_DR_col += w * cB_Dpx * ef * B[ki, np] * Rp[mi, ki]
            end
            src_Bc = 0.0
            for a in 1:K
                src_Bc += c_vec[a] * s_arr[a, np] * s_p[a, mi]
            end
            src_Bc *= ν_val * θ_arr[np]
            # Column update uses training κ (not Λ)
            B[mi, ni] = exp_κ * B[mi, np] +
                         φ₁_κ * Δt * (src_Bc + Δt * s_MB_col + Δt * s_DR_col)
        end

        # Diagonal B(ni, ni): combine row and column end.  At stationarity
        # it should be consistent; here take the average as the safest estimate.
        if true
            # One-step row estimate: apply ∂_t with mi=np+1 (not in past, so skip)
            # Take the row/column converged equal-time limit: extrapolate from last row step.
            B[ni, ni] = (B[ni, np] + B[np, ni]) / 2  # mild smoothing; both should agree at eq.
        end

        # ── Phase 5f: Advance s_p_a(ni) ─────────────────────────────
        # ṡ_p_a = (ν_p c_a/γ)(1-e^{-γt/2}) s_p_a - Λ s_p_a
        #         + ∫₀^t Mpx(t,u) s_a(u) du + ∫₀^t Mpp(t,u) s_p_a(u) du
        for a in 1:K
            α_pa = ν_p * c_vec[a] * θ_arr[np] - Λ_np
            g_p  = 0.0
            for ki in ki_min:np
                w = trapw(ki, ki_min, np)
                g_p += w * (Mpx_buf[ki] * s_arr[a, ki] + Mpp_buf[ki] * s_p[a, ki])
            end
            g_p *= Δt
            exp_pa = exp(α_pa * Δt)
            φ₁_pa  = φ₁(α_pa * Δt)
            s_p[a, ni] = exp_pa * s_p[a, np] + φ₁_pa * Δt * g_p
        end
    end

    if (ni - 1) % 250 == 0
        t_now = (ni - 1) * Δt
        elapsed = time() - t_start
        s_now = [s_arr[a, ni] for a in 1:K]
        println("  t=$(round(t_now; digits=1))  " *
                "s=$(round.(s_now; digits=5))  " *
                "κ=$(round(κ_np; digits=4))  ($(round(elapsed; digits=1))s)")
        flush(stdout)
    end
end

κ_arr[N_t] = κ_arr[N_t - 1]
if use_probe
    Λ_arr[N_t] = Λ_arr[N_t - 1]
end

elapsed_total = time() - t_start
println("Done in $(round(elapsed_total; digits=1))s")

# ── Derived observables ──────────────────────────────────────────────
# Signal block per channel: S_a(t) = c_a θ(t) − (K/2) ∫ e^{−γ(t−u)/2} s_a(u)² du
# (follows from W = W_GOE + Σ_a (θ_a/N) c_a c_a^T − K/(2N) ∫ e^{-γ(t-u)/2} x x^T du,
#  contracted with c_a/√N on both sides and using c_a·c_b = N δ_ab)
println("\nComputing derived observables...")
S_mat = zeros(K, N_t)
C_arr = zeros(N_t)

for ni in 1:N_t
    ki_lo = max(1, ni - N_mem)
    for a in 1:K
        int_s2 = 0.0
        for ki in ki_lo:ni
            w = trapw(ki, ki_lo, ni)
            int_s2 += w * exp_cache[ni - ki + 1] * s_arr[a, ki]^2
        end
        S_mat[a, ni] = c_vec[a] * θ_arr[ni] - (K / 2) * Δt * int_s2
    end
    C_arr[ni] = Qval(ni, 1)
end

println("\nFinal state (t=$T_max):")
for a in 1:K
    println("  s_$a = $(round(s_arr[a, end]; digits=5))   " *
            "S_$a = $(round(S_mat[a, end]; digits=5))")
end
println("  κ  = $(round(κ_arr[end]; digits=5))")
println("  C  = $(round(C_arr[end]; digits=5))")
if use_probe
    # Top eigenvalue estimator: for M=1 probe in the fast/cold limit,
    # the probe Ritz value is U_p = Λ/ν_p - 1/β_p ≈ λ_top(W).
    Up_est = Λ_arr[end] / ν_p - 1 / β_p
    println("  Λ  = $(round(Λ_arr[end]; digits=5))")
    println("  λ_top (probe Ritz estimator U_p = Λ/ν_p - 1/β_p) = $(round(Up_est; digits=5))")
    for a in 1:K
        println("  s_p_$a = $(round(s_p[a, end]; digits=5))  " *
                "(probe overlap with c_$a)")
    end
end

# ── Save results ─────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "..", "data", "MSR")
mkpath(datadir)
t_grid = [(i-1)*Δt for i in 1:N_t]

n_save = round(Int, parsed["nsave"])
if n_save > 0 && n_save < N_t
    save_idx = unique(round.(Int, range(1, N_t, length=n_save)))
else
    save_idx = 1:N_t
end

ν_tag = replace(string(ν_val), "." => "p")
c_tag = replace(join(c_vec, "_"), "." => "p")
outfile = isempty(outfile_arg) ?
    joinpath(datadir, "20260406_msr_K$(K)_c$(c_tag)_nu$(ν_tag).jld2") : outfile_arg

probe_payload = use_probe ? Dict(
    "M_probes"  => M_probes,
    "nu_p"      => ν_p,
    "beta_p"    => β_p,
    "Lambda"    => Λ_arr[save_idx],
    "s_p"       => s_p[:, save_idx],                      # K × nsave
    "lambda_top"=> [Λ_arr[i]/ν_p - 1/β_p for i in save_idx],
    "B_diag"    => [B[i, i] for i in save_idx],           # b(t) = v₁·x/√N (probe-training equal-time overlap)
) : Dict{String,Any}()

# Optional: save full Q (and R) matrices on the save_idx grid.
# Useful for downstream Schur-complement spectral computations that
# need the bulk resolvent G_⊥(u,v;λ,t) ≈ G_sc(λ)·[Q(u,v) − Σ s_c(u)s_c(v)].
# Memory: nsave² × 8 bytes per matrix; at nsave=500 that's 2 MB per matrix.
const save_full_QR = haskey(parsed, "save_full_QR") && parsed["save_full_QR"] > 0

if save_full_QR
    Q_full = zeros(length(save_idx), length(save_idx))
    R_full = zeros(length(save_idx), length(save_idx))
    for (ka, ia) in enumerate(save_idx), (kb, ib) in enumerate(save_idx)
        Q_full[ka, kb] = Qval(ia, ib)
        R_full[ka, kb] = ia >= ib ? R[ia, ib] : 0.0
    end
else
    Q_full = zeros(0, 0); R_full = zeros(0, 0)
end

jldsave(outfile;
    t = t_grid[save_idx],
    s = s_arr[:, save_idx],                  # K × nsave matrix
    κ = κ_arr[save_idx],
    S = S_mat[:, save_idx],                  # K × nsave matrix
    C = C_arr[save_idx],
    c = c_vec,
    K = K,
    use_probe = use_probe,
    probe = probe_payload,
    Q_slice_0 = [Qval(ni, 1) for ni in save_idx],
    Q_slice_mid = let mid = div(N_t, 2); [Qval(ni, mid) for ni in save_idx] end,
    Q_full = Q_full,
    R_full = R_full,
    save_full_QR = save_full_QR,
    params = Dict("γ"=>γ_val, "η"=>η_val, "β"=>β_val, "ν"=>ν_val,
                   "c"=>c_vec, "K"=>K,
                   "σ²"=>σ², "edge"=>edge_val, "Δt"=>Δt, "s₀"=>s0_vec,
                   "M_probes"=>M_probes, "nu_p"=>(use_probe ? ν_p : NaN),
                   "beta_p"=>(use_probe ? β_p : NaN),
                   "sp0"=>sp0_str,
                   "method"=>"MSR_Kgen_exp_integrator_trapezoidal")
)
println("\nSaved to $outfile")
