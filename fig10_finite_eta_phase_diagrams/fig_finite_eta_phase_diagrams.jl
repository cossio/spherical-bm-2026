# Finite-η phase structure: 4 rows × 3 columns (η ∈ {1, 3, 10}).
# Row 1: phase boundary ν_c(γ); rows 2–4: time-marched s_∞(γ) at two
# ν values, then s(t) in the PM and FM phases. β = 1, K = 1, c = 1.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))
using LinearAlgebra, JLD2, CairoMakie, Printf, LaTeXStrings
using Base.Threads: @threads, nthreads
using Statistics: mean

const β_fixed = 1.0
const η_list  = [1.0, 3.0, 10.0]

# ═══════════════════════════════════════════════════════════════════════
# A. Paramagnetic-bath (TTI stationary) solver — for ν_c(γ) phase boundary
# ═══════════════════════════════════════════════════════════════════════
# Fine τ-grid with convergence guard ν_c·Δτ < 0.6.
const N_τ_b   = 5000
const τ_max_b = 25.0
const Δτ_b    = τ_max_b / (N_τ_b - 1)
const NU_DT_MAX = 0.6

const _EX_b = zeros(N_τ_b)
const _M_b  = zeros(N_τ_b); const _Dr_b = zeros(N_τ_b)
const _AM_b = zeros(N_τ_b); const _AD_b = zeros(N_τ_b)
const _R_b  = fill(1.0, N_τ_b); const _Q_b  = fill(1.0, N_τ_b)
const _Rn_b = zeros(N_τ_b); const _Qn_b = zeros(N_τ_b)
const _μ_b  = Ref(0.01)

@inline function φ₁(z::Float64)
    abs(z) < 1e-4 ? @evalpoly(z, 1.0, 0.5, 1/6, 1/24) : expm1(z) / z
end

function tail_fit(R::AbstractVector, Δτ; tail_n=500)
    n = length(R); i0 = max(2, n - tail_n + 1)
    y = @view R[i0:n]
    all(y .> 0) || return (0.0, 1e-6)
    logy = log.(y)
    x = ((i0:n) .- 1) .* Δτ
    m = length(x); sx = sum(x); sy = sum(logy)
    sxx = sum(x .* x); sxy = sum(x .* logy)
    denom = m * sxx - sx * sx
    denom ≈ 0 && return (y[end], 1e-6)
    slope = (m * sxy - sx * sy) / denom
    A = exp((sy - slope * sx) / m); μt = -slope
    return (A, μt)
end

function setup_gamma_b!(γ)
    @inbounds for i in 1:N_τ_b
        _EX_b[i] = exp(-γ * (i-1) * Δτ_b / 2)
    end
end
function cold_start_b!(β, ν)
    μ = max(ν / β, 0.01); _μ_b[] = μ
    @inbounds for i in 1:N_τ_b
        _R_b[i] = exp(-μ * (i-1) * Δτ_b); _Q_b[i] = _R_b[i]
    end
end

function solve_bath!(γ, η, β, ν; max_iter=40, tol=5e-7)
    c_Q = -ν / 2; c_R = ν^2 / (η * γ)
    μ = _μ_b[]
    for _ in 1:max_iter
        μ_old = μ
        @inbounds @. _M_b  = _EX_b * (c_Q * _Q_b + c_R * _R_b)
        @inbounds @. _Dr_b = _EX_b * (c_R * _Q_b)
        μ = ν / β + Δτ_b * (dot(_M_b, _Q_b) + dot(_Dr_b, _R_b))
        isfinite(μ) || break
        decay = exp(-μ * Δτ_b); phi = φ₁(-μ * Δτ_b)
        _Rn_b[1] = 1.0
        @inbounds for n in 2:N_τ_b
            conv = 0.0
            @simd for k in 1:n-1; conv += _M_b[n-k] * _Rn_b[k]; end
            conv *= Δτ_b
            _Rn_b[n] = decay * _Rn_b[n-1] + Δτ_b * phi * conv
        end
        @inbounds for i in 1:N_τ_b
            sA = 0.0; sD = 0.0
            @simd for j in 1:(N_τ_b - i + 1)
                k = i + j - 1
                sA += _M_b[k]  * _Q_b[j]
                sD += _Dr_b[k] * _Rn_b[j]
            end
            _AM_b[i] = sA * Δτ_b
            _AD_b[i] = sD * Δτ_b
        end
        _Qn_b[1] = 1.0
        @inbounds for n in 2:N_τ_b
            conv = 0.0
            @simd for k in 1:n-1; conv += _M_b[n-k] * _Qn_b[k]; end
            conv *= Δτ_b
            _Qn_b[n] = decay * _Qn_b[n-1] + Δτ_b * phi * (conv + _AM_b[n-1] + _AD_b[n-1])
        end
        @inbounds for i in 1:N_τ_b; _R_b[i] = _Rn_b[i]; _Q_b[i] = _Qn_b[i]; end
        if abs(μ - μ_old) < tol * max(abs(μ), 1.0); break; end
    end
    _μ_b[] = μ
    χ_short = Δτ_b * sum(_R_b) - 0.5 * Δτ_b * (_R_b[1] + _R_b[N_τ_b])
    A, μ_tail = tail_fit(_R_b, Δτ_b)
    tail_int = (μ_tail > 1e-4 && A > 0) ? A * exp(-μ_tail * τ_max_b) / μ_tail : 0.0
    χ = χ_short + tail_int
    bath_valid = isfinite(μ) && μ > 1e-4 && isfinite(χ) && χ > 0
    return (; χ, μ, bath_valid)
end

function f_nuchi(γ, η, β, ν; cold=false, max_iter=100)
    cold && cold_start_b!(β, ν)
    s = solve_bath!(γ, η, β, ν; max_iter)
    return s.bath_valid ? (ν * s.χ - γ) : NaN
end

function bisect_nu_c(γ, η, β, ν_lo, ν_hi; tol=3e-3)
    while log(ν_hi) - log(ν_lo) > tol
        νm = exp(0.5 * (log(ν_lo) + log(ν_hi)))
        fm = f_nuchi(γ, η, β, νm)
        if isnan(fm) || fm < 0; ν_lo = νm else ν_hi = νm end
    end
    exp(0.5 * (log(ν_lo) + log(ν_hi)))
end

function ensure_bracket(γ, η, β, ν_guess)
    ν_lo = ν_guess * 0.35; ν_hi = ν_guess * 2.50
    f_lo = f_nuchi(γ, η, β, ν_lo)
    f_hi = f_nuchi(γ, η, β, ν_hi)
    # Broader, cold-restart-tolerant bracket search: on NaN (bath failed to
    # converge from the warm state), retry once with a cold start before
    # stepping further.  Also allow ν_lo to shrink further (to 1e-8) — at
    # small γ the boundary can dip well below 1e-3.
    t = 0
    while (isnan(f_lo) || f_lo ≥ 0) && ν_lo > 1e-8 && t < 24
        if isnan(f_lo)
            f_retry = f_nuchi(γ, η, β, ν_lo; cold=true, max_iter=200)
            if !isnan(f_retry) && f_retry < 0
                f_lo = f_retry; break
            end
        end
        ν_lo *= 0.4; f_lo = f_nuchi(γ, η, β, ν_lo); t += 1
    end
    t = 0
    while (isnan(f_hi) || f_hi ≤ 0) && ν_hi < 1e7 && t < 20
        if isnan(f_hi)
            f_retry = f_nuchi(γ, η, β, ν_hi; cold=true, max_iter=200)
            if !isnan(f_retry) && f_retry > 0
                f_hi = f_retry; break
            end
        end
        ν_hi *= 2.5; f_hi = f_nuchi(γ, η, β, ν_hi); t += 1
    end
    (isnan(f_lo) || f_lo ≥ 0 || isnan(f_hi) || f_hi ≤ 0) && return nothing
    return (ν_lo, ν_hi)
end

function locate_seed(η, β; γ₀_candidates=(0.7, 0.5, 0.3, 0.85, 0.9, 0.8, 0.2, 0.1))
    for γ₀ in γ₀_candidates
        setup_gamma_b!(γ₀)
        ν_probe = 10.0 .^ range(-2.0, 2.5; length=45)
        fs = [f_nuchi(γ₀, η, β, ν; cold=true, max_iter=300) for ν in ν_probe]
        idx = nothing
        for i in length(ν_probe):-1:2
            a, b = fs[i-1], fs[i]
            if isfinite(a) && isfinite(b) && a < 0 && b > 0
                idx = (i-1, i); break
            end
        end
        idx === nothing && continue
        ν_lo, ν_hi = ν_probe[idx[1]], ν_probe[idx[2]]
        while log(ν_hi) - log(ν_lo) > 1e-4
            νm = exp(0.5 * (log(ν_lo) + log(ν_hi)))
            fm = f_nuchi(γ₀, η, β, νm; cold=true, max_iter=300)
            if isnan(fm) || fm < 0; ν_lo = νm else ν_hi = νm end
        end
        νc₀ = exp(0.5 * (log(ν_lo) + log(ν_hi)))
        if νc₀ * Δτ_b < NU_DT_MAX
            cold_start_b!(β, νc₀); solve_bath!(γ₀, η, β, νc₀; max_iter=300)
            return (γ₀, νc₀)
        end
    end
    return nothing
end

function trace_eta(η; β=β_fixed, Δγ=0.02, γ_min_stop=0.05, γ_max_stop=1.05,
                   time_budget=60.0)
    seed = locate_seed(η, β)
    seed === nothing && return (Float64[], Float64[])
    γ₀, νc₀ = seed
    t_start = time()
    println(@sprintf("    seed: γ₀=%.3f  ν_c=%.4g", γ₀, νc₀))
    flush(stdout)
    γ_up = Float64[γ₀]; νc_up = Float64[νc₀]
    for step in 1:400
        if time() - t_start > time_budget
            println(@sprintf("    ⏰ time budget (%.0fs) reached on upward walk at γ=%.3f", time_budget, γ₀ + Δγ * step))
            break
        end
        γ = γ₀ + Δγ * step
        γ > γ_max_stop && break
        setup_gamma_b!(γ)
        br = ensure_bracket(γ, η, β, νc_up[end])
        br === nothing && break
        νc = bisect_nu_c(γ, η, β, br[1], br[2])
        !isfinite(νc) && break
        νc * Δτ_b > NU_DT_MAX && break
        push!(γ_up, γ); push!(νc_up, νc)
    end
    cold_start_b!(β, νc₀); solve_bath!(γ₀, η, β, νc₀; max_iter=100)
    γ_down = Float64[]; νc_down = Float64[]; νc_prev = νc₀
    t_down_start = time()
    for step in 1:400
        if time() - t_down_start > time_budget
            println(@sprintf("    ⏰ time budget (%.0fs) reached on downward walk at γ=%.3f", time_budget, γ₀ - Δγ * step))
            break
        end
        γ = γ₀ - Δγ * step
        γ < γ_min_stop && break
        setup_gamma_b!(γ)
        br = ensure_bracket(γ, η, β, νc_prev)
        if br === nothing
            # Warm state may have drifted onto a degenerate branch; try a
            # cold re-seed at νc_prev before accepting the failure.
            cold_start_b!(β, νc_prev); solve_bath!(γ, η, β, νc_prev; max_iter=300)
            br = ensure_bracket(γ, η, β, νc_prev)
            br === nothing && (println(@sprintf("    ↓ bracket failed at γ=%.3f (ν_prev=%.3g) — stopping", γ, νc_prev)); break)
        end
        νc = bisect_nu_c(γ, η, β, br[1], br[2])
        (!isfinite(νc) || νc < 1e-8) && break
        push!(γ_down, γ); push!(νc_down, νc); νc_prev = νc
    end
    (vcat(reverse(γ_down), γ_up), vcat(reverse(νc_down), νc_up))
end

# ═══════════════════════════════════════════════════════════════════════
# B. Stationary TTI solver for s_∞ (FM branch too) — adapted from
#    20260408_plot_s_vs_nu.jl.
# ═══════════════════════════════════════════════════════════════════════
function solve_stationary(γ, η, β, ν;
                          N_τ=600, maxiter=600, tol=1e-10,
                          s_prev=0.0,
                          s_scan=vcat(range(0.0, 0.15, length=20),
                                      range(0.15, 0.9, length=40)[2:end],
                                      range(0.9, 0.998, length=15)[2:end]))
    c_Q = -ν / 2; c_R = ν^2 / (η * γ)
    Δτ = clamp(sqrt(η * γ) / (5 * max(ν, 0.01)), 0.005, 0.2)
    EXP = [exp(-γ * k * Δτ / 2) for k in 0:2N_τ]
    tw = (k, lo, hi) -> lo >= hi ? 0.0 : (k == lo || k == hi) ? 0.5 : 1.0
    function solve_bath(s; Q0=nothing, R0=nothing, μ0=NaN)
        Q = Q0 !== nothing ? copy(Q0) :
            [s^2 + (1 - s^2) * EXP[k] for k in 1:N_τ]
        R = R0 !== nothing ? copy(R0) :
            Float64[EXP[k] for k in 1:N_τ]
        μ = isnan(μ0) ? ν / β + (ν / γ) * s^2 + 1.0 : μ0
        Qn = similar(Q); Rn = similar(R); src = (ν / γ) * s^2
        for _ in 1:maxiter
            Σ = [EXP[k] * (c_Q * Q[k] + c_R * R[k]) for k in 1:N_τ]
            iμ = 0.0
            for k in 1:N_τ; w = tw(k, 1, N_τ)
                iμ += w * (Σ[k] * Q[k] + c_R * EXP[k] * Q[k] * R[k]); end
            μn = src + ν / β + Δτ * iμ
            eμ = exp(-μn * Δτ); pμ = φ₁(-μn * Δτ)
            Rn[1] = 1.0
            for k in 2:N_τ; ir = 0.0
                for j in 1:k-1; ir += tw(j, 1, k-1) * Σ[j] * Rn[k-j]; end
                Rn[k] = eμ * Rn[k-1] + pμ * Δτ * (Δτ * ir); end
            Inc = Vector{Float64}(undef, N_τ)
            for k in 1:N_τ; Jm = N_τ - k + 1; v2 = 0.0; vd = 0.0
                for j in 1:Jm; w = tw(j, 1, Jm); idx = j + k - 1; ef = EXP[idx]
                    v2 += w * ef * (c_Q * Q[idx] + c_R * R[idx]) * Q[j]
                    vd += w * c_R * ef * Q[idx] * Rn[j]; end
                Inc[k] = Δτ * (v2 + vd); end
            Qn[1] = 1.0
            for k in 2:N_τ; iq = 0.0
                for j in 1:k-1; iq += tw(j, 1, k-1) * Σ[j] * Qn[k-j]; end
                Qn[k] = eμ * Qn[k-1] + pμ * Δτ * (src + Δτ * iq + Inc[k-1]); end
            err = abs(μn - μ)
            for k in 1:N_τ; err = max(err, abs(Qn[k] - Q[k]), abs(Rn[k] - R[k])); end
            Q .= Qn; R .= Rn; μ = μn
            err < tol && break
        end
        iΣ = 0.0
        for k in 1:N_τ; iΣ += tw(k, 1, N_τ) * EXP[k] * (c_Q * Q[k] + c_R * R[k]); end
        return (; F = ν / γ - μ + Δτ * iΣ, Q, R, μ)
    end
    F_vals = zeros(length(s_scan))
    Qw = nothing; Rw = nothing; μw = NaN
    for (i, s) in enumerate(s_scan)
        r = solve_bath(s; Q0=Qw, R0=Rw, μ0=μw)
        F_vals[i] = r.F
        Qw = copy(r.Q); Rw = copy(r.R); μw = r.μ
    end
    # If the warm-chained forward scan missed a sign change, redo it with
    # cold-starts at each s — warm-state drift can push F negative at small
    # γ and make the solver report spurious PM.
    has_cross = any(i -> F_vals[i] > 0 && F_vals[i+1] < 0, 1:length(s_scan)-1)
    deep_fm   = all(F_vals .> 0)
    if !has_cross && !deep_fm
        F_cold = zeros(length(s_scan))
        for (i, s) in enumerate(s_scan)
            r = solve_bath(s)
            F_cold[i] = r.F
        end
        has_cross_cold = any(i -> F_cold[i] > 0 && F_cold[i+1] < 0, 1:length(s_scan)-1)
        if has_cross_cold || all(F_cold .> 0)
            F_vals = F_cold
            Qw = nothing; Rw = nothing; μw = NaN
        end
    end
    # Locate every + → − transition and take the rightmost (outermost FM
    # fixed point).  Sweeping left to right means inner unstable roots are
    # skipped in favour of the stable outer branch.  Fallback: if F stays
    # positive all the way to s_scan[end], the system is deep FM and the
    # root sits past the scan grid — report s ≈ s_max.
    s_inf = 0.0
    crossings = Tuple{Float64,Float64}[]
    for i in 1:length(s_scan)-1
        if F_vals[i] > 0 && F_vals[i+1] < 0
            push!(crossings, (Float64(s_scan[i]), Float64(s_scan[i+1])))
        end
    end
    if !isempty(crossings)
        slo, shi = crossings[end]
        for _ in 1:40
            sm = (slo + shi) / 2
            rm = solve_bath(sm; Q0=Qw, R0=Rw, μ0=μw)
            Qw = copy(rm.Q); Rw = copy(rm.R); μw = rm.μ
            rm.F > 0 ? (slo = sm) : (shi = sm)
            (shi - slo) < 1e-8 && break
        end
        s_inf = (slo + shi) / 2
    elseif all(F_vals .> 0)
        s_inf = Float64(s_scan[end])
    end
    # γ-continuation fallback: if the scan said PM but the previous γ had an
    # FM root, the real root may sit in a narrow band around s_prev that the
    # scan grid stepped past.  Rebracket by cold-probing around s_prev; the
    # root should still lie below s_prev as γ moves toward the boundary.
    if s_inf == 0.0 && s_prev > 0.01
        s_hi = min(0.998, s_prev * 1.15 + 0.05)
        s_lo_trial = max(1e-4, s_prev * 0.2)
        r_hi = solve_bath(s_hi); F_hi = r_hi.F
        r_lo = solve_bath(s_lo_trial); F_lo = r_lo.F
        # Walk s_hi up if needed (F should be negative above the FM root)
        tries = 0
        while F_hi > 0 && s_hi < 0.998 && tries < 6
            s_hi = min(0.998, s_hi + 0.1); r_hi = solve_bath(s_hi); F_hi = r_hi.F; tries += 1
        end
        # Walk s_lo down if needed (F should be positive below the FM root,
        # down to the PM minimum — at the boundary F(0) → 0⁻ so a very small
        # s_lo may cross into negative territory, hence we step carefully)
        tries = 0
        while F_lo < 0 && s_lo_trial > 1e-5 && tries < 8
            s_lo_trial *= 0.5; r_lo = solve_bath(s_lo_trial); F_lo = r_lo.F; tries += 1
        end
        if F_lo > 0 && F_hi < 0
            slo, shi = s_lo_trial, s_hi
            Qw2 = copy(r_hi.Q); Rw2 = copy(r_hi.R); μw2 = r_hi.μ
            for _ in 1:50
                sm = (slo + shi) / 2
                rm = solve_bath(sm; Q0=Qw2, R0=Rw2, μ0=μw2)
                Qw2 = copy(rm.Q); Rw2 = copy(rm.R); μw2 = rm.μ
                rm.F > 0 ? (slo = sm) : (shi = sm)
                (shi - slo) < 1e-8 && break
            end
            s_inf = (slo + shi) / 2
        end
    end
    return s_inf
end

# ═══════════════════════════════════════════════════════════════════════
# C. Dynamical MSR solver — time-marching, K=1, c=1, full-matrix (small N_t)
# ═══════════════════════════════════════════════════════════════════════
@inline _tw(k, lo, hi) = lo >= hi ? 0.0 : (k == lo || k == hi) ? 0.5 : 1.0

function run_msr(γ, η, β, ν; c=1.0, s0=0.05, Tmax=40.0, Δt=-1.0, dt_floor=0.02)
    if Δt < 0
        Δt = clamp(sqrt(η * γ) / (5 * max(ν, 0.01)), dt_floor, 0.2)
    end
    N_t = round(Int, Tmax / Δt) + 1
    c_Q = -ν / 2; c_R = ν^2 / (η * γ)
    θ_arr = [(1 - exp(-γ * (i-1) * Δt / 2)) / γ for i in 1:N_t]
    ef = [exp(-γ * d * Δt / 2) for d in 0:N_t-1]
    Q = zeros(N_t, N_t); R = zeros(N_t, N_t)
    Q[1,1] = 1.0; R[1,1] = 1.0
    s = zeros(N_t); s[1] = s0
    κ = zeros(N_t)
    ΣR = zeros(N_t)
    for ni in 2:N_t
        np = ni - 1
        sum_cs2 = c * s[np]^2
        int_κ = 0.0
        @inbounds for ki in 1:np
            Q_np_ki = Q[np, ki]; R_np_ki = R[np, ki]
            M_ki = (c_Q * Q_np_ki + c_R * R_np_ki) * ef[np - ki + 1]
            ΣR[ki] = M_ki
            Dreg_ki = c_R * ef[np - ki + 1] * Q_np_ki
            w = _tw(ki, 1, np)
            int_κ += w * (M_ki * Q_np_ki + Dreg_ki * R_np_ki)
        end
        κ_np = ν * θ_arr[np] * sum_cs2 + ν / β + Δt * int_κ
        κ[np] = κ_np
        eκ = exp(-κ_np * Δt); φκ = φ₁(-κ_np * Δt)
        R[ni, ni] = 1.0
        @inbounds for mi in 1:np
            sum_MR = 0.0
            for ki in mi:np
                w = _tw(ki, mi, np)
                sum_MR += w * ΣR[ki] * R[ki, mi]
            end
            R[ni, mi] = eκ * R[np, mi] + φκ * Δt * (Δt * sum_MR)
        end
        α = ν * c * θ_arr[np] - κ_np
        g = 0.0
        for ki in 1:np
            w = _tw(ki, 1, np); g += w * ΣR[ki] * s[ki]
        end
        g *= Δt
        s[ni] = exp(α * Δt) * s[np] + φ₁(α * Δt) * Δt * g
        Q[ni, ni] = 1.0
        @inbounds for mi in 1:np
            sum_MQ = 0.0
            for ki in 1:np
                w = _tw(ki, 1, np)
                q_ki_mi = ki >= mi ? Q[ki, mi] : Q[mi, ki]
                sum_MQ += w * ΣR[ki] * q_ki_mi
            end
            ki_lo_d = 1; sum_DR = 0.0
            for ki in ki_lo_d:mi
                Dreg_ki = c_R * ef[np - ki + 1] * Q[np, ki]
                w = _tw(ki, ki_lo_d, mi)
                sum_DR += w * Dreg_ki * R[mi, ki]
            end
            source = ν * θ_arr[np] * c * s[np] * s[mi]
            Q[ni, mi] = eκ * Q[np, mi] + φκ * Δt * (source + Δt * sum_MQ + Δt * sum_DR)
        end
    end
    κ[N_t] = κ[N_t-1]
    return (; t = collect(0:N_t-1) .* Δt, s, κ)
end

# ═══════════════════════════════════════════════════════════════════════
# D. Assemble data
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Row 1: phase diagrams ν_c(γ) ===")
datadir = joinpath(@__DIR__, "data"); mkpath(datadir)
phase_cache = joinpath(datadir, "20260417_fig12_phase.jld2")
phase = Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}()
if isfile(phase_cache)
    c = load(phase_cache)
    for η in η_list
        k_g = "eta_$(η)_g"; k_n = "eta_$(η)_nu"
        if haskey(c, k_g) && haskey(c, k_n)
            phase[η] = (Vector{Float64}(c[k_g]), Vector{Float64}(c[k_n]))
            println("  cached  η=$η  ($(length(phase[η][1])) pts)")
        end
    end
end
for η in η_list
    haskey(phase, η) && continue
    println(@sprintf("\nη = %g", η))
    t0 = time()
    g, v = trace_eta(η; time_budget=180.0)
    phase[η] = (g, v)
    println(@sprintf("  %d pts  γ∈[%.3f,%.3f]  ν_c∈[%.3g,%.3g]  (%.1fs)",
                     length(g), isempty(g) ? NaN : minimum(g), isempty(g) ? NaN : maximum(g),
                     isempty(v) ? NaN : minimum(v), isempty(v) ? NaN : maximum(v), time()-t0))
    # Incremental save — JLDFile direct write (avoids keyword-splat type issues)
    jldopen(phase_cache, "w") do f
        for ηk in η_list
            if haskey(phase, ηk)
                f["eta_$(ηk)_g"]  = phase[ηk][1]
                f["eta_$(ηk)_nu"] = phase[ηk][2]
            end
        end
    end
end

println("\n=== Choose ν sweep values ===")
# Row 2 uses a fixed ν pair (0.1, 100): a decade deep in PM (blue) and two
# decades above the boundary (red), solved on the stationary branch.  Rows 3–4
# use six log-spaced integer values ν ∈ {1, 3, 6, 16, 40, 100} paired per η
# column so that (ν_lo, ν_hi) brackets each column's traced ν_c range.
const ν_LO_ROW2 = 0.1
const ν_HI_ROW2 = 100.0
const ν_row2 = (ν_LO_ROW2, ν_HI_ROW2)
ν_sweep = Dict{Float64,Tuple{Float64,Float64}}(
    1.0  => (1.0,  16.0),
    3.0  => (3.0,  40.0),
    10.0 => (6.0, 100.0),
)
for η in η_list
    println(@sprintf("  η=%g:  ν_row2 = %.3g,%.3g   ν_rows34 = %d,%d",
                     η, ν_row2..., Int(ν_sweep[η][1]), Int(ν_sweep[η][2])))
end

println("\n=== Dynamics families at fixed ν, sweeping γ across the transition ===")
# For each η and each of the two ν values, run MSR dynamics at a handful of
# γ values bracketing the FM/PM transition so the family shows some curves
# relaxing to zero (PM) and others settling to finite s (FM).  The γ grid
# is η-tuned around the transition γ implied by row-2 data.
γ_families = Dict(
    1.0  => [0.30, 0.50, 0.65, 0.72, 0.80, 0.92],
    3.0  => [0.30, 0.55, 0.75, 0.85, 0.90, 0.96],
    10.0 => [0.30, 0.60, 0.80, 0.88, 0.93, 0.97],
)
dyn_cache_path = joinpath(datadir, "20260417_fig12_dyn.jld2")
dyn_lowν  = Dict{Float64,Vector{Any}}()
dyn_highν = Dict{Float64,Vector{Any}}()
if isfile(dyn_cache_path)
    c = load(dyn_cache_path)
    for η in η_list
        klo = "eta_$(η)_lowν"; khi = "eta_$(η)_highν"
        kν1 = "eta_$(η)_ν1"; kν2 = "eta_$(η)_ν2"; kγs = "eta_$(η)_γs"
        if haskey(c, klo) && haskey(c, khi) &&
           haskey(c, kν1) && haskey(c, kν2) && haskey(c, kγs) &&
           isapprox(c[kν1], ν_sweep[η][1]; rtol=1e-6) &&
           isapprox(c[kν2], ν_sweep[η][2]; rtol=1e-6) &&
           Vector{Float64}(c[kγs]) == γ_families[η]
            dyn_lowν[η]  = c[klo]
            dyn_highν[η] = c[khi]
            println(@sprintf("  cached  η=%g  (%d γ × 2 ν)", η, length(γ_families[η])))
        end
    end
end
for η in η_list
    (haskey(dyn_lowν, η) && haskey(dyn_highν, η)) && continue
    ν1, ν2 = ν_sweep[η]
    γs = γ_families[η]
    dyn_lowν[η]  = Vector{Any}(undef, length(γs))
    dyn_highν[η] = Vector{Any}(undef, length(γs))
    Tmax_dyn = 10.0
    dtf_lo = min(0.02, 0.5 / max(ν1, 1.0))
    dtf_hi = min(0.02, 0.5 / max(ν2, 1.0))
    t0 = time()
    @threads for k in eachindex(γs)
        γ = γs[k]
        dlo = run_msr(γ, η, β_fixed, ν1; s0=0.1, Tmax=Tmax_dyn, dt_floor=dtf_lo)
        dhi = run_msr(γ, η, β_fixed, ν2; s0=0.1, Tmax=Tmax_dyn, dt_floor=dtf_hi)
        dyn_lowν[η][k]  = (; γ=γ, ν=ν1, t=dlo.t, s=dlo.s)
        dyn_highν[η][k] = (; γ=γ, ν=ν2, t=dhi.t, s=dhi.s)
    end
    println(@sprintf("  η=%g  done  (%.1fs, %d γ trajectories × 2 ν)",
                     η, time()-t0, length(γs))); flush(stdout)
    # Incremental save after each η
    jldopen(dyn_cache_path, "w") do f
        for ηk in η_list
            haskey(dyn_lowν, ηk) || continue
            f["eta_$(ηk)_lowν"]  = dyn_lowν[ηk]
            f["eta_$(ηk)_highν"] = dyn_highν[ηk]
            f["eta_$(ηk)_ν1"]    = ν_sweep[ηk][1]
            f["eta_$(ηk)_ν2"]    = ν_sweep[ηk][2]
            f["eta_$(ηk)_γs"]    = γ_families[ηk]
        end
    end
end

println("\n=== Row 2: stationary s_∞(γ) at ν=$(ν_LO_ROW2), $(ν_HI_ROW2) ===")

stat_cache = joinpath(datadir, "20260417_fig12_stat.jld2")
s_vs_γ = Dict{Float64,Any}()
if isfile(stat_cache)
    c = load(stat_cache)
    # Only reuse cache if its stored ν pair matches the current fixed pair.
    for η in η_list
        k = "eta_$η"
        if haskey(c, "$(k)_γ") && haskey(c, "$(k)_s1") && haskey(c, "$(k)_s2") &&
           haskey(c, "$(k)_ν1") && haskey(c, "$(k)_ν2") &&
           isapprox(c["$(k)_ν1"], ν_LO_ROW2; rtol=1e-6) &&
           isapprox(c["$(k)_ν2"], ν_HI_ROW2; rtol=1e-6)
            s_vs_γ[η] = (; γ=Vector{Float64}(c["$(k)_γ"]),
                           ν1=c["$(k)_ν1"], s1=Vector{Float64}(c["$(k)_s1"]),
                           ν2=c["$(k)_ν2"], s2=Vector{Float64}(c["$(k)_s2"]))
            println(@sprintf("  cached  η=%g  (%d pts)", η, length(s_vs_γ[η].γ)))
        end
    end
end
println("  threads available: $(nthreads())")
for η in η_list
    haskey(s_vs_γ, η) && continue
    ν1, ν2 = ν_row2
    g_grid = collect(range(0.05, 1.3, length=20))
    s1 = zeros(length(g_grid)); s2 = zeros(length(g_grid))
    t0 = time()
    # Stationary branch-follower: solves the TTI fixed-point equations for
    # s_∞ directly, dodging the O(N_t^3) cost of time-marching at ν=100.
    # Returns s_∞=0 in PM and the outer FM root otherwise.
    @threads for i in eachindex(g_grid)
        γ = g_grid[i]
        s1[i] = solve_stationary(γ, η, β_fixed, ν1)
        s2[i] = solve_stationary(γ, η, β_fixed, ν2)
    end
    println(@sprintf("  η=%g  done  (%.1fs, %d γ-points)", η, time()-t0, length(g_grid))); flush(stdout)
    s_vs_γ[η] = (; γ = g_grid, ν1, s1, ν2, s2)
    # Incremental save
    jldopen(stat_cache, "w") do f
        for ηk in η_list
            if haskey(s_vs_γ, ηk)
                d = s_vs_γ[ηk]
                write(f, "eta_$(ηk)_γ",  d.γ)
                write(f, "eta_$(ηk)_ν1", d.ν1)
                write(f, "eta_$(ηk)_s1", d.s1)
                write(f, "eta_$(ηk)_ν2", d.ν2)
                write(f, "eta_$(ηk)_s2", d.s2)
            end
        end
    end
end

jldsave(joinpath(datadir, "20260417_fig12_final.jld2");
        η_list = η_list, β = β_fixed,
        phase_γ    = Dict(η => phase[η][1] for η in η_list),
        phase_ν    = Dict(η => phase[η][2] for η in η_list),
        dyn_lowν   = dyn_lowν,
        dyn_highν  = dyn_highν,
        s_vs_γ     = s_vs_γ)

# ═══════════════════════════════════════════════════════════════════════
# E. Figure
# ═══════════════════════════════════════════════════════════════════════
set_theme!(Theme(fontsize=9,
                 Axis=(xlabelsize=10, ylabelsize=10, titlesize=10,
                       xticklabelsize=8, yticklabelsize=8)))

ncols = length(η_list); nrows = 4
fig = Figure(size=(220 * ncols + 40, 160 * nrows + 40))

# Row 1: phase diagrams ν_c(γ) — prefer refined boundary if available
refined_path = joinpath(datadir, "20260417_fig12_phase_refined.jld2")
refined = isfile(refined_path) ? load(refined_path) : nothing
for (col, η) in enumerate(η_list)
    ax = Axis(fig[1, col];
              xlabel = L"\gamma", ylabel = col == 1 ? L"\nu" : "",
              title = L"\eta = %$(Int(η))",
              yscale = log10,
              xticklabelsvisible = false,
              xlabelvisible = false,
              yticklabelsvisible = col == 1)
    # Pull refined boundary if we have it (with asymptotic fit exponents);
    # otherwise fall back to the coarse cache, filtered to γ < β (points at
    # γ ≥ β are the ν_c ∝ 1/Δτ artifact; see project_map_phase_diagram).
    A_fit = NaN; p_fit = NaN
    if refined !== nothing && haskey(refined, "eta_$(η)_g")
        γp = Vector{Float64}(refined["eta_$(η)_g"])
        νp = Vector{Float64}(refined["eta_$(η)_nu"])
        A_fit = refined["eta_$(η)_A"]
        p_fit = refined["eta_$(η)_p"]
    else
        γp, νp = phase[η]
        keep = γp .< β_fixed
        γp = γp[keep]; νp = νp[keep]
        # Inline log–log fit ν_c ≈ A·(β−γ)^(−p) from the top traced points
        # so the phase-diagram shading can extend all the way to γ=β without
        # depending on an external refined-cache file.
        if length(γp) ≥ 4
            n_fit = min(5, length(γp))
            γ_fit = γp[end-n_fit+1:end]
            ν_fit = νp[end-n_fit+1:end]
            x = log.(β_fixed .- γ_fit); y = log.(ν_fit)
            m = length(x); sx = sum(x); sy = sum(y)
            sxx = sum(x .* x); sxy = sum(x .* y)
            denom = m * sxx - sx * sx
            if denom > 0
                slope = (m * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / m
                if slope < 0
                    p_fit = -slope
                    A_fit = exp(intercept)
                end
            end
        end
    end
    # Strip any non-monotone outliers: ν_c(γ) must be strictly increasing
    # on the physical branch.  (Warm-start can wander onto a spurious
    # branch when the bracket narrows too fast near the guard limit; drop
    # those points rather than re-running the expensive sweep.)
    if length(γp) ≥ 2
        keep = trues(length(γp))
        ν_run = νp[1]
        for i in 2:length(γp)
            if νp[i] < ν_run * 0.98
                keep[i] = false
            else
                ν_run = max(ν_run, νp[i])
            end
        end
        γp = γp[keep]; νp = νp[keep]
    end
    if !isempty(γp)
        γ_plot_lo, γ_plot_hi = 0.0, 1.3
        ν_plot_lo, ν_plot_hi = 1e-2, 1e3
        γmin, γmax = minimum(γp), maximum(γp)
        νmax = maximum(νp)
        # Extend the ν_c path up to ν_plot_hi using the asymptotic fit
        # ν_c ≈ A·(β−γ)^(−p) if available; otherwise fall back to a vertical
        # continuation at γ=γmax.  Either way the band edge now hugs γ=β at
        # the top of the panel instead of cutting diagonally to (β, ν_hi).
        if isfinite(A_fit) && isfinite(p_fit) && p_fit > 0 && νmax < ν_plot_hi
            ν_tail = exp.(range(log(νmax), log(ν_plot_hi); length=40))
            γ_tail = β_fixed .- (A_fit ./ ν_tail) .^ (1.0 / p_fit)
            # Drop first point (duplicate of last trace point) and clamp γ<β.
            γ_tail = min.(γ_tail[2:end], β_fixed - 1e-6)
            ν_tail = ν_tail[2:end]
            γ_ext = vcat(γp, γ_tail)
            ν_ext = vcat(νp, ν_tail)
        else
            γ_ext = vcat(γp, [γmax, γmax])
            ν_ext = vcat(νp, [νmax, ν_plot_hi])
        end
        # Below curve in γ∈[γ_min, β]: PM
        band!(ax, γ_ext, fill(ν_plot_lo, length(γ_ext)), ν_ext; color=(:steelblue, 0.25))
        # Above curve in γ∈[γ_min, β]: FM
        band!(ax, γ_ext, ν_ext, fill(ν_plot_hi, length(γ_ext)); color=(:firebrick, 0.25))
        # γ < γ_min: full column FM
        γL = collect(range(γ_plot_lo, γmin; length=10))
        band!(ax, γL, fill(ν_plot_lo, 10), fill(ν_plot_hi, 10); color=(:firebrick, 0.25))
        # γ > β: full column PM
        γR = collect(range(β_fixed, γ_plot_hi; length=10))
        band!(ax, γR, fill(ν_plot_lo, 10), fill(ν_plot_hi, 10); color=(:steelblue, 0.25))
        # Solid curve: full boundary including the asymptotic tail, so the
        # black line reaches the top of the FM/PM shading at γ=β.  Prepend a
        # vertical drop at γ=γ_min from the bottom of the panel up to the
        # first traced point, so the boundary covers its full extent.
        γ_line = vcat([γmin], γ_ext)
        ν_line = vcat([ν_plot_lo], ν_ext)
        lines!(ax, γ_line, ν_line; color=:black, linewidth=1.8)
        vlines!(ax, [β_fixed]; color=:black, linestyle=:dash, linewidth=1.0)
        # Horizontal lines at the two ν values swept in row 4
        ν1 = s_vs_γ[η].ν1; ν2 = s_vs_γ[η].ν2
        hlines!(ax, [ν1]; color=(:steelblue, 0.85), linestyle=:dash, linewidth=1.0)
        hlines!(ax, [ν2]; color=(:firebrick, 0.85), linestyle=:dash, linewidth=1.0)
        xlims!(ax, γ_plot_lo, γ_plot_hi); ylims!(ax, ν_plot_lo, ν_plot_hi)
    end
end

# Top-row legend placed outside the panels, to the right of column 3.
let
    pm_patch = PolyElement(color=(:steelblue, 0.35), strokewidth=0)
    fm_patch = PolyElement(color=(:firebrick, 0.35), strokewidth=0)
    Legend(fig[1, length(η_list) + 1],
           [fm_patch, pm_patch],
           [L"s_{\mathrm{st}} \neq 0", L"s_{\mathrm{st}} = 0"];
           labelsize=8, framevisible=false,
           patchsize=(10, 8), rowgap=2,
           tellwidth=true, tellheight=false)
end

# Row 2: time-marched s_∞(γ) at two fixed ν values — shares γ axis with row 1
for (col, η) in enumerate(η_list)
    d = s_vs_γ[η]
    # NaN-safe max for plot limits (some stiff (γ,ν) points may have
    # produced a NaN trajectory and been masked in the sweep).
    finmax(x) = (f = filter(isfinite, x); isempty(f) ? 0.0 : maximum(f))
    ax = Axis(fig[2, col];
              xlabel = L"\gamma", ylabel = col == 1 ? L"s_\infty(\gamma)" : "",
              yticklabelsvisible = col == 1)
    ν1_lab = d.ν1 == round(d.ν1) ? string(Int(d.ν1)) : string(round(d.ν1; sigdigits=2))
    ν2_lab = d.ν2 == round(d.ν2) ? string(Int(d.ν2)) : string(round(d.ν2; sigdigits=2))
    lines!(ax, d.γ, d.s1; color=:steelblue, linewidth=1.6,
           label=L"\nu = %$(ν1_lab)")
    lines!(ax, d.γ, d.s2; color=:firebrick, linewidth=1.6,
           label=L"\nu = %$(ν2_lab)")
    # Markers at the actually-simulated γ grid points; lines are interpolation.
    scatter!(ax, d.γ, d.s1; color=:steelblue, marker=:circle, markersize=5,
             strokewidth=0.6, strokecolor=:black)
    scatter!(ax, d.γ, d.s2; color=:firebrick, marker=:circle, markersize=5,
             strokewidth=0.6, strokecolor=:black)
    vlines!(ax, [β_fixed]; color=:black, linestyle=:dash, linewidth=1.0)
    xlims!(ax, 0, 1.3)
    smax = max(finmax(d.s1), finmax(d.s2), 0.05)
    ylims!(ax, -0.02 * smax - 0.005, 1.15 * smax + 0.005)
    axislegend(ax; position=:rt, labelsize=8, framevisible=false,
               patchsize=(10, 1.5), rowgap=3)
end

# Rows 3 and 4 show families of s(t) at fixed ν, sweeping γ across the
# transition so some trajectories decay to zero (PM) and others condense to
# finite s (FM).  Row 3 uses the lower ν of the pair; row 4 the higher ν.
# γ values colored by a viridis gradient (low-γ dark, high-γ light).

function plot_dyn_family!(ax, family; ylabel_text="")
    hlines!(ax, [0.0]; color=(:gray, 0.5), linestyle=:dash, linewidth=0.8)
    n = length(family)
    cmap = cgrad(:viridis, n, categorical=true)
    smax_seen = 0.0; tmax_seen = 0.0
    for (k, d) in enumerate(family)
        all(isfinite, d.s) || continue
        lines!(ax, d.t, d.s; color=cmap[k], linewidth=1.4,
               label=L"\gamma = %$(round(d.γ; digits=2))")
        smax_seen = max(smax_seen, maximum(abs, d.s))
        tmax_seen = max(tmax_seen, maximum(d.t))
    end
    xlims!(ax, 0, max(tmax_seen, 1.0))
    smax_seen = max(smax_seen, 0.05)
    ylims!(ax, -0.02 * smax_seen - 0.005, 1.15 * smax_seen + 0.005)
end

# Row 3: fixed ν = ν_low, γ sweeps transition
for (col, η) in enumerate(η_list)
    ν1 = ν_sweep[η][1]
    ax = Axis(fig[3, col];
              xlabel = L"t", ylabel = col == 1 ? L"s(t)" : "",
              title  = L"(\nu,\eta) = (%$(Int(ν1)),\ %$(Int(η)))", titlesize=9,
              yticklabelsvisible = col == 1)
    plot_dyn_family!(ax, dyn_lowν[η])
    if col == length(η_list)
        axislegend(ax; position=:rc, labelsize=7, framevisible=true,
                   backgroundcolor=(:white, 0.85),
                   patchsize=(10, 1.5), rowgap=2, nbanks=1)
    end
end

# Row 4: fixed ν = ν_high, γ sweeps transition
for (col, η) in enumerate(η_list)
    ν2 = ν_sweep[η][2]
    ax = Axis(fig[4, col];
              xlabel = L"t", ylabel = col == 1 ? L"s(t)" : "",
              title  = L"(\nu,\eta) = (%$(Int(ν2)),\ %$(Int(η)))", titlesize=9,
              yticklabelsvisible = col == 1)
    plot_dyn_family!(ax, dyn_highν[η])
    if col == length(η_list)
        axislegend(ax; position=:rc, labelsize=7, framevisible=true,
                   backgroundcolor=(:white, 0.85),
                   patchsize=(10, 1.5), rowgap=2, nbanks=1)
    end
end

colgap!(fig.layout, 8); rowgap!(fig.layout, 6)

figdir = @__DIR__
pdf_path = joinpath(figdir, "fig_finite_eta_phase_diagrams.pdf")
png_path = joinpath(figdir, "fig_finite_eta_phase_diagrams.png")
save(pdf_path, fig); save(png_path, fig, px_per_unit=3)
println("\nSaved: $pdf_path")
println("Saved: $png_path")

# (Removed paper-directory publish loop — the standalone accompanying-code
# release writes only into this script's directory.)
for target in String[]
    if !isdir(dirname(target))
        println("Skipping (parent dir missing): $target")
        continue
    end
    try
        cp(pdf_path, target; force=true)
        println("Copied → $target")
    catch err
        println("Copy to $target failed: $err")
    end
end
println("Done.")
