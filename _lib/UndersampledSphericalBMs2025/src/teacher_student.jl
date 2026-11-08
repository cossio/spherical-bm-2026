# Teacher-student KL divergences for K=2 spherical Boltzmann machines.
#
# Teacher-student formulas from Appendix D of the accompanying manuscript.
# Uses RMT_Solution_v2.solve(c_vec, γ, η) for phase determination.
#
# Convention: all KL divergences are per N at leading order.
# The constant A = ln(2π)/2 from partition function normalizations
# cancels in all KL differences, so we work "mod A".
# The manuscript retains A in its full evidence Φ. The exact conversion is
# documented below in Phi1_data; the KL functions return fully normalized KLs.

module TeacherStudent

import ..RMT_Solution_v2 as RMT
import ..RMT_Solution

using LinearAlgebra: eigvals

# ── Semicircle helpers (parameter c = γη) ─────────────────────────────

G_sc(z, c) = RMT_Solution.stieltjes(z, c)
G_sc_inv(a, c) = RMT_Solution.stieltjes_inverse(a, c)
F_sc(z, c) = RMT_Solution.log_potential(z, c)

"Saddle action S_sc(z) = z - g²/(2c) + ln(g), g = G_sc(z)"
function S_sc(z, c)
    g = G_sc(z, c)
    return z - g^2 / (2c) + log(g)
end

# ── Teacher quantities (rank-1 teacher, ω* > 1) ──────────────────────

"Data eigenvalues for K=2: c₁ = 2 - 1/ω*, c₂ = 1/ω*"
teacher_c(ω) = (2 - 1/ω, 1/ω)

# All "noA" quantities drop the common constant A = ln(2π)/2.

"H[P_{W*}]/N - A = (1 - ln ω)/2"
teacher_entropy_noA(ω) = (1 - log(ω)) / 2

"ln Z(W*)/N - A = (ω - ln ω)/2"
teacher_logZ_noA(ω) = (ω - log(ω)) / 2

# ── Posterior saddle-point quantities ─────────────────────────────────

"""
    posterior_quantities(γ, η, ω) -> NamedTuple

Compute saddle-point quantities for the posterior with teacher signal ω.
"""
function posterior_quantities(γ::Real, η::Real, ω::Real)
    c1, c2 = teacher_c(ω)
    c_rmt = γ * η
    sol = RMT.solve([c1, c2], γ, η)

    # Z-saddle point: g = G(μ) at the partition function saddle
    # paramagnetic: g = 1, μ = G⁻¹(1)
    # ferromagnetic: g = g₁, μ = λ₁
    if sol.M ≈ 0
        g_Z = 1.0
        μ_Z = G_sc_inv(1.0, c_rmt)
    else
        g_Z = sol.g[1]
        μ_Z = sol.μ
    end

    F1 = S_sc(μ_Z, c_rmt) / 2

    return (; sol, c_rmt, c1, c2, g_Z, μ_Z, F1)
end

# ── Student entropy (averaged over posterior) ─────────────────────────

"""
Student entropy ⟨H[P_W]⟩/N - A = (1 - g²/(2c) + ln g) / 2

where g is the Stieltjes value at the Z-saddle (g=1 param, g=g₁ ferro)
and c = γη.
"""
function student_entropy_noA(g_Z, c_rmt)
    return (1 - g_Z^2 / (2c_rmt) + log(g_Z)) / 2
end

# ── Average energies ──────────────────────────────────────────────────

"""
⟨⟨E(x;W)⟩_{x~P_{W*}}⟩_{W~P_η} / N

Energy of teacher data under a typical posterior W.
= -(ξ₁-1)(ω-1)² / [η(2ω-1)²]
"""
function avg_energy_teacher_in_student(ξ1, η, ω)
    return -(ξ1 - 1) * (ω - 1)^2 / (η * (2ω - 1)^2)
end

"""
⟨E(x;W*)⟩_{P_pp} / N

Energy of predictive-posterior data under teacher W*.
= (1/2)[γ/η · ω²/(2ω-1)² · (ξ₁-1) - 1](ω-1)

In paramagnetic phases this equals 0 (the factor in brackets is 1-1=0).
"""
function avg_energy_pp_in_teacher(ξ1, γ, η, ω)
    return 0.5 * (γ / η * ω^2 / (2ω - 1)^2 * (ξ1 - 1) - 1) * (ω - 1)
end

# ── Φ₁^data computation ──────────────────────────────────────────────

"""
    Phi1_data(c_vec, γ, η) -> Float64

Compute Φ₁^data = Σ h_k - τη F₁ for eigenvalues `c_vec`.
R = length(c_vec), τ = sum(c_vec).

The manuscript's full evidence coefficient is
    Φ(C) = Phi1_data(c_vec, γ, η) + (R/2)log(γη) - τη A,
where A = log(2π)/2. For the rank-two teacher and rank-three predictive
source, the trace increases by 1/η, so
    ΔΦ = ΔPhi1_data + log(γη)/2 - A.
The final -A cancels the teacher log-partition constant in reverse KL,
and cancels the teacher entropy constant in forward KL. Thus the reduced
functions below compute the same KL values as the full manuscript formulas.
"""
function Phi1_data(c_vec::AbstractVector{<:Real}, γ::Real, η::Real)
    sol = RMT.solve(c_vec, γ, η)
    return _Phi1_data(c_vec, γ, η, sol)
end

function _Phi1_data(c_vec, γ, η, sol)
    c_rmt = γ * η
    τ = sum(c_vec)

    # F₁ term
    if sol.M ≈ 0  # paramagnetic / spin glass
        μ = G_sc_inv(1.0, c_rmt)
    else  # ferromagnetic
        μ = sol.μ
    end
    F1 = S_sc(μ, c_rmt) / 2

    # Per-mode h_k
    # Use ferromagnetic formula only when mode k is coalesced AND has u_k > 0.
    # In spin glass: d > 0 but all u_k = 0, so paramagnetic h_k must be used.
    h_total = 0.0
    for k in eachindex(c_vec)
        if k ≤ sol.d && sol.u[k] > 0
            # Ferromagnetic coalesced: h_k = -1/2 - γη/(4g₁²) + c_k g₁/(2γ) + η c_k/(2g₁) - ln(g₁ η c_k)/2
            g1 = sol.g[1]
            ck = c_vec[k]
            h_total += -0.5 - c_rmt / (4g1^2) + ck * g1 / (2γ) + η * ck / (2g1) - log(g1 * η * ck) / 2
        else
            # Paramagnetic: h_k = η c_k²/(4γ) - ln(γη)/2
            ck = c_vec[k]
            h_total += η * ck^2 / (4γ) - log(c_rmt) / 2
        end
    end

    return h_total - τ * η * F1
end

# ── β-generalized evidence functional Ψ_β ────────────────────────────
#
# Per `notes/temperature_tuning_kls_beta_note.md`, the β=1 identity
# P_pp(x) = Y(tilde_C_x) / Y(tilde_C) is replaced at β ≠ 1 by a
# generalized evidence Y^β(C) with separate log-Z factors for training
# (K_train·η·ln Z(W)) and sampling (1·ln Z(β W)). The saddle-point
# integration gives a new functional
#
#     Ψ_β(D) = extr_{λ,χ} [ (r/2) log(γη)
#                           − K_train·η · z_1(λ_max)
#                           − z_β(λ_max)
#                           + Σ_j { −γη/4 λ_j² + F(λ_j)
#                                    + (η d_j χ_j − F(χ_j) − log(η d_j))/2 } ]
#
# where z_b(λ_max) = (b·a_b − log b − F(a_b))/2 with a_b = G⁻¹(b) when
# b ≤ G(λ_max) (sampling PM) and a_b = λ_max when b > G(λ_max)
# (sampling pinning). At β=1, both training (b=1) and sampling (b=β=1)
# collapse onto a single pinning condition and Ψ_1(D) = Φ_1(D) up to a
# constant offset (r/2)·log(γη) that cancels in the KL differences
# Ψ_β(D) − Φ_1(c_1, c_2).

"z_b(λ_max) from the note: log Z of coupling with outlier at λ_max,
semicircle bulk, evaluated at sampling inverse temperature b."
function z_b_at_lambda(λ_max::Real, b::Real, γ::Real, η::Real)
    c_rmt = γ * η
    g_top = G_sc(λ_max, c_rmt)
    if b ≤ g_top
        # Sampling-paramagnetic: a_b = G⁻¹(b) = 1/b + b/(γη)
        a_b = G_sc_inv(b, c_rmt)
    else
        # Sampling-pins: a_b = λ_max
        a_b = λ_max
    end
    g_of_ab = G_sc(a_b, c_rmt)
    F_ab = g_of_ab^2 / (2 * c_rmt) - log(g_of_ab)   # F(a_b) = σ²g²/2 − log g
    return (b * a_b - log(b) - F_ab) / 2
end

"""
    psi_beta(D, β, γ, η; K_train=2) -> Float64

β-generalized evidence functional Ψ_β(D) for finite-rank source D,
sampled at inverse temperature β with training at β=1 and K_train data
points. At β=1, returns Φ_1(D) + (length(D)/2) log(γη).

Finds the smallest d ∈ 0:r for which the β-modified saddle has a
self-consistent top eigenvalue λ_max, with g_1 = G(λ_max) satisfying:
    (Kη·[train pin] + 1·[samp pin])·g₁²
  + g₁·(η·σ_cond − Kη·[train pin] − β·[samp pin])
  − d·γη = 0
where σ_cond = Σ (condensed d_j's), and each "[pin]" indicator is
whether the corresponding saddle pins (g₁ < 1 for training, g₁ < β for
sampling).
"""
function psi_beta(D::AbstractVector{<:Real}, β::Real, γ::Real, η::Real; K_train::Int=2)
    @assert issorted(D; rev=true) "D must be sorted descending"
    r = length(D)
    c_rmt = γ * η
    edge_g = sqrt(c_rmt)        # G at the bulk edge; g_1 ≤ edge_g in any condensed regime

    # Evaluate Ψ_β at a given saddle point (d_try, g_1, λ_max)
    function eval_psi(d_try, g_1, λ_max)
        z_1 = z_b_at_lambda(λ_max, 1.0, γ, η)
        z_β = z_b_at_lambda(λ_max, β, γ, η)
        h_total = 0.0
        for j in 1:r
            d_j = D[j]
            if j ≤ d_try
                # Condensed: λ_j = χ_j = λ_max. Ferromagnetic coalesced formula
                # from Phi1_data, which coincides with the Ψ_β per-mode term
                # for the condensed branch (verified algebraically).
                h_total += -0.5 - c_rmt / (4 * g_1^2) + d_j * g_1 / (2γ) + η * d_j / (2 * g_1) -
                           log(g_1 * η * d_j) / 2
            else
                # Not condensed: λ_j at the bulk edge, χ_j = G⁻¹(η d_j) (PM-aligned).
                # The per-mode value is the paramagnetic h_k of Phi1_data.
                h_total += η * d_j^2 / (4γ) - log(c_rmt) / 2
            end
        end
        return (r / 2) * log(c_rmt) - K_train * η * z_1 - z_β + h_total
    end

    # Try d = 0 (no condensation). Admissible when the paramagnetic
    # Lagrange multiplier 1 + 1/c_rmt is above the largest paramagnetic
    # outlier position: condition reduces to g_paramag[1] ≥ max(1, β),
    # where g_paramag[1] = c_rmt/max(a_1, √c_rmt). Under this condition,
    # λ_max equals the largest paramagnetic outlier position (edge if no
    # aligned mode, outlier 1/a_1 + a_1/c_rmt otherwise), with
    # g_1 = G(λ_max) = min(c_rmt/a_1, √c_rmt) for the aligned-outlier case.
    function try_d0()
        a1 = η * D[1]
        g_para_1 = c_rmt / max(a1, edge_g)
        # d=0 valid iff neither training (b=1) nor sampling (b=β) pins the
        # paramagnetic saddle Lagrange multiplier μ_pm = 1 + 1/c_rmt: i.e.,
        # 1 ≤ g_para_1 (training PM), β ≤ g_para_1 (sampling PM).
        g_para_1 ≥ 1.0 - 1e-10 || return nothing
        g_para_1 ≥ β - 1e-10   || return nothing
        if a1 > edge_g
            g_1 = c_rmt / a1
            λ_max = 1/a1 + a1/c_rmt
        else
            g_1 = edge_g
            λ_max = 2/edge_g
        end
        return (0, g_1, λ_max)
    end

    # Try d ≥ 1 with assumed pinning branches (train_pin, samp_pin)
    function try_d(d_try::Int, train_pin::Bool, samp_pin::Bool)
        σ_cond = sum(D[1:d_try])
        α = (train_pin ? K_train * η : 0.0) + (samp_pin ? 1.0 : 0.0)
        β_const = (train_pin ? K_train * η : 0.0) + (samp_pin ? β : 0.0)
        α ≤ 0 && return nothing    # no pinning possible at d ≥ 1
        # α·g² + g·(η σ_cond − β_const) − d·γη = 0
        a_q = α
        b_q = η * σ_cond - β_const
        c_q = -d_try * c_rmt
        disc = b_q^2 - 4 * a_q * c_q
        disc < 0 && return nothing
        g_1 = (-b_q + sqrt(disc)) / (2 * a_q)
        g_1 ≤ 0 && return nothing
        g_1 > edge_g + 1e-10 && return nothing    # must be below bulk edge
        # Check pinning self-consistency
        train_pin  && (g_1 > 1.0 + 1e-10) && return nothing
        !train_pin && (g_1 < 1.0 - 1e-10) && return nothing
        samp_pin   && (g_1 > β + 1e-10)   && return nothing
        !samp_pin  && (g_1 < β - 1e-10)   && return nothing
        # Check consistency with next mode: new outlier must be above the
        # would-be position of mode d_try+1 at paramagnetic (a_{d+1} > g_1)
        if d_try < r
            a_next = η * D[d_try + 1]
            g_paramag_next = c_rmt / max(a_next, sqrt(c_rmt))
            g_1 ≥ g_paramag_next - 1e-10 && return nothing
        end
        λ_max = 1 / g_1 + g_1 / c_rmt
        return (d_try, g_1, λ_max)
    end

    # Iterate d = 0, 1, ..., r over the four pinning regimes.
    for d_try in 0:r
        if d_try == 0
            res = try_d0()
            res === nothing || return eval_psi(res...)
        else
            for (tp, sp) in [(true, true), (true, false), (false, true), (false, false)]
                res = try_d(d_try, tp, sp)
                res === nothing || return eval_psi(res...)
            end
        end
    end
    # Fallback: ferromagnetic-sticky saddle (analog of RMT_Solution.solve's
    # final branch). All eigenvalues pin at the bulk edge; g_1 = √(γη),
    # λ_max = 2/√(γη). This occurs when no d ∈ {0,…,r} admits a physical
    # quadratic solution (g_1 > edge_g) and the saddle snaps to the edge.
    # For eval_psi we use d_eff = count of aligned modes so per-mode terms
    # correctly use the condensed formula for the aligned subset.
    let g_1 = edge_g, λ_max = 2 / edge_g
        d_eff = count(η * D[j] > edge_g for j in 1:r)
        d_eff = max(d_eff, 1)
        return eval_psi(d_eff, g_1, λ_max)
    end
end

# ── Perturbed eigenvalues for predictive posterior ────────────────────

"""
Eigenvalues of C̃_x when x ~ P_{W*} (forward KL of predictive posterior).
Returns sorted descending vector of length 3.
τ = λ₊ + λ₋ + c₂ = 2 + 1/η.
"""
function forward_pp_eigenvalues(c1, c2, η)
    Δ = sqrt((c1 - 1/η)^2 + 2/η * (c1 - c2)^2)
    λp = (c1 + 1/η + Δ) / 2
    λm = (c1 + 1/η - Δ) / 2
    return sort!([λp, λm, c2]; rev=true)
end

"""
β-generalized forward_pp_eigenvalues: eigenvalues of C + (β/(Nη))·xx'
for x ~ P_{W*} at sampling inverse temperature β. Per
`notes/temperature_tuning_kls_beta_note.md` §4, the formula follows
from replacing 1/η with β/η in the β=1 closed form. At β=1 it
reduces to forward_pp_eigenvalues.
    τ^(β) = λ₊^(β) + λ₋^(β) + c₂ = 2 + β/η.
"""
function forward_pp_eigenvalues_beta(c1, c2, η, β)
    Δ = sqrt((c1 - β/η)^2 + 2*β/η * (c1 - c2)^2)
    λp = (c1 + β/η + Δ) / 2
    λm = (c1 + β/η - Δ) / 2
    return sort!([λp, λm, c2]; rev=true)
end

"""
Eigenvalues of C̃_x when x ~ P_pp (predictive posterior entropy).
Returns sorted descending vector of length 3.
"""
function pp_self_eigenvalues(c1, c2, γ, η, sol)
    ξ1, ξ2 = sol.ξ[1], sol.ξ[2]

    q1 = c1 / 2 - γ / (2η) * (ξ1 - 1) / c1
    q2 = c2 / 2 - γ / (2η) * (ξ2 - 1) / c2
    q0 = 1 - q1 - q2

    # Cubic: λ(λ-c₁)(λ-c₂) - (1/η)[q₁λ(λ-c₂) + q₂λ(λ-c₁) + q₀(λ-c₁)(λ-c₂)] = 0
    # = λ³ - (c₁+c₂+1/η)λ² + [c₁c₂ + (q₁c₂+q₂c₁+q₀(c₁+c₂))/η]λ - q₀c₁c₂/η
    p = -(c1 + c2 + 1/η)
    q = c1 * c2 + (q1 * c2 + q2 * c1 + q0 * (c1 + c2)) / η
    r = -q0 * c1 * c2 / η

    # Companion matrix for λ³ + pλ² + qλ + r = 0
    C = [0.0 0.0 -r;
         1.0 0.0 -q;
         0.0 1.0 -p]
    roots = real.(eigvals(C))
    # Eigenvalues of a PSD matrix must be non-negative;
    # tiny negative values are floating-point artifacts.
    @assert all(roots .> -1e-6) "pp_self_eigenvalues: root too negative: $roots"
    clamp!(roots, 0, Inf)
    return sort!(roots; rev=true)
end

"""
β-generalized pp_self_eigenvalues: roots of the β-cubic of
`notes/temperature_tuning_kls_beta_note.md` §3,
    c̃(c̃−c₁)(c̃−c₂) = (β/η)[q₁c̃(c̃−c₂) + q₂c̃(c̃−c₁) + q₀(c̃−c₁)(c̃−c₂)]

At β=1 the q_k should reduce to the rigorous Eq.~533 / Eq.~630 values
    q_k^{β=1,exact} = c_k/2 − γ(ξ_k-1)/(2η c_k).
The note's closed form q_k^(β) = u_k(1 − g_1/β)_+ matches this
algebraically in the non-degenerate condensed regime (FMo d=1) — ratio
1.000 — but in the FMe (`ferromagnetic_sticky`) phase the degenerate
saddle breaks first-order PT and the two formulas disagree by a
γη-dependent factor (≈1.62 at our γ=0.1 FMe anchor; the note's
formula is 0.528 vs Eq.~533's 0.327). We calibrate by multiplying the
β-scaling of the note's formula onto the exact β=1 coefficient:
    q_k^(β) = [c_k/2 − γ(ξ_k-1)/(2η c_k)] · (1 − g_1/β)_+ / (1 − g_1)_+.
This reduces to Eq.~630's q_k at β=1 exactly (giving pp_self_eigenvalues
roots back), and at β > g_1 it scales identically to the note.
"""
function pp_self_eigenvalues_beta(c1, c2, γ, η, β, sol)
    g_1 = sol.g[1]
    ξ1, ξ2 = sol.ξ[1], sol.ξ[2]
    if sol.d ≥ 1
        # Posterior is condensed (FMo d=1/d=2 or FMe): calibrate the note's
        # β-scaling factor (1 − g_1/β)_+ onto the rigorous Eq.~533 coefficient
        # so β=1 matches pp_self_eigenvalues exactly in all condensed phases.
        q1_exact = c1/2 - γ/(2η) * (ξ1 - 1) / c1
        q2_exact = c2/2 - γ/(2η) * (ξ2 - 1) / c2
        scale = max(0.0, 1 - g_1 / β) / max(1e-14, 1 - g_1)
        q1 = q1_exact * scale
        q2 = q2_exact * scale
    else
        # Posterior is paramagnetic (PMo/PMe): q_k^{exact}(β=1) = 0 so the
        # calibration above is degenerate. Use the note's formula directly,
        # which captures the sampling-side sPM → sFM condensation for β > β_c.
        q1 = sol.u[1] * max(0.0, 1 - g_1 / β)
        q2 = sol.u[2] * max(0.0, 1 - g_1 / β)
    end
    q0 = 1 - q1 - q2

    p = -(c1 + c2 + β/η)
    q = c1 * c2 + (β/η) * (q1 * c2 + q2 * c1 + q0 * (c1 + c2))
    r = -(β/η) * q0 * c1 * c2

    C = [0.0 0.0 -r;
         1.0 0.0 -q;
         0.0 1.0 -p]
    roots = real.(eigvals(C))
    @assert all(roots .> -1e-6) "pp_self_eigenvalues_beta: root too negative: $roots"
    clamp!(roots, 0, Inf)
    return sort!(roots; rev=true)
end

# ── KL divergences (per N, leading order) ─────────────────────────────
# All four KLs are distinct O(N) quantities.

"""
    forward_kl_typical(γ, η, ω) -> Float64

⟨D_KL(P_{W*} || P_W)⟩_{P_η} / N

= avg_E_fwd + F₁ + (ln ω - 1)/2
"""
function forward_kl_typical(γ::Real, η::Real, ω::Real)
    pq = posterior_quantities(γ, η, ω)
    avg_E = avg_energy_teacher_in_student(pq.sol.ξ[1], η, ω)
    return avg_E + pq.F1 - teacher_entropy_noA(ω)
end

"""
    reverse_kl_typical(γ, η, ω) -> Float64

⟨D_KL(P_W || P_{W*})⟩_{P_η} / N

= avg_E_pp + (ω* - 1 - ln ω* + F_sc(μ))/2
"""
function reverse_kl_typical(γ::Real, η::Real, ω::Real)
    pq = posterior_quantities(γ, η, ω)
    avg_E_pp = avg_energy_pp_in_teacher(pq.sol.ξ[1], γ, η, ω)
    H_student = student_entropy_noA(pq.g_Z, pq.c_rmt)
    return avg_E_pp + teacher_logZ_noA(ω) - H_student
end

"""
    forward_kl_predictive(γ, η, ω) -> Float64

D_KL(P_{W*} || P_pp) / N

= -[Δβ₀ + Φ₁^data(λ₊,λ₋,c₂; τ') - Φ₁^data(c₁,c₂; τ)] - (1 - ln ω)/2
where Δβ₀ = ln(γη)/2  (see notes/Δβ₀ fix.md).
"""
function forward_kl_predictive(γ::Real, η::Real, ω::Real)
    c1, c2 = teacher_c(ω)
    c_rmt = γ * η

    # Perturbed eigenvalues (R=3, τ=2+1/η)
    c_pert = forward_pp_eigenvalues(c1, c2, η)

    Φ_orig = Phi1_data([c1, c2], γ, η)
    Φ_pert = Phi1_data(c_pert, γ, η)

    Δβ₀ = log(c_rmt) / 2
    return -(Δβ₀ + Φ_pert - Φ_orig) - teacher_entropy_noA(ω)
end

"""
    reverse_kl_predictive(γ, η, ω) -> Float64

D_KL(P_pp || P_{W*}) / N

= avg_E_pp + (ω - ln ω)/2 + Δβ₀ + Φ₁^data(pp; τ') - Φ₁^data(orig; τ)
where Δβ₀ = ln(γη)/2  (see notes/Δβ₀ fix.md).
"""
function reverse_kl_predictive(γ::Real, η::Real, ω::Real)
    c1, c2 = teacher_c(ω)
    c_rmt = γ * η
    sol = RMT.solve([c1, c2], γ, η)

    avg_E_pp = avg_energy_pp_in_teacher(sol.ξ[1], γ, η, ω)

    # PP self-eigenvalues (R=3, τ=2+1/η)
    c_pp = pp_self_eigenvalues(c1, c2, γ, η, sol)

    Φ_orig = _Phi1_data([c1, c2], γ, η, sol)
    Φ_pp = Phi1_data(c_pp, γ, η)

    Δβ₀ = log(c_rmt) / 2
    return avg_E_pp + teacher_logZ_noA(ω) + Δβ₀ + Φ_pp - Φ_orig
end

"""
    all_kl_divergences(γ, η, ω) -> NamedTuple

Compute all four KL divergences per N. These are four distinct O(N) quantities.

Returns:
- `fwd_typ`: ⟨D_KL(P_{W*} || P_W)⟩ / N
- `rev_typ`: ⟨D_KL(P_W || P_{W*})⟩ / N
- `fwd_pp`: D_KL(P_{W*} || P_pp) / N
- `rev_pp`: D_KL(P_pp || P_{W*}) / N
- `phase`, `M`, `ξ`, `g`: saddle-point quantities
"""
function all_kl_divergences(γ::Real, η::Real, ω::Real)
    pq = posterior_quantities(γ, η, ω)
    ξ1 = pq.sol.ξ[1]

    avg_E_fwd = avg_energy_teacher_in_student(ξ1, η, ω)
    fwd_typ = avg_E_fwd + pq.F1 - teacher_entropy_noA(ω)

    avg_E_pp = avg_energy_pp_in_teacher(ξ1, γ, η, ω)
    H_student = student_entropy_noA(pq.g_Z, pq.c_rmt)
    rev_typ = avg_E_pp + teacher_logZ_noA(ω) - H_student

    fwd_pp = forward_kl_predictive(γ, η, ω)
    rev_pp = reverse_kl_predictive(γ, η, ω)

    return (;
        fwd_typ, rev_typ, fwd_pp, rev_pp,
        phase = pq.sol.phase,
        M = pq.sol.M,
        ξ = pq.sol.ξ,
        g = pq.sol.g,
        u = pq.sol.u,
    )
end

# ── Phase boundary helper ─────────────────────────────────────────────

"""
    phase_label(γ, η, ω) -> Symbol

Return phase label for the given parameters.
"""
function phase_label(γ::Real, η::Real, ω::Real)
    c1, c2 = teacher_c(ω)
    sol = RMT.solve([c1, c2], γ, η)
    return sol.phase
end

end # module
