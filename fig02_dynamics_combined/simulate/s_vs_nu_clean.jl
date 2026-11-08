# Stationary s_∞(ν) sweep at γ ∈ {0.8, 0.85, 0.9} on a log-spaced ν grid
# in [1, 1000] via the FFT solver. ν_c is bracketed by bisection of
# F(s=0) over the same interval. Δτ = clamp(0.2/ν, 1e-5, 0.005), T=15,
# N_τ = nextpow(2, ⌈T/Δτ⌉) capped at 65 536. Outputs three JLD2 files
# (one per γ) into ../data/.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using JLD2
using Printf

include(joinpath(@__DIR__, "..", "..", "_lib", "msr", "stationary_fft.jl"))
using .StationaryFFT

const η = 3.0
const β = 1.0
const γ_list = [0.8, 0.85, 0.9]
const ν_lo = 1.0
const ν_hi = 1000.0
const N_NU = 30      # log-spaced points

datadir = joinpath(@__DIR__, "..", "data")
mkpath(datadir)

# ── (N_τ, Δτ) per ν: same scaling as the high-ν script, applied uniformly ──
function discretization_for(ν)
    Δτ = clamp(0.2 / ν, 1e-5, 0.005)
    T_target = 15.0
    Nreq = ceil(Int, T_target / Δτ)
    Nτ = nextpow(2, Nreq)
    Nτ = min(Nτ, 65_536)
    return (Nτ, Δτ)
end

# Tight s_scan: above ν_c the asymptote is small (h u_1 ≤ 0.31 at γ = 0.8).
function s_scan_for(γ)
    g1 = sqrt(γ)
    asy = sqrt(max(0.0, 1 - g1) * max(0.0, 1 - g1 / 3.0))
    smax = clamp(1.5 * asy + 0.1, 0.4, 0.8)
    return collect(range(0.0, smax, length=80))
end

function solve_one(γ, ν)
    Nτ, Δτ = discretization_for(ν)
    s_scan = s_scan_for(γ)
    r = solve_stationary_fft(γ, η, β, ν;
                             N_τ=Nτ, Δτ=Δτ,
                             s_scan=s_scan,
                             maxiter=2000, tol=1e-10)
    return (s_inf=r.s_inf, μ_inf=r.μ_inf, F0=r.F0, Nτ=Nτ, Δτ=Δτ)
end

function bisect_νc(γ, ν_a, ν_b; iters=40, tol=1e-3)
    lo, hi = ν_a, ν_b
    for _ in 1:iters
        mid = 0.5 * (lo + hi)
        r = solve_one(γ, mid)
        # F(s=0) > 0 ⇒ PM (no FM root at s=0); F(s=0) < 0 ⇒ FM. The PM
        # region is the high-ν side. So ν_c is the boundary where F0 → 0.
        # The convention used elsewhere (g0p85 script) brackets [F0<0, F0>0].
        r.F0 > 0 ? (hi = mid) : (lo = mid)
        (hi - lo) < tol && break
    end
    return 0.5 * (lo + hi)
end

# Convergence probe at one ν (used for the ad-hoc print). Compares the
# solver result at (N_τ, Δτ) and (2 N_τ, Δτ).
function convergence_check(γ, ν)
    Nτ, Δτ = discretization_for(ν)
    Nτ_double = min(2 * Nτ, 65_536)
    if Nτ_double == Nτ
        return (s1=NaN, s2=NaN, diff=NaN)
    end
    s_scan = s_scan_for(γ)
    r1 = solve_stationary_fft(γ, η, β, ν;
                              N_τ=Nτ, Δτ=Δτ, s_scan=s_scan,
                              maxiter=2000, tol=1e-10)
    r2 = solve_stationary_fft(γ, η, β, ν;
                              N_τ=Nτ_double, Δτ=Δτ, s_scan=s_scan,
                              maxiter=2000, tol=1e-10)
    return (s1=r1.s_inf, s2=r2.s_inf, diff=abs(r1.s_inf - r2.s_inf))
end

function compute_and_save(γ::Float64)
    println("\n──── γ = $γ ────")
    flush(stdout)

    # Log-spaced ν grid in [ν_lo, ν_hi]
    ν_grid = [10.0^x for x in range(log10(ν_lo), log10(ν_hi), length=N_NU)]
    ν_grid = sort(unique(ν_grid))

    println("Sweep: $(length(ν_grid)) ν ∈ [$ν_lo, $ν_hi]  (log-spaced)")
    flush(stdout)

    results = Dict{Float64,NamedTuple}()
    t0 = time()
    for (i, ν) in enumerate(ν_grid)
        Nτ, Δτ = discretization_for(ν)
        r = solve_one(γ, ν)
        results[ν] = r
        @printf("  %2d/%d  ν=%.4g  N_τ=%d  Δτ=%.5f   F0=%.4g   s_inf=%.6f   μ_inf=%.4f\n",
                i, length(ν_grid), ν, Nτ, Δτ, r.F0, r.s_inf, r.μ_inf)
        flush(stdout)
    end
    dt_sweep = time() - t0
    println("  sweep time: $(round(dt_sweep; digits=1)) s")

    ν_sorted = sort(collect(keys(results)))
    s_inf_v = [results[ν].s_inf for ν in ν_sorted]
    μ_inf_v = [results[ν].μ_inf for ν in ν_sorted]
    F0_v    = [results[ν].F0    for ν in ν_sorted]

    # ── ν_c via bisection on F0 sign change in [ν_lo, ν_hi] ──
    νc = NaN
    sgn0 = F0_v[1] > 0
    sgnE = F0_v[end] > 0
    if sgn0 != sgnE
        # Find first sign-change bracket [ν_i, ν_{i+1}]
        for i in 1:length(ν_sorted)-1
            if (F0_v[i] > 0) != (F0_v[i+1] > 0)
                νc = bisect_νc(γ, ν_sorted[i], ν_sorted[i+1])
                break
            end
        end
    elseif !sgn0
        # F0 negative throughout: PM region exists below ν_lo (or never)
        νc = NaN
    else
        # F0 positive throughout: PM throughout? ν_c < ν_lo
        νc = NaN
    end
    println("  ν_c = ", isnan(νc) ? "NaN (no F0 sign-change in [$ν_lo, $ν_hi])" :
                       string(round(νc; sigdigits=5)))
    flush(stdout)

    # ── Monotonicity check ──
    n_drops = count(i -> s_inf_v[i+1] < s_inf_v[i] - 1e-4, 1:length(s_inf_v)-1)
    @printf("  monotonicity: n_drops = %d (threshold 1e-4)\n", n_drops)
    if n_drops > 0
        for i in 1:length(s_inf_v)-1
            if s_inf_v[i+1] < s_inf_v[i] - 1e-4
                @printf("    drop @ ν[%d]=%.4g (s=%.6f) → ν[%d]=%.4g (s=%.6f)\n",
                        i, ν_sorted[i], s_inf_v[i],
                        i+1, ν_sorted[i+1], s_inf_v[i+1])
            end
        end
    end

    # ── Asymptote comparison ──
    g1 = sqrt(γ)
    asy = sqrt(max(0.0, 1 - g1) * max(0.0, 1 - g1 / 3.0))
    s_high = s_inf_v[end]
    rel_err = (s_high - asy) / asy
    @printf("  s_inf(ν=%.4g) = %.6f   h·u_1 = %.6f   rel.err = %+.3e\n",
            ν_sorted[end], s_high, asy, rel_err)

    # ── Convergence probe at ν = ν_hi ──
    cc = convergence_check(γ, ν_sorted[end])
    if isfinite(cc.diff)
        @printf("  N_τ-doubling probe @ ν=%.4g: |Δs_inf| = %.3e (target < 1e-3)\n",
                ν_sorted[end], cc.diff)
    end

    # Filename
    γstr = γ == 0.8 ? "g0p8" : γ == 0.85 ? "g0p85" : γ == 0.9 ? "g0p9" :
           "g" * replace(string(γ), "." => "p")
    out = joinpath(datadir, "20260505_s_vs_nu_$(γstr)_clean.jld2")
    jldsave(out;
        ν       = ν_sorted,
        s_inf   = s_inf_v,
        μ_inf   = μ_inf_v,
        F0      = F0_v,
        ν_c     = νc,
        γ       = γ,
        η       = η,
        β       = β)
    println("  saved → $out  ($(length(ν_sorted)) points)")
    return (γ=γ, ν=ν_sorted, s_inf=s_inf_v, νc=νc,
            asy=asy, n_drops=n_drops, dt=dt_sweep)
end

t_all = time()
summaries = [compute_and_save(γ) for γ in γ_list]
total = time() - t_all

println("\n========== SUMMARY ==========")
@printf("%-6s  %-8s  %-12s  %-12s  %-9s  %-9s\n",
        "γ", "n_drops", "s_inf(ν=1000)", "h·u_1", "rel.err", "ν_c")
for s in summaries
    rel = (s.s_inf[end] - s.asy) / s.asy
    νc_s = isnan(s.νc) ? "NaN" : @sprintf("%.4g", s.νc)
    @printf("%-6.3f  %-8d  %-12.6f  %-12.6f  %+-9.2e  %s\n",
            s.γ, s.n_drops, s.s_inf[end], s.asy, rel, νc_s)
end
println("Total wall clock: $(round(total; digits=1)) s")
