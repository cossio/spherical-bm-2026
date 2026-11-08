# Stationary s_∞(ν) sweep at γ = 0.99 (same protocol as s_vs_nu_clean.jl).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using JLD2
using Printf

include(joinpath(@__DIR__, "..", "..", "_lib", "msr", "stationary_fft.jl"))
using .StationaryFFT

const γ = 0.99
const η = 3.0
const β = 1.0
const ν_lo = 1.0
const ν_hi = 1000.0
const N_NU = 30

datadir = joinpath(@__DIR__, "..", "data")
mkpath(datadir)

function discretization_for(ν)
    Δτ = clamp(0.2 / ν, 1e-5, 0.005)
    T_target = 15.0
    Nreq = ceil(Int, T_target / Δτ)
    Nτ = nextpow(2, Nreq)
    Nτ = min(Nτ, 65_536)
    return (Nτ, Δτ)
end

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

println("──── γ = $γ ────")
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

# ν_c via first ν at which s_inf > 0 (more robust than F0 sign-change for high γ)
νc = NaN
for i in 1:length(ν_sorted)
    if s_inf_v[i] > 0 && (i == 1 || s_inf_v[i-1] == 0)
        if i == 1
            νc = ν_sorted[1]   # already condensed at ν_lo
        else
            νc = sqrt(ν_sorted[i-1] * ν_sorted[i])  # log-midpoint between PM and FM
        end
        break
    end
end
println("  ν_c ≈ ", isnan(νc) ? "NaN (no transition in displayed range)" :
                  string(round(νc; sigdigits=5)))

n_drops = count(i -> s_inf_v[i+1] < s_inf_v[i] - 1e-4, 1:length(s_inf_v)-1)
@printf("  monotonicity: n_drops = %d\n", n_drops)

g1 = sqrt(γ)
asy = sqrt(max(0.0, 1 - g1) * max(0.0, 1 - g1 / 3.0))
@printf("  s_inf(ν=%.4g) = %.6f   h·u_1 = %.6f   rel.err = %+.3e\n",
        ν_sorted[end], s_inf_v[end], asy, (s_inf_v[end]-asy)/asy)

out = joinpath(datadir, "20260505_s_vs_nu_g0p99_clean.jld2")
jldsave(out;
    ν     = ν_sorted,
    s_inf = s_inf_v,
    μ_inf = μ_inf_v,
    F0    = F0_v,
    ν_c   = νc,
    γ     = γ,
    η     = η,
    β     = β)
println("  saved → $out  ($(length(ν_sorted)) points)")
