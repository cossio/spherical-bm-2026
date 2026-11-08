#!/usr/bin/env julia
# Run the MSR/DMFT solver and finite-N Langevin (multiple seeds) on four
# K=1 or 2 covariance cases and write the per-case JLD2s the figure consumes.
# Full positive phase, one persistent negative-phase particle, GOE weight noise.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using JLD2, CairoMakie, Printf, Statistics

const SCRIPT_DIR = @__DIR__
const DATA_DIR   = joinpath(SCRIPT_DIR, "..", "data", "validate_msr_K2")
const FIG_DIR    = joinpath(SCRIPT_DIR, "..", "data")
mkpath(DATA_DIR); mkpath(FIG_DIR)

const MSR     = joinpath(SCRIPT_DIR, "..", "..", "_lib", "msr", "msr_solver.jl")
const FINITEN = joinpath(SCRIPT_DIR, "..", "..", "_lib", "msr", "finiteN.jl")

function run_script(script::String, kwargs::Dict; label::String="", outfile::String="")
    if !isempty(outfile) && isfile(outfile)
        println("  » [$label] SKIP (exists): $(basename(outfile))")
        return
    end
    args = String["--$(k)=$(v)" for (k, v) in kwargs]
    println("  » [$label] julia $(basename(script)) $(join(args, ' '))")
    flush(stdout)
    run(pipeline(`julia $script $args`; stdout=devnull, stderr=devnull))
end

const γ_phys = 0.5
const η_phys = 3.0
const β_phys = 1.0
const edge_val = 2 / sqrt(γ_phys * η_phys)

# Parameters chosen so the initial "transient dip" (W catching up to the
# signal while x randomizes) is shallow: min_t |s_1(t)| stays ≳ 0.15,
# comfortably above 1/√N = 0.016 at N=4000 (≈ 9-12× floor).

# Physical K=2 data constraint: c₁ + c₂ = K = 2 (trace of empirical
# covariance C = (1/N) Σ_a (data_a)(data_a)^T with ‖data_a‖² = N).
# Cases A-C use normalized data. Case D is the generalized-source
# uncondensed control with K=1 and c=0.3, as labeled in the manuscript.

struct Case
    tag::String
    c_csv::String        # source eigenvalues; their count sets K
    s0_csv::String       # initial overlaps
    ν::Float64           # persistent-particle sampling rate
    Tmax::Float64
    N_finN::Int
    n_seeds::Int
end

cases = [
    # N=4000 gives a noise floor 1/√N ≈ 0.016. The condensed cases start
    # at s₁(0)=0.30 so the transient remains resolvable above that floor.
    # Uncondensed channels are compared with their finite-N fluctuations.
    Case("K2_FM_near", "1.5,0.5", "0.30,0.25", 0.3, 30.0, 4000, 5),
    Case("K2_FM_asym", "1.8,0.2", "0.30,0.15", 0.3, 30.0, 4000, 5),
    # Clean K=1 FM (physical Σc=K=1): single channel, no PM coexisting.
    Case("K1_FM",      "1.0",     "0.30",      0.3, 30.0, 4000, 5),
    # Pure global-PM test: K=1 with c below threshold. Rule is waived.
    Case("K1_PM",      "0.3",     "0.10",      0.3, 30.0, 4000, 5),
]

load_msr(path) = (d = load(path);
                  (t = Vector{Float64}(d["t"]),
                   s = Array{Float64}(d["s"]),
                   κ = Vector{Float64}(d["κ"])))

load_fn(path) = (d = load(path);
                 (t = Vector{Float64}(d["t"]),
                  s = Array{Float64}(d["s"]),
                  μ = Vector{Float64}(d["μ"])))

results = Dict{String, Any}()

for case in cases
    println("="^74)
    println("CASE $(case.tag):  c=$(case.c_csv), s0=$(case.s0_csv), Tmax=$(case.Tmax)")
    println("="^74)

    c_vec = parse.(Float64, split(case.c_csv, ','))
    K_case = length(c_vec)

    msr_out = joinpath(DATA_DIR, "$(case.tag)_msr.jld2")
    fn_outs = [joinpath(DATA_DIR, "$(case.tag)_fn_seed$(s).jld2") for s in 1:case.n_seeds]

    # ── MSR (deterministic, one run) ──────────────────────────────────
    run_script(MSR, Dict(
        "gamma"=>γ_phys, "eta"=>η_phys, "beta"=>β_phys, "nu"=>case.ν,
        "c"=>case.c_csv, "s0"=>case.s0_csv,
        "Tmax"=>case.Tmax, "nsave"=>80,
        "outfile"=>msr_out); label="MSR", outfile=msr_out)

    # ── Finite-N (n_seeds independent runs) ────────────────────────────
    for (i, fn_out) in enumerate(fn_outs)
        run_script(FINITEN, Dict(
            "gamma"=>γ_phys, "eta"=>η_phys, "beta"=>β_phys, "nu"=>case.ν,
            "c"=>case.c_csv, "s0"=>case.s0_csv,
            "N"=>case.N_finN, "Tmax"=>case.Tmax,
            "nsave"=>200, "seed"=>100 + i,
            "outfile"=>fn_out); label="FN seed=$(100+i)", outfile=fn_out)
    end

    msr = load_msr(msr_out)
    fns_raw = [load_fn(p) for p in fn_outs]

    # ── Sign-alignment: the dynamics is invariant under x → -x (global Z₂),
    # which flips all s_a simultaneously but leaves μ = ν(U + 1/β) invariant.
    # Finite-N noise can flip the condensed channel over long T. Align each
    # seed so sign(s₁_end) matches MSR (only when MSR s₁ is macroscopic).
    msr_s1_end = msr.s[1, end]
    align_threshold = 0.1
    fns = map(fns_raw) do fn
        if abs(msr_s1_end) > align_threshold && sign(fn.s[1, end]) != sign(msr_s1_end)
            (; t=fn.t, s=-fn.s, μ=fn.μ, flipped=true)
        else
            (; fn..., flipped=false)
        end
    end
    n_flipped = count(f -> f.flipped, fns)
    println("  Sign-aligned: flipped $n_flipped / $(length(fns)) seeds")

    results[case.tag] = (; case, msr, fns, c_vec, K_case)

    @printf("  MSR  end   s = %s   κ = %.4f\n",
            round.(msr.s[:, end]; digits=4), msr.κ[end])
    for (i, fn) in enumerate(fns)
        @printf("  FN#%d end  s = %s   μ = %.4f\n",
                i, round.(fn.s[:, end]; digits=4), fn.μ[end])
    end
    flush(stdout)
end

# ──────────────────────────────────────────────────────────────────────
# Interpolate finite-N seeds onto MSR time grid, compute mean/SEM
# ──────────────────────────────────────────────────────────────────────
function interp_linear(t::Vector{Float64}, y::AbstractVector{<:Real}, tq::AbstractVector{Float64})
    yq = Vector{Float64}(undef, length(tq))
    for (i, tx) in enumerate(tq)
        if tx <= first(t); yq[i] = first(y)
        elseif tx >= last(t); yq[i] = last(y)
        else
            j = searchsortedfirst(t, tx)
            α = (tx - t[j-1]) / (t[j] - t[j-1])
            yq[i] = (1 - α) * y[j-1] + α * y[j]
        end
    end
    return yq
end

function fn_mean_sem(fns, key::Symbol, t_ref::Vector{Float64}; row::Int=0)
    n = length(fns)
    mat = zeros(n, length(t_ref))
    for (i, fn) in enumerate(fns)
        y = row == 0 ? getfield(fn, key) : getfield(fn, key)[row, :]
        mat[i, :] = interp_linear(fn.t, y, t_ref)
    end
    m = vec(mean(mat; dims=1))
    se = n > 1 ? vec(std(mat; dims=1)) ./ sqrt(n) : zeros(length(t_ref))
    return m, se
end

# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────
println("\nBuilding figure…")
set_theme!(Theme(
    fontsize=12,
    Axis=(xlabelsize=13, ylabelsize=13, titlesize=13,
          xticklabelpad=8, yticklabelpad=6),
    Lines=(linewidth=2.0,),
))

chan_colors = (:tomato, :royalblue)

fig = Figure(size=(1650, 820))
legend_axes = Ref{Any}(nothing)
for (col, case) in enumerate(cases)
    res = results[case.tag]
    K_case = res.K_case
    t_ref = res.msr.t

    # Row 1: s_a(t) overlay — hide x-tick labels (shared with row 2)
    ax_s = Axis(fig[1, col];
                ylabel="s_a(t)",
                xticklabelsvisible=false, xlabelvisible=false,
                title="$(case.tag):  c=$(case.c_csv), ν=$(case.ν)")
    for a in 1:K_case
        fn_m, fn_se = fn_mean_sem(res.fns, :s, t_ref; row=a)
        band!(ax_s, t_ref, fn_m .- fn_se, fn_m .+ fn_se;
              color=(chan_colors[a], 0.25))
        lines!(ax_s, t_ref, fn_m;
               color=(chan_colors[a], 0.6), linewidth=1.4,
               linestyle=:dash,
               label=(col == 1 ? "FN ⟨s$a⟩ ±SEM" : nothing))
        lines!(ax_s, res.msr.t, Vector(res.msr.s[a, :]);
               color=chan_colors[a], linewidth=2.4,
               label=(col == 1 ? "MSR s$a" : nothing))
    end

    # Row 2: κ(t) overlay — compare MSR κ vs finite-N μ (already ν·(U+1/β))
    ax_κ = Axis(fig[2, col]; xlabel="t", ylabel="κ(t)",
                xticks=(0:6:18), xtickalign=1.0)
    fn_m, fn_se = fn_mean_sem(res.fns, :μ, t_ref)
    band!(ax_κ, t_ref, fn_m .- fn_se, fn_m .+ fn_se;
          color=(:grey, 0.30))
    lines!(ax_κ, t_ref, fn_m;
           color=(:grey25, 0.85), linewidth=1.4, linestyle=:dash,
           label=(col == 1 ? "FN ⟨μ⟩ ±SEM" : nothing))
    lines!(ax_κ, res.msr.t, res.msr.κ;
           color=:black, linewidth=2.4,
           label=(col == 1 ? "MSR κ" : nothing))

    if col == 1
        legend_axes[] = (ax_s, ax_κ)
    end
end

# Single shared legend below the figure (pulls from col-1 axes)
if legend_axes[] !== nothing
    ax_s1, ax_κ1 = legend_axes[]
    ncols = length(cases)
    Legend(fig[3, 1:ncols], ax_s1;
           orientation=:horizontal, nbanks=2, labelsize=10,
           tellwidth=false, tellheight=true, framevisible=true,
           merge=true)
    Legend(fig[4, 1:ncols], ax_κ1;
           orientation=:horizontal, nbanks=1, labelsize=10,
           tellwidth=false, tellheight=true, framevisible=true,
           merge=true)
end

Label(fig[0, :], "MSR/DMFT vs finite-N Langevin training  (N=$(cases[1].N_finN), $(cases[1].n_seeds) seeds per case)";
      fontsize=16)

fig_path = joinpath(FIG_DIR, "20260414_validate_msr_vs_finiteN_K2.pdf")
save(fig_path, fig)
println("Saved figure → $fig_path")

# ──────────────────────────────────────────────────────────────────────
# Quantitative discrepancy (last half of trajectory)
# ──────────────────────────────────────────────────────────────────────
println("\n" * "="^74)
println("QUANTITATIVE SUMMARY  —  discrepancy on last half of trajectory")
println("="^74)

summary_rows = Tuple{String, Float64, Float64, Float64, Float64}[]
for case in cases
    res = results[case.tag]
    K_case = res.K_case
    t_ref = res.msr.t
    mask = t_ref .> last(t_ref) / 2
    t_cmp = t_ref[mask]

    for a in 1:K_case
        fn_m, _ = fn_mean_sem(res.fns, :s, t_cmp; row=a)
        msr_y = Vector(res.msr.s[a, mask])
        dif = abs.(fn_m .- msr_y)
        rmse = sqrt(mean(dif.^2)); mx = maximum(dif)
        tag = "$(case.tag) s$a"
        @printf("  %-14s RMSE=%.4f  max=%.4f   [MSR end=%.4f, FN end=%.4f]\n",
                tag, rmse, mx, last(msr_y), last(fn_m))
        push!(summary_rows, (tag, rmse, mx, last(msr_y), last(fn_m)))
    end
    fn_m, _ = fn_mean_sem(res.fns, :μ, t_cmp)
    msr_y = res.msr.κ[mask]
    dif = abs.(fn_m .- msr_y)
    rmse = sqrt(mean(dif.^2)); mx = maximum(dif)
    tag = "$(case.tag) κ"
    @printf("  %-14s RMSE=%.4f  max=%.4f   [MSR end=%.4f, FN end=%.4f]\n",
            tag, rmse, mx, last(msr_y), last(fn_m))
    push!(summary_rows, (tag, rmse, mx, last(msr_y), last(fn_m)))
end

bundle_path = joinpath(DATA_DIR, "20260414_validate_msr_vs_finiteN_K2.jld2")
jldsave(bundle_path;
    cases=[c.tag for c in cases],
    γ=γ_phys, η=η_phys, β=β_phys, ν_by_case=[c.ν for c in cases],
    summary=summary_rows)
println("\nSaved summary → $bundle_path")
