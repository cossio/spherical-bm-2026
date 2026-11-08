#!/usr/bin/env julia
# Validation of the coupled-Langevin SBM dynamics against the large-N
# MSR/DMFT solution at K=1 or 2, β=1, for four cases at (γ,η)=(0.5,3).
# Full positive phase, one persistent negative-phase particle, GOE weight noise.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

using JLD2, CairoMakie, LaTeXStrings, Printf, Statistics

const SCRIPT_DIR = @__DIR__
const DATA_DIR   = joinpath(SCRIPT_DIR, "data", "validate_msr_K2")
const FIG_DIR    = SCRIPT_DIR

struct PanelCase
    tag::String                # data-file prefix
    label::String              # panel title
    c_text::String             # c summary for subtitle
    condensed::Bool            # condensation state (drives phase-label color)
    phase_label::LaTeXString   # (h, u_k) order-parameter label
    K::Int
end

# Phases evaluated at γ=0.5, η=3:
#   A) c=(1.5,0.5): γ<c_1 (condensed), η c_2² = 0.75 > γ (mode 2 outlier).
#   B) c=(1.8,0.2): γ<c_1 (condensed), η c_2² = 0.12 < γ (mode 2 in bulk).
#   C) c=1:        γ<c (condensed), η c² = 3 > γ (outlier).
#   D) c=0.3:      γ>c (uncondensed), η c² = 0.27 < γ (no outlier).
cases = [
    PanelCase("K2_FM_near", "A", "(c_1,c_2)=(1.5,\\,0.5)", true,  L"h,u_1,u_2\ne 0", 2),
    PanelCase("K2_FM_asym", "B", "(c_1,c_2)=(1.8,\\,0.2)", true,  L"h,u_1\ne 0,\,u_2=0", 2),
    PanelCase("K1_FM",      "C", "c=1.0",                  true,  L"h,u_1\ne 0", 1),
    PanelCase("K1_PM",      "D", "c=0.3",                  false, L"h=u_1=0", 1),
]

const N_finN = 4000
const n_seeds = 5
const γ_phys = 0.5
const η_phys = 3.0
const β_phys = 1.0
const ν_phys = 0.3
const noise_floor = 1 / sqrt(N_finN)   # ≈ 0.0158

function load_msr(tag)
    d = load(joinpath(DATA_DIR, "$(tag)_msr.jld2"))
    (; t=Vector{Float64}(d["t"]),
       s=Array{Float64}(d["s"]),
       μ=Vector{Float64}(d["κ"]))         # MSR solver stores as "κ"; paper uses μ
end

function load_fn(tag)
    fns = map(1:n_seeds) do i
        d = load(joinpath(DATA_DIR, "$(tag)_fn_seed$(i).jld2"))
        (; t=Vector{Float64}(d["t"]),
           s=Array{Float64}(d["s"]),
           μ=Vector{Float64}(d["μ"]))
    end
    return fns
end

# Linear interp for mean/SEM on a reference grid
function interp_linear(t, y, tq)
    yq = similar(tq)
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

function fn_mean_sem(fns, key, t_ref; row=0)
    n = length(fns)
    mat = zeros(n, length(t_ref))
    for (i, fn) in enumerate(fns)
        y = row == 0 ? getfield(fn, key) : getfield(fn, key)[row, :]
        mat[i, :] = interp_linear(fn.t, y, t_ref)
    end
    return vec(mean(mat; dims=1)),
           (n > 1 ? vec(std(mat; dims=1)) ./ sqrt(n) : zeros(length(t_ref)))
end

# Sign-align finite-N seeds to the MSR solution. The Langevin dynamics
# is invariant under x → −x (Z₂), which flips all s_a simultaneously
# but leaves μ invariant; over long T, finite-N noise can flip sign.
function align_fn(fns, msr; threshold=0.1)
    s1_end = msr.s[1, end]
    map(fns) do fn
        if abs(s1_end) > threshold && sign(fn.s[1, end]) != sign(s1_end)
            (; t=fn.t, s=-fn.s, μ=fn.μ)
        else
            fn
        end
    end
end

# ── Style: Okabe-Ito colorblind-safe palette ─────────────────────────
const CHAN_COLORS = (RGBf(0.902, 0.624, 0.0),   # s₁ — orange
                     RGBf(0.337, 0.706, 0.914)) # s₂ — sky blue
const MSR_COLOR   = :black
const FN_COLOR    = RGBf(0.800, 0.475, 0.655)   # reddish purple
const FN_ALPHA    = 0.35
const BAND_ALPHA  = 0.28

# ── Figure ────────────────────────────────────────────────────────────
set_theme!(Theme(
    fonts = (regular="TeX Gyre Pagella", bold="TeX Gyre Pagella Bold"),
    fontsize = 10,
    Axis = (
        xlabelsize = 12, ylabelsize = 12, titlesize = 11,
        xticklabelsize = 9, yticklabelsize = 9,
        xgridvisible = false, ygridvisible = false,
        xtickalign = 1.0, ytickalign = 1.0,
        spinewidth = 0.8,
    ),
    Lines = (linewidth = 1.4,),
))

# ~1.7-column width in REVTeX: use ~540 pt ≈ 7.5 in ≈ 19 cm
fig = Figure(size = (560, 360))

# load & align everything up front
data = map(cases) do c
    msr = load_msr(c.tag)
    fns = align_fn(load_fn(c.tag), msr)
    (; case=c, msr, fns)
end

for (col, d) in enumerate(data)
    c     = d.case
    msr   = d.msr
    fns   = d.fns
    t_ref = msr.t

    # Row 1: s_a(t)
    ax_s = Axis(fig[1, col];
                ylabel = col == 1 ? L"s_a(t)" : "",
                xticklabelsvisible = false, xlabelvisible = false,
                title = L"%$(c.c_text)",
                titlegap = 2, xticks = 0:10:30)
    Label(fig[1, col, TopLeft()], c.label;
          fontsize = 11, font = :bold, halign = :left, valign = :bottom,
          padding = (0, 0, 4, 0))

    # Noise-floor reference band (±1/√N around 0) as a faint gray region
    band!(ax_s, t_ref, fill(-noise_floor, length(t_ref)),
                        fill( noise_floor, length(t_ref));
          color = (:gray, 0.15))

    for a in 1:c.K
        fn_m, fn_se = fn_mean_sem(fns, :s, t_ref; row=a)
        band!(ax_s, t_ref, fn_m .- fn_se, fn_m .+ fn_se;
              color = (CHAN_COLORS[a], BAND_ALPHA))
        lines!(ax_s, t_ref, fn_m;
               color = (CHAN_COLORS[a], 0.85), linewidth = 1.0,
               linestyle = :dash)
        lines!(ax_s, msr.t, Vector(msr.s[a, :]);
               color = CHAN_COLORS[a], linewidth = 1.6)
    end

    # Annotate (h, u_k) order-parameter phase label — upper-right, inset
    # from the frame so it can't collide with the noise-floor band edge.
    phase_col = c.condensed ? RGBf(0.84, 0.37, 0.00) : RGBf(0.00, 0.45, 0.70)
    text!(ax_s, 0.96, 0.92; space = :relative, text = c.phase_label,
          fontsize = 10, color = (phase_col, 0.95),
          align = (:right, :top))

    # Row 2: κ(t)
    ax_μ = Axis(fig[2, col];
                xlabel = L"t",
                ylabel = col == 1 ? L"\kappa(t)" : "",
                xticks = 0:10:30, xtickalign = 1.0)
    fn_m, fn_se = fn_mean_sem(fns, :μ, t_ref)
    band!(ax_μ, t_ref, fn_m .- fn_se, fn_m .+ fn_se;
          color = (FN_COLOR, BAND_ALPHA))
    lines!(ax_μ, t_ref, fn_m;
           color = (FN_COLOR, 0.85), linewidth = 1.0, linestyle = :dash)
    lines!(ax_μ, msr.t, msr.μ;
           color = MSR_COLOR, linewidth = 1.6)

    # Shared x-limits for both rows, with a tiny right margin so the
    # final tick label ("30") doesn't sit flush against the axis line.
    Tmax = maximum(t_ref)
    xlims!(ax_s, -0.2, Tmax + 0.5)
    xlims!(ax_μ, -0.2, Tmax + 0.5)
    # Headroom on s-axis so the (h, u_k) phase label clears the s_1 plateau.
    ylims!(ax_s, -0.15, 1.05)
end

# ── Shared legend below ──────────────────────────────────────────────
legend_elems = [
    LineElement(color = MSR_COLOR,   linewidth = 1.6, linestyle = :solid),
    LineElement(color = (FN_COLOR, 0.9), linewidth = 1.0, linestyle = :dash),
    PolyElement(color = (:gray, 0.2), strokecolor = :transparent),
    LineElement(color = CHAN_COLORS[1], linewidth = 1.6, linestyle = :solid),
    LineElement(color = CHAN_COLORS[2], linewidth = 1.6, linestyle = :solid),
]
legend_labels = [
    L"\text{MSR/DMFT}\ (N\!\to\!\infty)",
    L"\text{finite-}N\ \langle\cdot\rangle\,\pm\,\mathrm{SEM}",
    L"\pm 1/\sqrt{N}",
    L"a=1",
    L"a=2",
]
Legend(fig[3, 1:length(cases)], legend_elems, legend_labels;
       orientation = :horizontal, nbanks = 1,
       framevisible = false, labelsize = 8, patchsize = (16, 8),
       tellwidth = false, tellheight = true, padding = (0, 0, 0, 0))

rowgap!(fig.layout, 1, 6)
rowgap!(fig.layout, 2, 2)
colgap!(fig.layout, 10)

# Save
out_pdf = joinpath(FIG_DIR, "fig_msr_langevin_validation.pdf")
save(out_pdf, fig)
println("Saved → $out_pdf")
