#!/usr/bin/env julia
# fig_ooe_wrap.jl
#
# Reproduce manuscript Fig. 6 by overlaying the large-N DMFT predictions on
# finite-N Langevin trajectories.
#
# The bundled theory CSVs cover the full simulation ν-grid
# (15 values from 0.05 to 3.00) on t ∈ [0, ~993] with 901 time points.
# Header columns are encoded as `nu_XpYY` (`p` for the decimal point).
# The bundled JLD2 file contains the finite-N trajectories used in the paper.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

using DelimitedFiles: readdlm
using JLD2: load
using Printf: @sprintf
using Statistics: mean
import CairoMakie
import Makie
using Makie: @L_str

const SCRIPT_DIR = @__DIR__
const DATA_DIR   = joinpath(SCRIPT_DIR, "data")
const DATA_FILE  = joinpath(DATA_DIR, "grokking_note_kl_nu_sweep_long.jld2")
const THEORY_DIR = joinpath(DATA_DIR, "theory")
const FIG_OUT    = get(ENV, "FIGURE_OUTPUT_DIR", SCRIPT_DIR)

isfile(DATA_FILE) || error("missing data file: $DATA_FILE")
mkpath(FIG_OUT)
d = load(DATA_FILE)

t          = Vector{Float64}(d["t"])
ν_grid     = Vector{Float64}(d["ν_grid"])
λ1         = Array{Float64,3}(d["λ1"])             # (T, ν, seed)
u1sq       = Array{Float64,3}(d["u1sq"])           # (T, ν, seed)
Drev_mean  = Matrix{Float64}(d["Drev_mean"])       # (T, ν)
pm_val     = Float64(d["pm_plateau"])
γ          = Float64(d["γ"])
η          = Float64(d["η"])
γη         = γ*η
λ_edge     = 2/sqrt(γη)

# Seed-means
λ1_mean   = dropdims(mean(λ1;   dims=3); dims=3)    # (T, ν)
u1sq_mean = dropdims(mean(u1sq; dims=3); dims=3)    # (T, ν)

# ── Theory data (theory-data-v2/) ───────────────────────────────────────
# Headers encode ν as `nu_XpYY`, e.g. `nu_0p05`, `nu_1p30`, `nu_3p00`.
function parse_nu_header(h::AbstractString)
    m = match(r"^nu_([0-9]+)p([0-9]+)$", h)
    m === nothing && return nothing
    return parse(Float64, m.captures[1] * "." * m.captures[2])
end

function load_theory(path)
    raw, hdr = readdlm(path, ','; header=true)
    hdrs = vec(hdr)
    tvec = Vector{Float64}(raw[:, 1])
    νs = Float64[]
    cols = Vector{Vector{Float64}}()
    for j in 2:length(hdrs)
        ν = parse_nu_header(String(hdrs[j]))
        ν === nothing && continue
        push!(νs, ν)
        push!(cols, Vector{Float64}(raw[:, j]))
    end
    return tvec, νs, cols
end

t_th_kl,  ν_th_kl,  cols_kl  = load_theory(joinpath(THEORY_DIR, "reverse_KL_merged.csv"))
t_th_l1,  ν_th_l1,  cols_l1  = load_theory(joinpath(THEORY_DIR, "lambda1_merged.csv"))
t_th_u1,  ν_th_u1,  cols_u1  = load_theory(joinpath(THEORY_DIR, "u1sq_merged.csv"))

const CMAP    = :viridis
cmap          = Makie.cgrad(CMAP)

# Show only every other ν value to reduce visual density.
ν_show_idx = 1:3:length(ν_grid)
ν_show     = ν_grid[ν_show_idx]
log_ν_lo, log_ν_hi = extrema(log10.(ν_show))
ν_to_frac(ν)  = (log10(ν) - log_ν_lo) / (log_ν_hi - log_ν_lo)
# ── Layout: three stacked narrow panels, sized for ~0.36\linewidth wrapfig
const FS = 6
fig = Makie.Figure(size=(130, 155), fontsize=FS,
                   figure_padding=(4, 5, 6, 14))

axkw = (xscale = log10,
        xgridvisible = false, ygridvisible = false,
        xlabelsize = FS, ylabelsize = FS,
        xticklabelsize = FS-1, yticklabelsize = FS-1,
        xminorticksvisible = true, yminorticksvisible = true,
        xminorticksize = 1, yminorticksize = 1,
        xticksize = 2, yticksize = 2,
        spinewidth = 0.5)

ax1 = Makie.Axis(fig[1, 1];
    ylabel = L"D_{\mathrm{KL}}(P_{W}\Vert P^\star) / N",
    xticklabelsvisible = false, axkw...)

ax2 = Makie.Axis(fig[2, 1];
    ylabel = L"\lambda_1(t)",
    xticklabelsvisible = false, axkw...)

ax3 = Makie.Axis(fig[3, 1];
    xlabel = L"t",
    ylabel = L"u_1^2(t)",
    axkw...)

Makie.linkxaxes!(ax1, ax2, ax3)
Makie.ylims!(ax1, 0.18, nothing)

keep = t .> 0
tlog = t[keep]
KL_log = Drev_mean[keep, :]
λ_log  = λ1_mean[keep, :]
u_log  = u1sq_mean[keep, :]

# Simulation lines: every other ν, colored by the same viridis gradient as
# theory but more transparent and thicker, so they read as a "fuzzy halo"
# around the crisp theory curves.
for j in ν_show_idx
    sim_color = (cmap[ν_to_frac(ν_grid[j])], 0.35)
    Makie.lines!(ax1, tlog, KL_log[:, j]; color = sim_color, linewidth = 1.6)
    Makie.lines!(ax2, tlog, λ_log[:,  j]; color = sim_color, linewidth = 1.6)
    Makie.lines!(ax3, tlog, u_log[:,  j]; color = sim_color, linewidth = 1.6)
end

# Theory overlays: solid colored lines, only at the displayed ν values.
function overlay_theory!(ax, tvec, νs, cols)
    keep_th = tvec .> 0
    for (νj, yvec) in zip(νs, cols)
        any(ν -> isapprox(ν, νj; atol=1e-6), ν_show) || continue
        col = cmap[ν_to_frac(νj)]
        Makie.lines!(ax, tvec[keep_th], yvec[keep_th];
            color = col, linewidth = 0.6)
    end
end
overlay_theory!(ax1, t_th_kl, ν_th_kl, cols_kl)
overlay_theory!(ax2, t_th_l1, ν_th_l1, cols_l1)
overlay_theory!(ax3, t_th_u1, ν_th_u1, cols_u1)

# Per-curve ν annotations on panel A: each label sits just above its theory
# curve in the plateau region, in the curve's color. The largest ν goes
# below to avoid crowding; the second-largest (ν≈0.85) is shifted left so
# its "ν=…" prefix does not collide with the other annotations.
let
    displayed_νs = sort!([νj for νj in ν_th_kl
                          if any(ν -> isapprox(ν, νj; atol=1e-6), ν_show)])
    n = length(displayed_νs)
    darken(c, f) = Makie.RGBAf(c.r * f, c.g * f, c.b * f, 1.0)
    for (i, νj) in enumerate(displayed_νs)
        col = cmap[ν_to_frac(νj)]
        label_col = darken(col, 0.55)
        yvec = cols_kl[findfirst(ν -> isapprox(ν, νj; atol=1e-6), ν_th_kl)]
        # ν=0.05, 0.20, 0.40 (i=1..3) sit above their plateau at t≈400, well
        # to the right; ν=0.40 specifically goes below its curve so it does
        # not crowd ν=0.20. ν≈0.85 (i=n−1) shifts left to t≈50 to avoid the
        # column on the right; the largest ν (i=n) goes below at t≈200.
        # 0.05, 0.20, 0.40, 0.85 go above their curves; 1.70 (largest ν)
        # goes below. All labels sit at t≈400 in a vertically aligned
        # column, flush against their curve (zero gap).
        place_below = (i == n)
        label_t = 400.0
        idx = argmin(abs.(t_th_kl .- label_t))
        y_at = yvec[idx]
        ν_str = @sprintf("%.2f", νj)
        Makie.text!(ax1, label_t, y_at;
            text = L"\nu=%$ν_str",
            fontsize = FS-3, color = label_col,
            align = place_below ? (:center, :top) : (:center, :bottom),
            offset = (0, 0))
    end
end

# Reference lines
Makie.hlines!(ax1, [pm_val];
    color = (:black, 0.55), linestyle = :dash, linewidth = 0.5)
Makie.text!(ax1, tlog[2]*5, pm_val;
    text = L"h=0", fontsize = FS-1,
    align = (:left, :top), color = (:black, 0.7), offset = (1, -1))

Makie.hlines!(ax2, [λ_edge];
    color = (:black, 0.55), linestyle = :dash, linewidth = 0.5)
Makie.text!(ax2, tlog[2]*1.5, λ_edge;
    text = L"2\sigma", fontsize = FS-1,
    align = (:left, :bottom), color = (:black, 0.7), offset = (1, 1))

# Static-equilibrium reference lines (the ν → ∞ limit) for the rank-K=2 SBM
# saddle (γ=0.4, η=10, ω*=2.5, c=(1.6, 0.4)).
const λmax_EQ = 2.10723
const u1_EQ   = 0.96511
const Drev_EQ = 0.20312
let lc = (:red, 0.55), lw = 0.5
    Makie.hlines!(ax1, [Drev_EQ]; color = lc, linestyle = :dot, linewidth = lw)
    Makie.hlines!(ax2, [λmax_EQ]; color = lc, linestyle = :dot, linewidth = lw)
    Makie.hlines!(ax3, [u1_EQ];   color = lc, linestyle = :dot, linewidth = lw)
end

# Annotate the equilibrium reference on panel B only.
Makie.text!(ax2, 0.03, λmax_EQ;
    text = "equilibrium", fontsize = FS-2, color = :orange,
    align = (:left, :bottom), offset = (0, 1))

# Vertical reference at t = 3.47 across all panels
for ax in (ax1, ax2, ax3)
    Makie.vlines!(ax, [3.47];
        color = :blue, linestyle = :dash, linewidth = 0.5)
end

# Panel labels A / B / C
for (ax, lbl) in zip((ax1, ax2, ax3), ("A", "B", "C"))
    Makie.text!(ax, 0.03, 0.94; space = :relative,
        text = lbl, fontsize = FS+1, font = :bold,
        align = (:left, :top), color = :black)
end

Makie.rowgap!(fig.layout, 1, 1)   # KL ↔ λ
Makie.rowgap!(fig.layout, 2, 1)   # λ  ↔ u

outpdf = joinpath(FIG_OUT, "fig_ooe_wrap.pdf")
outpng = joinpath(FIG_OUT, "fig_ooe_wrap.png")
Makie.save(outpdf, fig; pdf_version="1.4")
Makie.save(outpng, fig; px_per_unit=4)
@info "Saved Figure 6" pdf=outpdf png=outpng
