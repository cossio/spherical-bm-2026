#!/usr/bin/env julia
# fig:dynamics — three-panel figure.
# Panels A/B: γ=0.4, η=10, β=1, ν=0.7, c=(1.7, 0.3), s₀=0.1.
# Panel C: γ ∈ {0.8, 0.85, 0.9, 0.95, 0.99}, η=3, β=1, c=1; dashed line
# at the ν→∞ static asymptote h·u₁ from the K=1 coalesced saddle.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

using JLD2
import CairoMakie, Makie
using Makie: @L_str
using Printf, Statistics

const SCRIPT_DIR  = @__DIR__
const DATA_DIR    = joinpath(SCRIPT_DIR, "data")
const FIG_OUT_DIR = SCRIPT_DIR

# Panels A, B — produced by simulate/run_FM_trajectories.jl
fm_file_finN = joinpath(DATA_DIR, "finN_N3000_seed42_eig_nu0p7.jld2")
fm_file_msr  = joinpath(DATA_DIR, "msr_FM_nu0p7.jld2")
isfile(fm_file_finN) || error("Missing $fm_file_finN — run simulate/run_FM_trajectories.jl")
isfile(fm_file_msr)  || error("Missing $fm_file_msr — run simulate/run_FM_trajectories.jl")
finN_eig = load(fm_file_finN)
msr      = load(fm_file_msr)

# Panel C — stationary s_∞(ν) at five γ values (γ ∈ {0.8, 0.85, 0.9, 0.95, 0.99}).
# Produced by simulate/s_vs_nu_clean.jl (γ ∈ {0.8, 0.85, 0.9}) plus
# simulate/s_vs_nu_g0p95_clean.jl and simulate/s_vs_nu_g0p99_clean.jl.
snu_file_g099 = joinpath(DATA_DIR, "20260505_s_vs_nu_g0p99_clean.jld2")
snu_file_g095 = joinpath(DATA_DIR, "20260505_s_vs_nu_g0p95_clean.jld2")
snu_file_g09  = joinpath(DATA_DIR, "20260505_s_vs_nu_g0p9_clean.jld2")
snu_file_g085 = joinpath(DATA_DIR, "20260505_s_vs_nu_g0p85_clean.jld2")
snu_file_g08  = joinpath(DATA_DIR, "20260505_s_vs_nu_g0p8_clean.jld2")
for f in (snu_file_g099, snu_file_g095, snu_file_g09, snu_file_g085, snu_file_g08)
    isfile(f) || error("Missing $f")
end
snu_g099 = load(snu_file_g099)
snu_g095 = load(snu_file_g095)
snu_g09  = load(snu_file_g09)
snu_g085 = load(snu_file_g085)
snu_g08  = load(snu_file_g08)

# Panel D — MAP-limit phase diagram (unchanged)
map_file = joinpath(USBM_DATA, "20260416_map_phase_diagram_hires.jld2")
isfile(map_file) || error("Missing $map_file")
map_data = jldopen(map_file, "r")
γ_path_map  = map_data["γ_path"]
νc_path_map = map_data["νc_path"]
β_map       = map_data["β"]
close(map_data)

# Physical parameters for panels A, B
const γ_val = 0.4
const η_val = 10.0
const β_val = 1.0
const c_vec = [1.7, 0.3]                           # NEW: Tr C = 2 = K
const ν_FM  = 0.7
const Tmax  = 25.0
const N_EIG = 3000

# Analytic predictions
t_out(c_a) = c_a > sqrt(γ_val/η_val) ?
             -(2/γ_val) * log(1 - sqrt(γ_val/η_val) / c_a) : NaN
θ_pm(t, c_a)  = (c_a / γ_val) * (1 - exp(-γ_val * t / 2))
λ_para(t, c_a) = (th = θ_pm(t, c_a); th + 1 / (γ_val * η_val * th))
const edge_val = 2 / sqrt(γ_val * η_val)
t_detach_1 = t_out(c_vec[1])
t_detach_2 = t_out(c_vec[2])

function load_seed_set(Nval, seeds)
    curves = Vector{Vector{Float64}}()
    tvec = nothing
    for s in seeds
        f = joinpath(NEW_DATA, "finN_N$(Nval)_seed$(s)_nu0p7.jld2")
        isfile(f) || continue
        d = load(f)
        tvec === nothing && (tvec = Vector{Float64}(d["t"]))
        push!(curves, d["s"][1, :] .^ 2)
    end
    return tvec, hcat(curves...)
end

seed_sets = [
    (1500,  900:909),
    (6000,  1000:1009),
    (12000, 1100:1109),
]

# ─────────────────────────────────────────────────────────────────────
const FS = 18
fig = Makie.Figure(size=(1280, 320), fontsize=FS)
ng = (; xgridvisible=false, ygridvisible=false,
      xlabelsize=FS+3, ylabelsize=FS+3,
      xticklabelsize=FS, yticklabelsize=FS)

# ── Panel A: eigenvalues of W(t) ─────────────────────────────────────
axA = Makie.Axis(fig[1, 1]; ng...,
    xlabel = L"t",
    ylabel = L"\lambda_i(t)",
    xticks = 0:5:Int(Tmax),
    xminorticksvisible = true,
)

t_finN = Vector{Float64}(finN_eig["t"])
λ_finN = Matrix{Float64}(finN_eig["eig"]["lambda"])
λ_top_max = maximum(λ_finN[1, :])
y_top = max(ceil(λ_top_max * 1.05; digits=1),
            ceil(λ_para(Tmax, c_vec[1]) * 1.02; digits=1))
y_bot = 0.85

Makie.hlines!(axA, [edge_val]; color=:steelblue, linestyle=:dash, linewidth=1.2)

for r in reverse(axes(λ_finN, 1))
    is_top = r == 1
    Makie.lines!(axA, t_finN, λ_finN[r, :];
        color = is_top ? :black : (:gray60, 0.7),
        linewidth = is_top ? 1.6 : 0.7)
end

t_grid = range(0, Tmax; length=400)
for ca in c_vec
    ta = t_out(ca)
    isfinite(ta) || continue
    mask = t_grid .>= ta
    Makie.lines!(axA, t_grid[mask], λ_para.(t_grid[mask], ca);
        color=:crimson, linestyle=:dot, linewidth=2.0)
end

Makie.vlines!(axA, [t_detach_1]; color=:seagreen, linestyle=:dash, linewidth=1.5)
Makie.vlines!(axA, [t_detach_2]; color=:darkorange, linestyle=:dash, linewidth=1.5)

# Place t_out and t_* labels above the negative-phase pullback
# (~y_label > top of finite-N trajectory)
λ_top_max = maximum(λ_finN[1, :])
y_label_t_out = max(λ_top_max + 0.55, edge_val + 0.80*(y_top - edge_val))
# t_{out,1}, t_{out,2} as horizontal labels just above the top frame of
# panel A. Place them on `axA.blockscene` (parent scene, no axis clipping)
# at pixel coordinates derived from the live axis viewport.
let viewport = axA.scene.viewport, lims = axA.finallimits
    for (tdetach, txt, col) in [(t_detach_1, L"t_{\mathrm{out},1}", :seagreen),
                                 (t_detach_2, L"t_{\mathrm{out},2}", :darkorange)]
        pos = Makie.lift(viewport, lims) do vp, lim
            x_frac = (tdetach - lim.origin[1]) / lim.widths[1]
            x_pix = vp.origin[1] + x_frac * vp.widths[1]
            y_pix = vp.origin[2] + vp.widths[2] + 4
            Makie.Point2f(x_pix, y_pix)
        end
        Makie.text!(axA.blockscene, pos;
            text=txt, fontsize=FS+1, color=col,
            align=(:center, :bottom))
    end
end

# Recovery time t_* from the DMFT trace
const s₀_label = 0.1
const t_star_rec = let
    s1 = msr["s"][1, :]
    idx_dip = argmin(s1)
    idx_rec = findfirst(i -> i > idx_dip && s1[i] >= s₀_label, eachindex(s1))
    idx_rec === nothing ? NaN : Vector{Float64}(msr["t"])[idx_rec]
end

if isfinite(t_star_rec)
    Makie.vlines!(axA, [t_star_rec]; color=:purple, linestyle=:dash, linewidth=1.5)
    # Horizontal label, placed inside the panel just below the top frame
    # (above the t_out labels would conflict — those are outside the frame).
    Makie.text!(axA, t_star_rec - 0.40, y_top - 0.10; text=L"t_{\star}",
        fontsize=FS+1, color=:purple, align=(:right, :top))
end

Makie.xlims!(axA, 0, Tmax)
Makie.ylims!(axA, y_bot, y_top)

# Inline curve annotations
λ_para_top = λ_para(Tmax, c_vec[1])
Makie.text!(axA, 16.0, λ_para_top - 0.25; text=L"\text{early } t",
    fontsize=FS-1, color=:crimson, align=(:center, :top))
λ_top_late = mean(λ_finN[1, end-10:end])
Makie.text!(axA, 18.0, λ_top_late - 0.10; text=L"\lambda_i(t),\ N{=}%$(N_EIG)",
    fontsize=FS-1, color=:black, align=(:center, :top))
# "2σ" outside the right of panel A at the level of the blue edge line.
# Use a Makie.Label placed in the Right() sub-cell of axA, with valign
# nudged down so the label's vertical center sits ON the dashed line.
ylim_lo, ylim_hi = y_bot, y_top
v_align_2sigma = (edge_val - ylim_lo) / (ylim_hi - ylim_lo) - 0.025
Makie.Label(fig[1, 1, Makie.Right()], L"2\sigma";
    fontsize=FS-1, color=:steelblue,
    halign=:left, valign=v_align_2sigma,
    padding=(3, 0, 0, 0),
    tellwidth=false, tellheight=false)

# ── Panel B: s_1(t)² DMFT vs finite-N averages ────────────────────────
axB = Makie.Axis(fig[1, 2]; ng...,
    xlabel = L"t",
    ylabel = L"s_1(t)^2",
    xticks = 0:5:Int(Tmax),
    xminorticksvisible = true,
)

cmap_N = let g = Makie.cgrad(:Blues, length(seed_sets) + 2; categorical=true, rev=true)
    [g[i] for i in 1:length(seed_sets)]
end
t_msr = Vector{Float64}(msr["t"])
s_msr1 = msr["s"][1, :]
col_msr = Makie.RGBf(0.85, 0.15, 0.20)

for (k, (Nv, seeds)) in enumerate(seed_sets)
    tN, SN = load_seed_set(Nv, seeds)
    nseeds = size(SN, 2)
    nseeds == 0 && continue
    avg = vec(mean(SN; dims=2))
    se  = vec(std(SN;  dims=2)) ./ sqrt(nseeds)
    col = cmap_N[length(seed_sets)-k+1]
    Makie.band!(axB, tN, avg .- se, avg .+ se; color=(col, 0.20))
    Makie.lines!(axB, tN, avg; color=col, linewidth=1.5,
        label=L"N{=}%$(Nv)")
end

Makie.lines!(axB, t_msr, s_msr1 .^ 2; color=col_msr, linewidth=2.0, linestyle=:dash,
    label="DMFT")

# t_* on B
if isfinite(t_star_rec)
    Makie.vlines!(axB, [t_star_rec]; color=:purple, linestyle=:dash, linewidth=1.5)
    s2_top = maximum(s_msr1 .^ 2)
    Makie.text!(axB, t_star_rec - 0.3, 0.95 * s2_top; text=L"t_{\star}",
        fontsize=FS+1, color=:purple, align=(:right, :top))
end

Makie.xlims!(axB, 0, Tmax)
Makie.axislegend(axB; position=:rb, framevisible=false, labelsize=FS-2,
    rowgap=1, patchsize=(12, 2), margin=(0, -5, 0, 0))

# ── Panel C: s_∞(ν) at γ ∈ {0.7, 0.8, 0.9} ────────────────────────────
const η_C, c1_C = 3.0, 1.0
function hu1_asymptote(γ)
    g1 = sqrt(γ)                              # K=1, c_1=1
    return sqrt(max(0.0, 1 - g1) * max(0.0, 1 - g1 / (η_C * c1_C)))
end

# Red (low γ) → dark blue (high γ) gradient for γ = 0.8 → 0.85 → 0.9 → 0.95 → 0.99.
# The dark-blue endpoint is matched to the deepest tone of the Blues map used
# previously, so γ=0.99 keeps its prior color.
γ_list = [0.8, 0.85, 0.9, 0.95, 0.99]
snu_list = [snu_g08, snu_g085, snu_g09, snu_g095, snu_g099]
cmap_C = let blue_hi = Makie.cgrad(:Blues, 7; categorical=true)[7],
            red_lo  = Makie.RGBf(0.75, 0.15, 0.15)
    g = Makie.cgrad([red_lo, blue_hi], length(γ_list); categorical=true)
    [g[i] for i in 1:length(γ_list)]
end
asy_list = [hu1_asymptote(γ) for γ in γ_list]
νc_list  = [Float64(d["ν_c"]) for d in snu_list]   # may contain NaN (γ=0.8: no PM region in [1,1000])

# Truncate the displayed ν range to skip the small-ν region where the
# γ ∈ {0.8, 0.85} stationary FM root has not yet stabilised (real feature
# of the solver — verified at fine grids in probe_dip_g0p8.jl). The dip is
# at ν ≲ 0.7; ν_min = 1.0 leaves a clean approach to the asymptote at all γ.
const ν_lo_C, ν_hi_C = 1.0, 1000.0

axC = Makie.Axis(fig[1, 3]; ng...,
    xlabel = L"\nu",
    ylabel = L"s_{\mathrm{st}}",
    xscale = log10,
    xminorticksvisible = true,
    yminorticksvisible = true,
)

# Curves and their h·u_1 asymptotes (color-matched, dashed)
for (i, (γ, d, col, asy, νc)) in enumerate(zip(γ_list, snu_list, cmap_C, asy_list, νc_list))
    ν_v = Vector{Float64}(d["ν"])
    s_v = Vector{Float64}(d["s_inf"])
    # Clamp solver artifacts below ν_c (static branch is exactly 0 there).
    # If ν_c is NaN (no PM region in displayed range), no clamp.
    νc_eff = isfinite(νc) ? νc : -Inf
    s_clamped = [ν_v[k] < νc_eff ? 0.0 : s_v[k] for k in eachindex(ν_v)]
    Makie.lines!(axC, ν_v, s_clamped; color=col, linewidth=1.8)
    Makie.hlines!(axC, [asy]; color=col, linestyle=:dash, linewidth=1.2)
end

# Inline γ=... annotations at the saturating right end of each curve.
# Each label sits below its curve / asymptote, color-matched. Left-aligned
# at ν_label_C so labels stack flush along a common left edge. Per-γ offset
# accounts for how close the curve is to its asymptote at ν_label_C.
const ν_label_C = 200.0
const y_offset_C = Dict(0.8  => -0.003,
                        0.85 => -0.003,
                        0.9  => -0.012,    # γ=0.9 curve still close to asy at ν=200
                        0.95 => -0.012,    # γ=0.95 likewise
                        0.99 => -0.025)
for (γ, col, asy) in zip(γ_list, cmap_C, asy_list)
    Makie.text!(axC, ν_label_C, asy + y_offset_C[γ]; text=L"\gamma{=}%$(γ)",
        fontsize=FS-2, color=col, align=(:left, :top))
end

Makie.xlims!(axC, ν_lo_C, ν_hi_C)
Makie.ylims!(axC, -0.015, 0.32)

# ── Panel D: MAP-limit phase diagram ──────────────────────────────────
axD = Makie.Axis(fig[1, 4]; ng...,
    xlabel = L"\gamma",
    ylabel = L"\nu",
    yscale = log10,
)

γ_plot_lo, γ_plot_hi = 0.6, 1.5
ν_plot_lo, ν_plot_hi = 1e-2, max(maximum(νc_path_map) * 5, 100.0)
γ_min_obs = minimum(γ_path_map)

γ_pm_right = collect(range(β_map, γ_plot_hi; length=40))
Makie.band!(axD, γ_pm_right,
    fill(ν_plot_lo, length(γ_pm_right)),
    fill(ν_plot_hi, length(γ_pm_right));
    color=(:steelblue, 0.25))
Makie.band!(axD, γ_path_map, fill(ν_plot_lo, length(γ_path_map)), νc_path_map;
    color=(:steelblue, 0.25))
Makie.band!(axD, γ_path_map, νc_path_map, fill(ν_plot_hi, length(γ_path_map));
    color=(:firebrick, 0.25))
γ_cond_left = collect(range(γ_plot_lo, γ_min_obs; length=20))
Makie.band!(axD, γ_cond_left,
    fill(ν_plot_lo, 20), fill(ν_plot_hi, 20);
    color=(:firebrick, 0.25))

γ_boundary = vcat(γ_path_map[1], γ_path_map, β_map)
ν_boundary = vcat(ν_plot_lo, νc_path_map, ν_plot_hi)
Makie.lines!(axD, γ_boundary, ν_boundary; color=:black, linewidth=1.8,
    label=L"\nu_c(\gamma)")
Makie.vlines!(axD, [β_map]; color=:black, linestyle=:dash, linewidth=1)

Makie.text!(axD, 0.72, 0.3;
    text=L"s_{\mathrm{st}} \neq 0", fontsize=FS,
    align=(:center, :center), color=(:black, 0.85))
Makie.text!(axD, 1.25, 10.0;
    text=L"\text{static } s_{\mathrm{st}} = 0", fontsize=FS,
    align=(:center, :center), color=(:black, 0.85))
Makie.text!(axD, 0.965, 0.5;
    text=L"\text{dyn. } s_{\mathrm{st}} = 0", fontsize=FS-2,
    align=(:center, :center), color=(:black, 0.85), rotation=π/2)

Makie.axislegend(axD; position=:rt, framevisible=false, labelsize=FS-2,
    rowgap=1, patchsize=(12, 2))
Makie.xlims!(axD, γ_plot_lo, γ_plot_hi)
Makie.ylims!(axD, ν_plot_lo, ν_plot_hi)

for (col, tag) in enumerate(("A)", "B)", "C)", "D)"))
    Makie.Label(fig[1, col, Makie.TopLeft()], tag;
        fontsize=FS+4, font=:bold, padding=(0, 5, 0, 0))
end

Makie.colgap!(fig.layout, 12)
Makie.resize_to_layout!(fig)

outpdf = joinpath(FIG_OUT_DIR, "fig_dynamics_combined_traceK_with_hu1.pdf")
outpng = joinpath(FIG_OUT_DIR, "fig_dynamics_combined_traceK_with_hu1.png")
Makie.save(outpdf, fig; pdf_version="1.4")
Makie.save(outpng, fig; px_per_unit=3)
println("Saved → $outpdf")
println("Saved → $outpng")
println("\nDiagnostics:")
println("  c_vec = $c_vec   (Tr C = $(sum(c_vec)))")
println("  t_out,1 = $(round(t_detach_1; digits=3))   t_out,2 = $(round(t_detach_2; digits=3))")
println("  λ_para,1(∞) = $(round(c_vec[1]/γ_val + 1/(η_val*c_vec[1]); digits=3))")
println("  λ_para,2(∞) = $(round(c_vec[2]/γ_val + 1/(η_val*c_vec[2]); digits=3))")
println("  panel A ylim = ($y_bot, $y_top)")
println("  t_* (DMFT recovery) = $(round(t_star_rec; digits=3))")
