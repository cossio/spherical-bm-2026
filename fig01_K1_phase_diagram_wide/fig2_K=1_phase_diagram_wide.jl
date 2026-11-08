# Fig. 2 — horizontal layout: A) 2×2 schematics, B) phase diagram, C) η=0.5, D) η=5
#
# Run from accompanying-code/fig01_K1_phase_diagram_wide/:
#     julia fig2_K=1_phase_diagram_wide.jl
#
# (First run requires `julia --project=../_julia_env ../_julia_env/setup.jl`
# to instantiate the shared environment.)

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import Makie, CairoMakie
using Makie: @L_str
using UndersampledSphericalBMs2025
const RMT = UndersampledSphericalBMs2025.RMT_Solution

wigner(λ::Real) = sqrt(max(0, 4 - λ^2)) / 2π

figures_dir = joinpath(@__DIR__, "..", "figures")

teal = Makie.RGBf(0.0, 0.55, 0.55)
fig = Makie.Figure(; figure_padding=(15, 10, 5, 5))
L = 3

# ═══ Panel B: Phase diagram (same as original) ═══
ax = Makie.Axis(fig[1, 2];
    width=220, height=220,
    xlabel=L"\gamma", ylabel=L"\eta",
    xlabelsize=20, ylabelsize=20,
    xticklabelsize=16, yticklabelsize=16,
    xgridvisible=false, ygridvisible=false,
    xticks=0:3, yticks=0:3)

Makie.lines!(ax, [0, 1], [1, 1]; linewidth=3.5, color=teal)
Makie.lines!(ax, [1, L], [1, L]; linewidth=3.5, color=teal)
Makie.lines!(ax, [0, 1], [0, 1]; linewidth=3.5, color=:black)
Makie.lines!(ax, [1, 1], [1, L]; linewidth=3.5, color=:black)
γs = range(1, L; length=500)
Makie.lines!(ax, collect(γs), inv.(collect(γs)); linewidth=3.5, color=:black)

Makie.text!(ax, 0.25, 2.3; text="FMo", fontsize=20)
Makie.text!(ax, 1.5, 2.7; text="PMo", fontsize=20)
Makie.text!(ax, 0.1, 0.65; text="FMe", fontsize=20)
Makie.text!(ax, 2.3, 1.4; text="PMe", fontsize=20)
Makie.text!(ax, 1.0, 0.25; text="SG", fontsize=20)

Makie.text!(ax, 1.55, 1.75; text="outlier", fontsize=18, color=teal,
    rotation=π/4, align=(:center, :bottom), font=:italic)
Makie.text!(ax, 1.8, 1.6; text="edge", fontsize=18, color=teal,
    rotation=π/4, align=(:center, :top), font=:italic)

Makie.xlims!(ax, 0, L)
Makie.ylims!(ax, 0, L)

# ═══ Panel A: Eigenvalue distribution diagrams (2×2, same texts as original) ═══
grid_b = fig[1, 1] = Makie.GridLayout()

configs = [
    # (row, col, title, λ₁, μ)
    (1, 1, "paramagnetic-outlier (PMo)",                        3.0, 4.5),
    (1, 2, "paramagnetic-edge (PMe)",                           2.0, 4.0),
    (2, 1, "ferromagnetic-outlier (FMo)",                       3.9, 4.3),
    (2, 2, "ferromagnetic-edge (FMe)\nor spin glass (SG)",      2.0, 2.1),
]

xs = collect(-2:0.01:2)
for (r, c, title, λ1, μ) in configs
    Makie.Label(grid_b[2r-1, c], title; fontsize=14, halign=:center, valign=:bottom)
    ax_ev = Makie.Axis(grid_b[2r, c]; width=180, height=65,
        xgridvisible=false, ygridvisible=false)
    Makie.band!(ax_ev, xs, zero.(xs), wigner.(xs); color=:lightblue)
    Makie.vlines!(ax_ev, λ1; color=:blue, linewidth=3)
    Makie.vlines!(ax_ev, μ; color=:red, linewidth=3, linestyle=:dash)
    Makie.vlines!(ax_ev, 0; color=:black, linewidth=1)
    Makie.hlines!(ax_ev, 0; color=:black, linewidth=1)
    Makie.xlims!(ax_ev, -2.5, 5.5)
    Makie.ylims!(ax_ev, -0.02, 0.4)
    Makie.hidedecorations!(ax_ev)
    Makie.hidespines!(ax_ev)
end
Makie.colgap!(grid_b, 15)
Makie.rowgap!(grid_b, 10)

# ═══ Panels C, D: λ₁ vs γ (same aspect ratio as original: wider than tall) ═══
function solve_K1(γ, η)
    c = γ * η
    return RMT.solve([η], η, c)
end

bulk_edge(γ, η) = 2 / sqrt(γ * η)

γ_max = 8.0
γ_range = range(0.05, γ_max; length=600)

function plot_eigval_panel!(fig_pos, η, phases; ylims_val=(0, 6), ylabel=L"\lambda_1")
    ax = Makie.Axis(fig_pos; width=220, height=200,
        xlabel=L"\gamma", ylabel=ylabel,
        xlabelsize=18, ylabelsize=18,
        xticklabelsize=14, yticklabelsize=14,
        xgridvisible=false, ygridvisible=false)

    # Bulk band
    upper = [bulk_edge(γ, η) for γ in γ_range]
    Makie.band!(ax, collect(γ_range), zero(upper), upper; color=(:lightblue, 0.5))

    # Phase boundary lines and labels
    for phase in phases
        label, γ_lo, γ_hi = phase[1], phase[2], phase[3]
        γ_pos = length(phase) ≥ 4 ? phase[4] : (max(γ_lo, 0.0) + min(γ_hi, γ_max)) / 2
        rot   = length(phase) ≥ 5 ? phase[5] : 0.0
        y_pos = ylims_val[1] + 0.8 * (ylims_val[2] - ylims_val[1])
        if γ_lo > 0
            Makie.vlines!(ax, γ_lo; color=:gray50, linewidth=1.2, linestyle=:dash)
        end
        Makie.text!(ax, γ_pos, y_pos;
            text=string(label), fontsize=16, align=(:center, :center),
            rotation=rot, color=:black)
    end

    # Eigenvalue and μ
    sols = [solve_K1(γ, η) for γ in γ_range]
    λ1 = [s.λ[1] for s in sols]
    μs = [s.μ for s in sols]
    Makie.band!(ax, [NaN], [NaN], [NaN]; color=(:lightblue, 0.5), label="bulk")
    Makie.lines!(ax, collect(γ_range), λ1; color=:blue, linewidth=2.5, label=L"\lambda_1")
    Makie.lines!(ax, collect(γ_range), μs; color=:red, linewidth=2.5, linestyle=:dash, label=L"\mu")

    Makie.xlims!(ax, 0, γ_max)
    Makie.ylims!(ax, ylims_val...)
    return ax
end

# Panel C: η=0.5
ax_c = plot_eigval_panel!(fig[1, 3], 0.5, [
    ("FMe", 0.0, 0.5, 0.2, π/2),
    ("SG",  0.5, 2.0, 1.5),
    ("PMe", 2.0, γ_max, 3.5),
]; ylims_val=(0, 4))
Makie.axislegend(ax_c; position=:rt, labelsize=12, framevisible=false, padding=(0,0,0,0))

# Panel D: η=5
ax_d = plot_eigval_panel!(fig[1, 4], 5, [
    ("FMo", 0.0, 1.0, 0.45, π/2),
    ("PMo", 1.0, 5.0),
    ("PMe", 5.0, γ_max),
]; ylims_val=(0, 4), ylabel="")
Makie.hideydecorations!(ax_d; ticks=false, grid=false)

# ═══ Panel labels ═══
Makie.Label(fig[1, 1, Makie.TopLeft()], "A)"; fontsize=18, font=:bold, halign=:left, padding=(0, 0, 0, 0))
Makie.Label(fig[1, 2, Makie.TopLeft()], "B)"; fontsize=18, font=:bold, halign=:left, padding=(0, 0, 0, 0))
Makie.Label(fig[1, 3, Makie.TopLeft()], "C)"; fontsize=18, font=:bold, halign=:left, padding=(0, 0, 0, 0))
Makie.Label(fig[1, 4, Makie.TopLeft()], "D)"; fontsize=18, font=:bold, halign=:left, padding=(0, 0, 0, 0))

Makie.colgap!(fig.layout, 5)
Makie.resize_to_layout!(fig)
Makie.save(joinpath(figures_dir, "fig2_K=1-phase-diagram-wide.pdf"), fig)
Makie.save(joinpath(figures_dir, "fig2_K=1-phase-diagram-wide.png"), fig; px_per_unit=3)
