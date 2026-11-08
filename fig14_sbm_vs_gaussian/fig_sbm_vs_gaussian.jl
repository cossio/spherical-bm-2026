# Publication figure: qualitative SBM vs Gaussian phase comparison
#
# K=1 spherical phase diagram with the Gaussian-accessible region shaded.
#
# Run: julia -t auto fig_sbm_vs_gaussian.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import CairoMakie, Makie
using Makie: @L_str
figures_dir = joinpath(@__DIR__, "reproduced")
mkpath(figures_dir)
const TEAL = Makie.RGBf(0.0, 0.55, 0.55)

# ── Figure ───────────────────────────────────────────────────────────

println("Building figure...")
fig = Makie.Figure(size=(430, 400); figure_padding=15)

# K=1 phase diagram

ax2 = Makie.Axis(fig[1, 1];
    width=330, height=300,
    xlabel=L"\gamma", ylabel=L"\eta",
    xgridvisible=false, ygridvisible=false,
    xticks=0:3, yticks=0:5)

L_γ = 3.5
L_η = 5.0

# Shade FM + SG region (uniquely spherical)
Makie.poly!(ax2,
    Makie.Point2f[(0, 0), (1, 0), (1, L_η), (0, L_η)];
    color=(:lightsalmon, 0.35))

γ_sg = collect(range(1, L_γ; length=300))
η_sg = 1 ./ γ_sg
Makie.poly!(ax2, vcat(
    [Makie.Point2f(1, 0)],
    [Makie.Point2f(g, e) for (g, e) in zip(γ_sg, η_sg)],
    [Makie.Point2f(L_γ, 0)]
); color=(:lightsalmon, 0.35))

# Shade PM region (Gaussian-accessible)
Makie.poly!(ax2, vcat(
    [Makie.Point2f(1, 1)],
    [Makie.Point2f(g, e) for (g, e) in zip(γ_sg, η_sg)],
    [Makie.Point2f(L_γ, 1/L_γ)],
    [Makie.Point2f(L_γ, L_η)],
    [Makie.Point2f(1, L_η)]
); color=(:lightblue, 0.25))

# Boundary lines — teal (outlier ↔ edge)
Makie.lines!(ax2, [0, 1], [1, 1]; linewidth=3, color=TEAL)
Makie.lines!(ax2, [1, L_γ], [1, L_γ]; linewidth=3, color=TEAL)

# Boundary lines — black (FM ↔ PM, SG)
Makie.lines!(ax2, [0, 1], [0, 1]; linewidth=3, color=:black)
Makie.lines!(ax2, [1, 1], [1, L_η]; linewidth=3, color=:black)
γ_dense = range(1, L_γ; length=500)
Makie.lines!(ax2, collect(γ_dense), inv.(collect(γ_dense)); linewidth=3, color=:black)

# Phase labels — (h, u_1) order-parameter notation matching Fig. 1
Makie.text!(ax2, 0.5, 3.0; text=L"h,u_1\ne 0", fontsize=18, align=(:center,:center))
Makie.text!(ax2, 2.0, 4.2; text=L"h=0", fontsize=16, align=(:center,:center))
Makie.text!(ax2, 2.0, 3.6; text=L"u_1\ne 0", fontsize=16, align=(:center,:center))
Makie.text!(ax2, 2.7, 1.3; text=L"h=u_1=0", fontsize=18, align=(:center,:center))
Makie.text!(ax2, 1.7, 0.42; text=L"h\ne 0", fontsize=13, align=(:center,:center))
Makie.text!(ax2, 1.7, 0.10; text=L"u_1=0", fontsize=13, align=(:center,:center))

# Annotation
Makie.text!(ax2, 2.3, 3.0; text="Gaussian\naccessible", fontsize=13,
    color=:steelblue, align=(:center, :center), font=:italic)
Makie.text!(ax2, 0.5, 2.0; text="uniquely\nspherical", fontsize=13,
    color=:firebrick, align=(:center, :center), font=:italic)
Makie.text!(ax2, 1.45, 1.65; text="outlier", fontsize=13, color=TEAL,
    rotation=π / 4, align=(:center, :bottom), font=:italic)
Makie.text!(ax2, 1.85, 1.55; text="edge", fontsize=13, color=TEAL,
    rotation=π / 4, align=(:center, :top), font=:italic)

Makie.xlims!(ax2, 0, L_γ)
Makie.ylims!(ax2, 0, L_η)

Makie.resize_to_layout!(fig)

Makie.save(joinpath(figures_dir, "fig_sbm_vs_gaussian.pdf"), fig)
Makie.save(joinpath(figures_dir, "fig_sbm_vs_gaussian.png"), fig; px_per_unit=3)
println("Saved to: $(joinpath(figures_dir, "fig_sbm_vs_gaussian.{pdf,png}"))")
