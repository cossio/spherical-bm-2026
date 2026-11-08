# 2×2 grid of K=2 phase diagrams at four values of c₁, showing how the
# phase topology evolves as the signal asymmetry grows.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import CairoMakie, Makie
using Makie: @L_str

figdir = @__DIR__

# ─── phase region classification ──────────────────

gcollision_k2_d1(γ::Real, χ2::Real) = (χ2 + sqrt(χ2^2 + 8γ)) / 4

function phase_region_k2(γ::Real, η::Real; χ1::Real=1.3, tol::Real=1e-10)
    χ2 = 2.0 - χ1
    γ >= 0 && η >= 0 || return 0
    c = γ*η
    sqrt_c = sqrt(max(c, 0.0)); sqrt_γ = sqrt(max(γ, 0.0))
    ηχ1 = η*χ1; ηχ2 = η*χ2
    gcoll = gcollision_k2_d1(γ, χ2)
    le(x, y) = x <= y + tol; lt(x, y) = x < y - tol; ge(x, y) = x >= y - tol
    if     ge(c, max(1.0, ηχ1^2));                             return 1
    elseif ge(c, ηχ1^2) && le(c, 1.0);                         return 2
    elseif lt(ηχ2, sqrt_c) && lt(sqrt_c, min(1.0, ηχ1)) &&
           lt((2*η-1)*sqrt_c + ηχ1, 2*η);                      return 3
    elseif lt(sqrt_c, min(1.0, ηχ2)) && lt(η, 1.0);            return 4
    elseif ge(c, ηχ1) && le(c, ηχ1^2)                          # PMo: γ ≥ c_1, mode 1 outlier
        # Split by whether mode 2 also detaches (γ < η c_2^2 ⇔ c_2 > √(γ/η)).
        return le(γ, η*χ2^2) ? 8 : 5
    elseif ge(gcoll, γ/χ1) &&
           le(gcoll, min(1.0, ηχ1, γ/χ2, sqrt_c));             return 6
    elseif ge(sqrt_γ, γ/χ2) && le(sqrt_γ, min(1.0, ηχ2, sqrt_c)); return 7
    else;                                                      return 0
    end
end

function phase_regions_grid_k2(; χ1=1.3, γlims=(0.0, 2.0), ηlims=(0.0, 2.0),
                                 n=1200, tol=1e-10)
    γs = range(max(γlims[1], 1e-6), γlims[2]; length=n)
    ηs = range(max(ηlims[1], 1e-6), ηlims[2]; length=n)
    regions = Matrix{Int}(undef, length(γs), length(ηs))
    @inbounds for i in eachindex(γs), j in eachindex(ηs)
        regions[i, j] = phase_region_k2(γs[i], ηs[j]; χ1, tol)
    end
    γs, ηs, regions
end

const OUTLIER_REGIONS = Set([5, 6, 7, 8])
const TEAL = Makie.RGBf(0.0, 0.45, 0.43)

# Label text for each phase region (main label, optional subtitle), expressed
# in the (h, u_1, u_2) order parameters used in Fig. 1B. Outlier vs. edge is
# carried by the teal contour, so the labels themselves do not repeat it.
const REGION_LABELS = Dict(
    1 => (L"h = 0",         L"u_1 = u_2 = 0"), # PMe
    2 => (L"h \ne 0",       L"u_1 = u_2 = 0"), # SG
    3 => (L"h,u_1 \ne 0",   L"u_2 = 0"),       # FMe, d=1
    4 => (L"h,u_1 \ne 0",   L"u_2 \ne 0"),     # FMe, d=2
    5 => (L"h = 0",         L"u_1 \ne 0,u_2 = 0"), # PMo, mode 1 outlier
    6 => (L"h,u_1 \ne 0",   L"u_2 = 0"),       # FMo, d=1
    7 => (L"h,u_1 \ne 0",   L"u_2 \ne 0"),     # FMo, d=2
    8 => (L"h = 0",         L"u_1,u_2 \ne 0"), # PMo, both modes outlier
)

# Pole of inaccessibility: for each region, find the (downsampled) grid point
# farthest from any non-region point (i.e., deepest inside the region). This
# keeps labels clear of phase boundaries even when the region is concave.
function pole_of_inaccessibility(γs, ηs, regions, r; stride=20)
    nγ, nη = length(γs), length(ηs)
    γlo, γhi = extrema(γs); ηlo, ηhi = extrema(ηs)
    cand = [(i,j) for i in 1:stride:nγ, j in 1:stride:nη]
    non_pts = [(γs[i], ηs[j]) for (i,j) in cand if regions[i,j] != r]
    in_pts  = [(i,j) for (i,j) in cand if regions[i,j] == r]
    (isempty(non_pts) || isempty(in_pts)) && return nothing
    best_d² = -1.0; best = (0.0, 0.0)
    for (i,j) in in_pts
        γ, η = γs[i], ηs[j]
        # Treat axis edges as boundaries so labels stay inside the panel.
        dmin² = min((γ - γlo)^2, (γhi - γ)^2, (η - ηlo)^2, (ηhi - η)^2)
        for (γn, ηn) in non_pts
            d² = (γ-γn)^2 + (η-ηn)^2
            d² < dmin² && (dmin² = d²)
        end
        if dmin² > best_d²
            best_d² = dmin²; best = (γ, η)
        end
    end
    return (best..., sqrt(best_d²))
end

function label_regions!(ax, γs, ηs, regions; fs=14, fssub=13, offset=0.06,
                        min_radius=0.12, stack_radius=0.0,
                        skip::Set{Int}=Set{Int}(),
                        y_shift::Dict{Int,Float64}=Dict{Int,Float64}())
    for (r, (main, sub)) in REGION_LABELS
        r ∈ skip && continue
        res = pole_of_inaccessibility(γs, ηs, regions, r)
        res === nothing && continue
        γ0, η0, radius = res
        radius < min_radius && continue
        η0 += get(y_shift, r, 0.0)
        if sub === nothing || radius < stack_radius
            Makie.text!(ax, γ0, η0; text=main, fontsize=fs, align=(:center,:center))
        else
            Makie.text!(ax, γ0, η0 + offset; text=main, fontsize=fs, align=(:center,:center))
            Makie.text!(ax, γ0, η0 - offset; text=sub,  fontsize=fssub, align=(:center,:center))
        end
    end
end

function draw_panel!(fig, row, col, χ1; first_col=false, last_row=false,
                     show_labels=false,
                     skip_label_regions::Set{Int}=Set{Int}(),
                     y_shift::Dict{Int,Float64}=Dict{Int,Float64}())
    γs, ηs, regions = phase_regions_grid_k2(; χ1, n=1200)
    c1r = round(χ1; digits=2)
    ax = Makie.Axis(fig[row, col];
        xlabel = last_row ? L"\gamma" : "",
        xticklabelsvisible = last_row,
        ylabel = first_col ? L"\eta" : "",
        yticklabelsvisible = first_col,
        title = L"c_1 = %$c1r",
        titlesize = 18,
        xlabelsize = 18, ylabelsize = 18,
        xticklabelsize = 14, yticklabelsize = 14,
        width=340, height=340,
        limits=(0.0, 2.0, 0.0, 2.0),
        xticks=0.0:0.5:2.0, yticks=0.0:0.5:2.0,
        xgridvisible=false, ygridvisible=false)
    for region in 1:8
        mask = Float64.(regions .== region)
        if any(>(0), mask) && any(<(1), mask)
            Makie.contour!(ax, γs, ηs, mask; levels=[0.5], color=:black, linewidth=1.3)
        end
    end
    outlier_mask = Float64.([r ∈ OUTLIER_REGIONS for r in regions])
    if any(>(0), outlier_mask) && any(<(1), outlier_mask)
        Makie.contour!(ax, γs, ηs, outlier_mask; levels=[0.5], color=TEAL, linewidth=2.8)
    end
    show_labels && label_regions!(ax, γs, ηs, regions; skip=skip_label_regions,
                                  y_shift=y_shift)
    ax
end

# ─── build figure ──────────────────────────────────────────────────────

χ1_values = [1.1, 1.3, 1.5, 1.7]

# 2 rows × 2 cols, larger overall figure.
const NCOLS = 2
fig = Makie.Figure(; size=(800, 800))

for (i, χ1) in enumerate(χ1_values)
    row = (i - 1) ÷ NCOLS + 1
    col = (i - 1) %  NCOLS + 1
    skip = if χ1 == 1.1
        Set([3, 6])    # FMe d=1, FMo d=1 narrow strips → manual rotated
    elseif χ1 == 1.5
        Set([7])       # FMo d=2 narrow wedge → manual vertical
    elseif χ1 == 1.7
        Set([5])       # PMo narrow strip → manual vertical
    else
        Set{Int}()
    end
    # Nudge labels off neighbouring boundary curves in the auto-labelled
    # panels.
    yshift = if χ1 == 1.1
        Dict(7 => -0.10)   # push FMo d=2 label down, away from FMe d=1 strip

    elseif χ1 == 1.5
        # PMe (region 1) sits naturally lower in this panel; no shift.
        Dict(5 => 0.04, 6 => 0.06, 7 => 0.04)
    else
        Dict(1 => 0.08, 5 => 0.04, 6 => 0.06, 7 => 0.04)
    end
    ax = draw_panel!(fig, row, col, χ1;
                     first_col = (col == 1), last_row = (row == 2),
                     show_labels = true,
                     skip_label_regions = skip, y_shift = yshift)
    if χ1 == 1.1
        Makie.text!(ax, 0.70, 0.70; text=L"h,u_1\ne 0,\ u_2 = 0",
            fontsize=11, align=(:center,:center), rotation=π/4)  # FMe d=1
        Makie.text!(ax, 1.05, 1.20; text=L"h,u_1\ne 0,\ u_2 = 0",
            fontsize=12, align=(:center,:center), rotation=π/2)  # FMo d=1
    elseif χ1 == 1.5
        Makie.text!(ax, 0.18, 1.40; text=L"h,u_1,u_2 \ne 0",
            fontsize=12, align=(:center,:center), rotation=π/2)  # FMo d=2
    elseif χ1 == 1.7
        Makie.text!(ax, 1.85, 1.00; text=L"h = 0,\ u_1 \ne 0,\ u_2 = 0",
            fontsize=12, align=(:center,:center), rotation=π/2)  # PMo
    end
end

Makie.resize_to_layout!(fig)

out = joinpath(figdir, "fig_phase_diagram_k=2_appendix")
Makie.save(out * ".pdf", fig)
Makie.save(out * ".png", fig; px_per_unit=3)
@info "Saved $(out).{pdf,png}"
