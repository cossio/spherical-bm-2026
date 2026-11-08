# Double descent in the teacher-student SBM at ω*=2.5.
# A) Typical reverse KL vs γ for a grid of η. Red dashed = MAP limit
#    (η → ∞); black dashed = saddle-node slice η_*(ω*=2.5) ≈ 1.342.
# B) K=2 phase diagram in (γ, η) at c₁=1.6, c₂=0.4.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import CairoMakie, Makie
using Makie: @L_str
using UndersampledSphericalBMs2025
const TS = UndersampledSphericalBMs2025.TeacherStudent

figdir = @__DIR__

# ─── Teacher parameters (match fig:kl panel A) ─────────────────────────
ω       = 2.5
χ1      = 2 - 1/ω    # c₁ = 1.6
χ2      = 1/ω        # c₂ = 0.4
η_star  = 1.342      # saddle-node threshold at ω=2.5 (see fig_kls_selected.jl)

η_vals  = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
γ_sweep = range(0.05, 5.0; length=400)

function safe_kl(γ, η, ω)
    try; return TS.all_kl_divergences(γ, η, ω)
    catch; return (; fwd_typ=NaN, rev_typ=NaN, fwd_pp=NaN, rev_pp=NaN,
                     phase=:error, M=NaN, ξ=[NaN,NaN], g=[NaN,NaN], u=[NaN,NaN]); end
end

# ─── K=2 phase-region classifier (copied from fig_phase_diagram_k=2.jl) ─
gcollision_k2_d1(γ::Real, χ2::Real) = (χ2 + sqrt(χ2^2 + 8γ)) / 4

function phase_region_k2(γ::Real, η::Real; χ1::Real=χ1, tol::Real=1e-10)
    χ2 = 2.0 - χ1
    γ >= 0 && η >= 0 || return 0
    c = γ*η
    sqrt_c = sqrt(max(c, 0.0)); sqrt_γ = sqrt(max(γ, 0.0))
    ηχ1 = η*χ1; ηχ2 = η*χ2
    gcoll = gcollision_k2_d1(γ, χ2)
    le(x, y) = x <= y + tol; lt(x, y) = x < y - tol; ge(x, y) = x >= y - tol
    if     ge(c, max(1.0, ηχ1^2));                                return 1  # PMe
    elseif ge(c, ηχ1^2) && le(c, 1.0);                            return 2  # SG
    elseif lt(ηχ2, sqrt_c) && lt(sqrt_c, min(1.0, ηχ1)) &&
           lt((2*η-1)*sqrt_c + ηχ1, 2*η);                         return 3  # FMe d=1
    elseif lt(sqrt_c, min(1.0, ηχ2)) && lt(η, 1.0);               return 4  # FMe d=2
    elseif ge(c, ηχ1) && le(c, ηχ1^2);                            return 5  # PMo
    elseif ge(gcoll, γ/χ1) &&
           le(gcoll, min(1.0, ηχ1, γ/χ2, sqrt_c));                return 6  # FMo d=1
    elseif ge(sqrt_γ, γ/χ2) &&
           le(sqrt_γ, min(1.0, ηχ2, sqrt_c));                     return 7  # FMo d=2
    else;                                                         return 0
    end
end

function phase_regions_grid_k2(; χ1=χ1, γlims=(0.0, 2.0), ηlims=(0.0, 2.0),
                                 n=1200, tol=1e-10)
    γs = range(max(γlims[1], 1e-6), γlims[2]; length=n)
    ηs = range(max(ηlims[1], 1e-6), ηlims[2]; length=n)
    regions = Matrix{Int}(undef, length(γs), length(ηs))
    @inbounds for i in eachindex(γs), j in eachindex(ηs)
        regions[i, j] = phase_region_k2(γs[i], ηs[j]; χ1, tol)
    end
    γs, ηs, regions
end

const OUTLIER_REGIONS = Set([5, 6, 7])
const TEAL = Makie.RGBf(0.0, 0.45, 0.43)
const ORANGE = Makie.RGBf(0.90, 0.45, 0.10)  # shared color for the two analytic limits

const REGION_LABELS = Dict(
    1 => (L"\mathrm{PMe}", nothing),
    2 => (L"\mathrm{SG}",  nothing),
    3 => (L"\mathrm{FMe}", L"d=1"),
    4 => (L"\mathrm{FMe}", L"d=2"),
    5 => (L"\mathrm{PMo}", nothing),
    6 => (L"\mathrm{FMo}", L"d=1"),
    7 => (L"\mathrm{FMo}", L"d=2"),
)

# Pole of inaccessibility: deepest grid point inside each region (for labels).
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
        dmin² = min((γ-γlo)^2, (γhi-γ)^2, (η-ηlo)^2, (ηhi-η)^2)
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

function label_regions!(ax, γs, ηs, regions; fs=12, fssub=11, offset=0.09,
                        min_radius=0.10, stack_radius=0.22)
    for (r, (main, sub)) in REGION_LABELS
        res = pole_of_inaccessibility(γs, ηs, regions, r)
        res === nothing && continue
        γ0, η0, radius = res
        radius < min_radius && continue
        if sub === nothing || radius < stack_radius
            Makie.text!(ax, γ0, η0; text=main, fontsize=fs, align=(:center,:center))
        else
            Makie.text!(ax, γ0, η0 + offset; text=main, fontsize=fs, align=(:center,:center))
            Makie.text!(ax, γ0, η0 - offset; text=sub,  fontsize=fssub, align=(:center,:center))
        end
    end
end

# ─── Build figure ──────────────────────────────────────────────────────
# Single-panel layout for a NeurIPS wrapfigure: reverse-typical KL vs γ.
cmap = Makie.cgrad(:viridis, length(η_vals); categorical=true)

fig = Makie.Figure(size=(180, 140), fontsize=8)
ng  = (; xgridvisible=false, ygridvisible=false)
ytk = Makie.WilkinsonTicks(4)
xtk = Makie.WilkinsonTicks(4)

# Row 1: Panel A (KL) + colorbar + Panel B (phase diagram).
ax_a = Makie.Axis(fig[1, 1]; ng...,
    xlabel = L"\gamma",
    ylabel = L"D_\mathrm{KL}(P_W\,\Vert\,P^{\ast})/N",
    yticks = ytk, xticks = xtk,
    width = 120, height = 85)

# Mark FMo minimum (γ < c₁) on every curve below the saddle-node threshold.
γs_vec = collect(γ_sweep)
fmo_mask = γs_vec .< χ1
γs_fmo  = γs_vec[fmo_mask]

function fmo_min(ys)
    ys_fmo = ys[fmo_mask]
    i_min = argmin(replace(ys_fmo, NaN => Inf))
    return γs_fmo[i_min], ys_fmo[i_min]
end

mins_γ = Float64[]
mins_y = Float64[]
# Keep every curve's y values so we can place inline labels anywhere along it.
curves = Vector{Vector{Float64}}(undef, length(η_vals))
for (i, η) in enumerate(η_vals)
    ys = [safe_kl(γ, η, ω).rev_typ for γ in γ_sweep]
    curves[i] = ys
    Makie.lines!(ax_a, γs_vec, ys; color=cmap[i], linewidth=1.0)
    if η > η_star
        γm, ym = fmo_min(ys)
        push!(mins_γ, γm); push!(mins_y, ym)
    end
end
# MAP limit η → ∞ (orange dashed)
ys_map = [safe_kl(γ, 5000.0, ω).rev_typ for γ in γ_sweep]
Makie.lines!(ax_a, γs_vec, ys_map;
             color=ORANGE, linewidth=1.3, linestyle=:dash)
let (γm, ym) = fmo_min(ys_map)
    push!(mins_γ, γm); push!(mins_y, ym)
end
# FMo→PMo boundary γ = c₁
Makie.vlines!(ax_a, [χ1]; color=:black, linewidth=0.8, linestyle=:dash)
# "h≠0" / "h=0" phase markers flanking the boundary at the bottom of panel A.
let y_tag = 0.165, dx = 0.12
    Makie.text!(ax_a, χ1 - dx, y_tag; text = L"h\neq 0",
                color = :black, fontsize = 7, align = (:right, :bottom))
    Makie.text!(ax_a, χ1 + dx, y_tag; text = L"h=0",
                color = :black, fontsize = 7, align = (:left, :bottom))
end
# Saddle-node threshold slice η = η_*(ω=2.5) ≈ 1.342 (orange dashed, paired
# with the MAP limit as the two "critical" analytic anchors).
ys_star = [safe_kl(γ, η_star, ω).rev_typ for γ in γ_sweep]
Makie.lines!(ax_a, γs_vec, ys_star;
             color=ORANGE, linestyle=:dash, linewidth=1.3)
# Black dots at the FMo minima (drawn on top of the curves but below the labels)
Makie.scatter!(ax_a, mins_γ, mins_y;
               color=:black, markersize=5, strokewidth=0)
Makie.ylims!(ax_a, 0.15, 0.5)

# ─── Inline η labels ON each curve (matplotlib-label-lines style) ───────
# For each curve, pick a γ along it and place the η value rotated to match
# the curve's local slope (in screen coords, not data coords), with a thick
# white stroke so the line appears to pass "behind" the text.

# Screen→data scales, needed to convert local data slope to display slope.
panel_w_pt, panel_h_pt = 120, 85
γ_min_ax, γ_max_ax = γs_vec[1], γs_vec[end]
y_min_ax, y_max_ax = 0.15, 0.5
sx = panel_w_pt / (γ_max_ax - γ_min_ax)   # pt per γ unit
sy = panel_h_pt / (y_max_ax - y_min_ax)   # pt per KL unit

function label_on_curve!(ax, γ_target, ys, txt, color; fontsize=7, mask_chars=nothing)
    i = argmin(abs.(γs_vec .- γ_target))
    γ_l = γs_vec[i]
    y_l = ys[i]
    (y_min_ax < y_l < y_max_ax) || return
    # Local slope in data coords (central difference when possible)
    di = (i == 1) ? 1 : (i == length(ys) ? -1 : 1)
    dx = γs_vec[i + di] - γ_l
    dy = ys[i + di]    - y_l
    rot = atan(dy * sy, dx * sx)                     # slope in screen coords
    # Mask the line by painting a short white segment of the curve behind
    # the text, matching the curve's local direction and length. For LaTeX
    # strings, length(txt) over-counts (counts source chars, not glyphs);
    # callers can pass `mask_chars` to set the visible glyph count manually.
    char_w = fontsize * 0.6
    nchars = mask_chars === nothing ? length(string(txt)) : mask_chars
    half   = 1.05 * nchars * char_w / 2
    cosθ, sinθ = cos(rot), sin(rot)
    dxd, dyd = half / sx * cosθ, half / sy * sinθ
    Makie.lines!(ax, [γ_l - dxd, γ_l + dxd], [y_l - dyd, y_l + dyd];
                 color = :white, linewidth = fontsize + 1)
    Makie.text!(ax, γ_l, y_l; text = txt, color = color,
                fontsize = fontsize, align = (:center, :center),
                rotation = rot)
end

# Stagger the label γ positions along the axis so labels don't collide. The
# flatter a curve is at its chosen γ, the more readable the label gets.
label_γs = Dict(
    0.1 => 4.5, 0.2 => 4.4,     # off-chart curves; label_on_curve! will skip
    0.4 => 4.6, 0.7 => 4.3, 1.0 => 3.2,
    1.5 => 2.7, 2.0 => 2.2,
    3.0 => mins_γ[3] + 0.45,     # shifted right of the FMo-minimum disk
    5.0 => mins_γ[4] + 0.55,
)
for (i, η) in enumerate(η_vals)
    label_on_curve!(ax_a, label_γs[η], curves[i], string(η), cmap[i])
end
# Saddle-node threshold η_DD ≈ 1.342: label in the descending PMo branch.
# LaTeX rendering gives a proper "DD" subscript; mask_chars accounts for
# the narrower subscripts and parens — visible width is closer to 7 full
# glyphs than the 10 source characters would suggest.
label_on_curve!(ax_a, 4.5, ys_star, L"1.3\;(\eta_{\mathrm{DD}})", ORANGE;
                fontsize=7, mask_chars=7)
# MAP: label on the PMo flat, where the curve is horizontal.
label_on_curve!(ax_a, 3.7, ys_map, "∞ (MAP)", ORANGE; fontsize=7)

Makie.resize_to_layout!(fig)

out = joinpath(figdir, "fig_double_descent")
Makie.save(out * ".pdf", fig)
Makie.save(out * ".png", fig; px_per_unit=3)
@info "Saved $(out).{pdf,png}"
