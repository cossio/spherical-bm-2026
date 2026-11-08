# PMo temperature tuning. A) Cartoon of the β-rescaled bulk + outlier.
# B) Forward typical KL vs β for three (γ, η) anchors at η=5.0, ω*=2.5.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import CairoMakie, Makie
using Makie: @L_str
using UndersampledSphericalBMs2025
const TS = UndersampledSphericalBMs2025.TeacherStudent
const RMT = UndersampledSphericalBMs2025.RMT_Solution
const RMT2 = UndersampledSphericalBMs2025.RMT_Solution_v2

# ── Sampling saddle (mirror of generate_temperature_tuning.jl) ──────

function sampling_saddle(γ, η, ω, β)
    c1, c2 = TS.teacher_c(ω)
    c_rmt = γ * η
    c_eff = c_rmt / β^2
    sol = RMT2.solve([c1, c2], γ, η)
    λ_top = sol.λ[1]
    βλ_top = β * λ_top
    if c_eff ≥ 1
        μ_pm = RMT.stieltjes_inverse(1.0, c_eff)
        if βλ_top > μ_pm
            μ̃ = βλ_top
            g̃ = RMT.stieltjes(μ̃, c_eff)
            phase = :sFM
        else
            μ̃ = μ_pm; g̃ = 1.0; phase = :sPM
        end
    else
        μ̃ = βλ_top
        g̃ = RMT.stieltjes(μ̃, c_eff)
        phase = :sSG
    end
    return (; μ̃, g̃, m² = max(0.0, 1 - g̃), c_eff, λ_top, βλ_top, phase, sol)
end

function fwd_typ_beta(γ, η, ω, β)
    ss = sampling_saddle(γ, η, ω, β)
    avg_E = TS.avg_energy_teacher_in_student(ss.sol.ξ[1], η, ω)
    F1β = TS.S_sc(ss.μ̃, ss.c_eff) / 2
    return β * avg_E + F1β - TS.teacher_entropy_noA(ω)
end

beta_condensation_onset(γ, η, ω) = (sol = RMT2.solve([TS.teacher_c(ω)...], γ, η);
                                    sol.u[1] ≤ 0 ? NaN : sol.g[1])
β_sFM_sSG(γ, η) = sqrt(γ * η)

# ── Figure ────────────────────────────────────────────────────────

fig = Makie.Figure(; figure_padding=(8, 14, 4, 8), fontsize=11)

# ─── Panel A: PMo cartoon with TT overlay ──────────────────────────
# Replicates exactly the orange overlay block from
# scripts/fig2_K=1_phase_diagram_wide.jl.

wigner(λ) = sqrt(max(0, 4 - λ^2)) / 2π

panel_a = fig[1, 1] = Makie.GridLayout()
ax_a = Makie.Axis(panel_a[1, 1]; width=160, height=50,
                  xgridvisible=false, ygridvisible=false)

# PMo configuration (same numerical values as Fig. 1 panel B):
λ1_a = 3.0
μ_a  = 4.5

# Temperature-tuning overlay (β > β_c rescales the bulk to a wider
# semicircle and pushes the outlier from λ₁ → β λ₁; the spherical
# saddle then pins the new top, marking the sFM regime).
β_tune    = 1.6
xs        = collect(-2:0.01:2)
xs_beta   = collect(-2*β_tune:0.01:2*β_tune)
wigner_β(λ) = sqrt(max(0, 4*β_tune^2 - λ^2)) / (2π * β_tune^2)
tune_color  = Makie.RGBf(0.90, 0.55, 0.15)

Makie.band!(ax_a, xs_beta, zero.(xs_beta), wigner_β.(xs_beta);
            color=(tune_color, 0.25))
Makie.lines!(ax_a, [β_tune * λ1_a, β_tune * λ1_a], [0.0, 0.42];
             color=tune_color, linewidth=2)
arrow_y  = 0.13
arrow_x0 = λ1_a + 0.25
arrow_x1 = β_tune * λ1_a - 0.35
Makie.lines!(ax_a, [arrow_x0, arrow_x1], [arrow_y, arrow_y];
             color=tune_color, linestyle=:dash, linewidth=0.9)
Makie.scatter!(ax_a, [arrow_x1], [arrow_y];
               color=tune_color, marker=:rtriangle, markersize=7, strokewidth=0)
Makie.text!(ax_a, (arrow_x0 + arrow_x1)/2, arrow_y + 0.02;
            text="TT", fontsize=10, color=tune_color, align=(:center, :bottom))
Makie.text!(ax_a, β_tune * λ1_a + 0.15, 0.38;
            text=L"\beta\lambda_1", fontsize=11, color=tune_color,
            align=(:left, :top))

Makie.band!(ax_a, xs, zero.(xs), wigner.(xs); color=:lightblue)
Makie.lines!(ax_a, [λ1_a, λ1_a], [0.0, 0.42]; color=:blue, linewidth=2.5)
Makie.lines!(ax_a, [μ_a,  μ_a],  [0.0, 0.42]; color=:red,  linewidth=2.5, linestyle=:dash)
Makie.text!(ax_a, λ1_a, 0.43;
            text=L"\lambda_1", fontsize=11, color=:blue,
            align=(:center, :bottom))
Makie.text!(ax_a, μ_a, 0.43;
            text=L"\mu", fontsize=11, color=:red,
            align=(:center, :bottom))
Makie.vlines!(ax_a, 0;    color=:black, linewidth=1)
Makie.hlines!(ax_a, 0;    color=:black, linewidth=1)
Makie.xlims!(ax_a, -2.2, 5.7)
Makie.ylims!(ax_a, -0.02, 0.55)
Makie.hidedecorations!(ax_a)
Makie.hidespines!(ax_a)

# ─── Panel B: fwd_typ KL vs β at η=5, three γ values stacked ──────
# Same anchors as the η=5 row of report Fig. 7. Stacked one above the
# other (shared β axis); single colour across all three; γ labelled
# inside each subpanel.

const ω = 2.5
const η = 5.0
const γS = [1.7, 2.0, 3.0]
const NB = 600
β_grid = collect(range(0.3, 5.0; length=NB))
curve_color = Makie.RGBf(0.15, 0.25, 0.55)  # single dark navy for all γ

println("PMo η=$η anchors at ω*=$ω:")

panel_b = fig[2, 1] = Makie.GridLayout()

b_axes = Makie.Axis[]
for (i, γ) in enumerate(γS)
    is_bottom = (i == length(γS))
    ax = Makie.Axis(panel_b[i, 1];
                    width=160, height=38,
                    xgridvisible=false, ygridvisible=false,
                    xlabel = is_bottom ? L"\beta" : "",
                    ylabel = i == 2 ?
                        L"D_{\mathrm{KL}}(P^{\ast}\,\Vert\,P_{\beta W})/N" : "",
                    yticks = Makie.WilkinsonTicks(2; k_min=2, k_max=3),
                    xticks = 0:1:5,
                    yticklabelsize = 8, xticklabelsize = 8,
                    xlabelsize = 10, ylabelsize = 10,
                    xlabelpadding = 2, ylabelpadding = 10)
    push!(b_axes, ax)

    fwd  = [fwd_typ_beta(γ, η, ω, β) for β in β_grid]
    βc   = beta_condensation_onset(γ, η, ω)
    βx   = β_sFM_sSG(γ, η)
    println("  γ=$γ : β_c=$(round(βc, digits=3)), √(γη)=$(round(βx, digits=3))")

    # Shaded band: temperature-tuned FM window (β_c, √(γη)).
    if !isnan(βc) && βc < βx
        Makie.vspan!(ax, βc, βx; color=(tune_color, 0.18))
    end
    Makie.vlines!(ax, [1.0]; color=:lightgray, linewidth=1.0, linestyle=:dash)

    Makie.lines!(ax, β_grid, fwd; color=curve_color, linewidth=1.8)
    Makie.xlims!(ax, 0.3, 5.0)

    # Phase labels live in the bottom subpanel only: "h=0" at the bottom-
    # left of its phase interval, "edge" at the bottom-right, "h≠0" near
    # the top of the orange band. The bottom subpanel gets extra y-padding
    # below the U-curve so the h=0/edge labels sit clearly below the curve.
    if i == length(γS) && !isnan(βc) && βc < βx
        ymin_data = minimum(fwd)
        ymax_data = maximum(fwd)
        yrange = ymax_data - ymin_data
        Makie.ylims!(ax, ymin_data - 0.55*yrange, ymax_data + 0.05*yrange)
        β_to_rel = β -> (β - 0.3) / (5.0 - 0.3)
        Makie.text!(ax, β_to_rel((1.1 + βc) / 2), 0.05;
                    text=L"h=0", fontsize=10, color=:black,
                    space=:relative, align=(:center, :bottom))
        Makie.text!(ax, β_to_rel((βc + βx) / 2), 0.95;
                    text=L"h\neq 0", fontsize=10, color=tune_color,
                    space=:relative, align=(:center, :top))
        Makie.text!(ax, β_to_rel((βx + 5.0) / 2), 0.05;
                    text="edge", fontsize=10, color=:black,
                    space=:relative, align=(:center, :bottom))
    end

    # Star at the local minimum (post-condensation dip).
    β_opt = NaN; fwd_opt = NaN
    for k in 2:length(fwd)-1
        if fwd[k-1] > fwd[k] && fwd[k+1] > fwd[k] && β_grid[k] > 1.0
            if isnan(β_opt) || fwd[k] < fwd_opt
                β_opt = β_grid[k]; fwd_opt = fwd[k]
            end
        end
    end
    if !isnan(β_opt)
        Makie.scatter!(ax, [β_opt], [fwd_opt];
                       color=curve_color, marker=:star5, markersize=10)
    end

    # γ label inside the subpanel (top-left corner of the data area).
    Makie.text!(ax, 0.0, 1.0;
                text=L"\gamma=%$γ", fontsize=10, color=:black,
                space=:relative, align=(:left, :top),
                offset=(4, -2))

    if !is_bottom
        Makie.hidexdecorations!(ax; ticks=false, grid=false)
    end
end

# ─── Panel labels ─────────────────────────────────────────────────
Makie.Label(fig[1, 1, Makie.TopLeft()], "A)";
            fontsize=14, font=:bold, halign=:left, padding=(0, 0, -10, 0))
Makie.Label(fig[2, 1, Makie.TopLeft()], "B)";
            fontsize=14, font=:bold, halign=:left, padding=(0, 0, 0, 0))

Makie.rowgap!(panel_b, 4)
Makie.rowgap!(fig.layout, -4)
Makie.resize_to_layout!(fig)

figures_dir = @__DIR__
mkpath(figures_dir)
out_pdf = joinpath(figures_dir, "fig_temperature_tuning_pmo.pdf")
out_png = joinpath(figures_dir, "fig_temperature_tuning_pmo.png")
Makie.save(out_pdf, fig)
Makie.save(out_png, fig; px_per_unit=3)
println("Saved ", out_pdf)
println("Saved ", out_png)

# (Removed best-effort copy into the paper figure directory; the
#  standalone accompanying-code release writes only into this
#  script's directory.)
