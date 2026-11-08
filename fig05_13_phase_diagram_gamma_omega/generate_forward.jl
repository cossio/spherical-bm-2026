# Data generator (CSV) for the predictive-posterior forward KL panel of the
# fwd-KL temperature phase diagram.  Plotting happens downstream in
# matplotlib (see plot_forward.py); here we just tabulate
# D_KL(P*||P_pp,η)/N at omega*=2.2 on a grid of eta for four gammas covering
# the warm, mixed-tie, cold, and MAP regimes.

include(joinpath(@__DIR__, "..", "_lib", "UndersampledSphericalBMs2025", "src", "UndersampledSphericalBMs2025.jl"))
using .UndersampledSphericalBMs2025
const TS = UndersampledSphericalBMs2025.TeacherStudent

function safe_kl(γ, η, ω)
    try; return TS.forward_kl_predictive(γ, η, ω)
    catch; return NaN; end
end

ω        = 2.2
γ_warm   = 0.10           # warm flat regime
γ_mixed  = 0.215          # mixed/tie sliver  (γ_wc ≈ 0.2066, γ_flat ≈ 0.2226)
γ_cold   = 0.30           # unique cold optimum
γ_map    = 0.80           # MAP regime  (γ_∞ ≈ 0.4769)
η_sweep  = collect(range(0.02, 5.0; length=600))

ys_warm  = [safe_kl(γ_warm,  η, ω) for η in η_sweep]
ys_mixed = [safe_kl(γ_mixed, η, ω) for η in η_sweep]
ys_cold  = [safe_kl(γ_cold,  η, ω) for η in η_sweep]
ys_map   = [safe_kl(γ_map,   η, ω) for η in η_sweep]

data_dir = isempty(ARGS) ? joinpath(@__DIR__, "data", "forward") : abspath(ARGS[1])
mkpath(data_dir)
out = joinpath(data_dir, "pp_fwd_kl_4gamma.csv")
open(out, "w") do io
    println(io, "eta,kl_gamma_010,kl_gamma_215,kl_gamma_030,kl_gamma_080")
    for (i, η) in enumerate(η_sweep)
        println(io, η, ",",
                ys_warm[i],  ",",
                ys_mixed[i], ",",
                ys_cold[i],  ",",
                ys_map[i])
    end
end
println("wrote $out  (omega*=", ω, ")")

i_w = argmin(replace(ys_warm,  NaN => Inf))
i_x = argmin(replace(ys_mixed, NaN => Inf))
i_c = argmin(replace(ys_cold,  NaN => Inf))
i_m = argmin(replace(ys_map,   NaN => Inf))
println("γ=", γ_warm,  ": min at η=", η_sweep[i_w], "  (KL=", ys_warm[i_w],  ")")
println("γ=", γ_mixed, ": min at η=", η_sweep[i_x], "  (KL=", ys_mixed[i_x], ")")
println("γ=", γ_cold,  ": min at η=", η_sweep[i_c], "  (KL=", ys_cold[i_c],  ")")
println("γ=", γ_map,   ": min at η=", η_sweep[i_m], "  (KL=", ys_map[i_m],   ")")
