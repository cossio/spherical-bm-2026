# Data generator (CSV) for the predictive-posterior reverse KL panel of
# fig:warm-cold-pp.  Plotting itself happens downstream in matplotlib so the
# panel aligns visually with the phase-diagram panel; here we just compute
# D_KL(PP_eta || P*)/N along a grid of eta at omega*=2.2 for two gammas.

include(joinpath(@__DIR__, "..", "_lib", "UndersampledSphericalBMs2025", "src", "UndersampledSphericalBMs2025.jl"))
using .UndersampledSphericalBMs2025
const TS = UndersampledSphericalBMs2025.TeacherStudent

function safe_kl(γ, η, ω)
    try; return TS.all_kl_divergences(γ, η, ω)
    catch; return (; fwd_typ=NaN, rev_typ=NaN, fwd_pp=NaN, rev_pp=NaN); end
end

ω        = 2.2
γ_warm   = 0.1
γ_cold   = 0.3
γ_map    = 1.0
η_sweep  = collect(range(0.05, 5.0; length=400))

ys_warm = [getfield(safe_kl(γ_warm, η, ω), :rev_pp) for η in η_sweep]
ys_cold = [getfield(safe_kl(γ_cold, η, ω), :rev_pp) for η in η_sweep]
ys_map  = [getfield(safe_kl(γ_map,  η, ω), :rev_pp) for η in η_sweep]

data_dir = isempty(ARGS) ? joinpath(@__DIR__, "data", "reverse") : abspath(ARGS[1])
mkpath(data_dir)
out = joinpath(data_dir, "pp_rev_kl_2gamma.csv")
open(out, "w") do io
    println(io, "eta,kl_gamma_0.1,kl_gamma_0.3,kl_gamma_1.0")
    for (i, η) in enumerate(η_sweep)
        println(io, η, ",", ys_warm[i], ",", ys_cold[i], ",", ys_map[i])
    end
end
println("wrote $out  (omega*=", ω, ")")
i_w = argmin(replace(ys_warm, NaN => Inf))
i_c = argmin(replace(ys_cold, NaN => Inf))
i_m = argmin(replace(ys_map, NaN => Inf))
println("gamma=0.1 minimum at eta=", η_sweep[i_w], " (KL=", ys_warm[i_w], ")")
println("gamma=0.3 minimum at eta=", η_sweep[i_c], " (KL=", ys_cold[i_c], ")")
println("gamma=1.0 minimum at eta=", η_sweep[i_m], " (KL=", ys_map[i_m], ")")
