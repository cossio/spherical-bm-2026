module UndersampledSphericalBMs2025

# Slim accompanying-code subset: only the analytic RMT and teacher-student
# closed-form solvers used by the figure scripts. The full upstream package
# additionally ships several MCMC eigenvalue samplers (`mala*`,
# `SimEigvalsOnly2026v*`) and the FredholmDeterminants/TracyWidom helpers.
# None of those are needed to reproduce the figures in this release.

include("rmt.jl")
include("rmt2.jl")
include("teacher_student.jl")

end
