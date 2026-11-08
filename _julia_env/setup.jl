# One-shot environment setup for accompanying-code.
#
# Run from accompanying-code/ (or anywhere — paths are relative to this
# file) with:
#
#     julia --project=_julia_env _julia_env/setup.jl
#
# This activates the shared environment, registers the slim local
# UndersampledSphericalBMs2025 package via `Pkg.develop`, and resolves /
# precompiles all dependencies. After it succeeds, every Julia figure /
# simulation script in accompanying-code can be run directly — the
# activation prologue at the top of each script reuses this environment.

import Pkg

const ENV_DIR = @__DIR__
const LIB     = joinpath(ENV_DIR, "..", "_lib", "UndersampledSphericalBMs2025")

Pkg.activate(ENV_DIR)
Pkg.develop(path=LIB)
Pkg.instantiate()
Pkg.precompile()

@info "accompanying-code Julia environment ready" envdir=ENV_DIR pkg=LIB
