import Pkg

inclusive_scans_jl = Pkg.PackageSpec(path="..")
constainted_gaussians_jl = Pkg.PackageSpec(path="./ConstrainedGaussians")

Pkg.develop([inclusive_scans_jl,constainted_gaussians_jl])
Pkg.instantiate()

