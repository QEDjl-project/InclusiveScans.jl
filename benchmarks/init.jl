import Pkg

inclusive_scans_jl = Pkg.PackageSpec(path = "..")

Pkg.develop([inclusive_scans_jl])
Pkg.instantiate()