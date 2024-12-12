# InclusiveScans

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://QEDjl-project.github.io/InclusiveScans.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://QEDjl-project.github.io/InclusiveScans.jl/dev)
[![Build Status](https://github.com/QEDjl-project/InclusiveScans.jl/workflows/Test/badge.svg)](https://github.com/QEDjl-project/InclusiveScans.jl/actions)
[![Test workflow status](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/formatter.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/QEDjl-project/InclusiveScans.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/QEDjl-project/InclusiveScans.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/QEDjl-project/InclusiveScans.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## Running Tests

You can run the provided test suite to ensure that the package is functioning correctly.
You need to open your julia REPL within the project's directory

```bash
julia --project=@.
```

In the Julia REPL, you need to enter the pkg mode and run the tests:

```julia-repl
julia> # press ]
pkg> activate .
pkg> test
```

This will execute the test suite and display the results, indicating whether the package functions as expected.

## Usage

To use the package in your Julia code, simply import it:

```julia-repl
using InclusiveScans
```

For detailed documentation on available functions and examples, please refer to the package documentation or source code.

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://QEDjl-project.github.io/InclusiveScans.jl/dev/90-contributing/)
