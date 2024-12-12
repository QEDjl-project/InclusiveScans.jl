using InclusiveScans
using Documenter

DocMeta.setdocmeta!(
    InclusiveScans,
    :DocTestSetup,
    :(using InclusiveScans);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [InclusiveScans],
    authors = "Uwe Hernandez Acosta, Simeon Ehrig",
    repo = "https://github.com/QEDjl-project/InclusiveScans.jl/blob/{commit}{path}#{line}",
    sitename = "InclusiveScans.jl",
    format = Documenter.HTML(;
        canonical = "https://QEDjl-project.github.io/InclusiveScans.jl",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/QEDjl-project/InclusiveScans.jl")
