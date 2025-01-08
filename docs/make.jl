using Pkg
using Documenter
cd(@__DIR__)
makedocs(
  sitename="hFlux",
  authors = "Oleksii Beznosov",
  format=Documenter.HTMLWriter.HTML(size_threshold = nothing),
  pages = 
  [
    "Home" => "index.md",
    "Quickstart Guide" => "quickstart.md",
    "For Developers" => "devel.md"
  ]
)

deploydocs(; repo = "github.com/obeznosov-LANL/hFlux")