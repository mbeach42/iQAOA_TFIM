module VITA2D

export Model, Config, Gradient, Thermo, StochasticMatrix, SGDParams
export half, third
export warmup!, mc, pmc
export SGDParams
export sgd

using Parameters
using DelimitedFiles
using Dates
using Random
using LinearAlgebra
using Statistics
using Distributed

set_zero_subnormals(true)

# @everywhere using Pkg
# @everywhere Pkg.activate("/home/mbeach/VITA2D.jl")
# @everywhere Pkg.instantiate()
# @everywhere Pkg.update()
# @everywhere Pkg.resolve()



include("types.jl")
include("utils.jl")
include("sr.jl")
include("metro.jl")
include("update.jl")
include("mc.jl")
include("parallel.jl")
include("parallel_optimise.jl")


end # module
