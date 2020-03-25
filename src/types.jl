struct Model{D}
    dims::Dims{3}
    h::NamedTuple{(:ZZ, :X, :Z), NTuple{3, Float64}}
    J::NamedTuple{(:ZZ, :X, :Z), NTuple{3, Vector{Float64}}}
    ∂Jt::Vector{Float64}
end
Base.size(M) = M.dims
Base.length(M) = prod(M.dims)

function couplings(θ::Vector{Float64})#, hx::Float64, hz::Float64)# ::ImmutableDict{Symbol, Float64})
    P = third(length(θ))
    θZZ = θ[1:3:end]
    θX = θ[2:3:end]
    θZ = θ[3:3:end]

    Jx = vcat(θZZ, zeros(1), reverse(θZZ))

    Jt = 0.5*log.(coth.(abs.(vcat(θX, reverse(θX)))))
    Jt = vcat(Jt, zeros(1))

    Jh = vcat(θZ, zeros(1), reverse(θZ))

    dJ = -csch.(2*vcat(θX, reverse(θX)))
    dJ = vcat(dJ, zeros(1))

    return Jx, Jt, Jh, dJ
end

function Model(L::Int, θ::Vector{Float64}, h)
    @assert mod(length(θ), 3) ≡ 0
    P = third(length(θ))
    Jx, Jt, Jh, ∂Jt = couplings(θ)
    J = (ZZ = Jx,    
         X = Jt, 
         Z = Jh)
    dims = L * ones(Int, 2)
    push!(dims, 2P+1)

    # println("Size is :\t", dims)
    # println("h is :\t", h)
    # println("J is :\t", J)
    Model{3}(Tuple(dims), h, J, ∂Jt)
end 

struct Config{D,S}
    config::Array{Int, D}
    sites::Array{CartesianIndex{D},D}
    neighbours_table::Array{CartesianIndex{D},S}
end    
Base.size(A::Config) = size(A.config)
Base.length(A::Config) = prod(size(A.config))

function Config(dims)
    D = length(dims)
    config = rand(-1:2:1, dims)
    sites = CartesianIndices(dims)
    neighbours_table = [sites .+ unit_tuple(D, 1, i) for i in -1:2:1]
    for j in 2:D # Open boundary conditions
        append!(neighbours_table, [sites .+ unit_tuple(D, j, i) for i in -1:2:1])
    end
    nn_table = cat(neighbours_table..., dims=D+1)
    nn_tab = permutedims(nn_table, [4, 1, 2, 3])
    Config{D,D+1}(config, sites, nn_tab)
end

@with_kw mutable struct Energy
    E::Float64 = 0.0
    ZZ::Float64 = 0.0
    X::Float64 = 0.0
    Z::Float64 = 0.0

    ZZ²::Float64 = 0.0
    X²::Float64 = 0.0
    Z²::Float64 = 0.0

    ZZ_X::Float64 = 0.0
    ZZ_Z::Float64 = 0.0
    X_Z::Float64 = 0.0
end

struct Derivatives
    tmp_∂E::NamedTuple{(:α, :β, :γ), NTuple{3, Vector{Float64}}}
    ∂E::NamedTuple{(:α, :β, :γ), NTuple{3, Vector{Float64}}}
    E_∂E::NamedTuple{(:α, :β, :γ), NTuple{3, Vector{Float64}}}
    X_∂X::Vector{Float64}
    tmp_X_∂X::Vector{Float64}
end

function Derivatives(P)
    tmp_∂E = (α = zeros(P),
          β = zeros(P),
          γ = zeros(P))
    E_∂E = (α = zeros(P),
          β = zeros(P),
          γ = zeros(P))
    ∂E = (α = zeros(P),
          β = zeros(P),
          γ = zeros(P))
    X_∂X = zeros(P)
    tmp_X_∂X = zeros(P)
    return Derivatives(tmp_∂E, ∂E, E_∂E, X_∂X, tmp_X_∂X)
end

mutable struct Thermo
    E::Energy
    mag::Float64
    D::Derivatives
end

function Thermo(P)
    return Thermo(Energy(), 0.0, Derivatives(P))
end

struct Gradient
    ∇E::NamedTuple{(:α, :β, :γ, :total), NTuple{4, Vector{Float64}}}
end

function Gradient(O::Thermo, P::Int)
    ∇E = (α = zeros(P),
          β = zeros(P),
          γ = zeros(P),
          total = zeros(3P))
    Gradient(∇E)
end
