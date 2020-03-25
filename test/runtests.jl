using VITA2D
using Test
using Random
Random.seed!(42)

@testset "VITA2D.jl" begin
    @test half(4) == 2
    @test third(30) == 10

    L = 6
    P = 2
    θ = rand(3P)
    h = (ZZ = 1.0,
         X = 1.0,
         Z = 1.0)

    M = Model(L, θ, h)
    Obs = Thermo(P)
    Grad = Gradient(Obs, P)
    Sto = StochasticMatrix(Obs, P)
    C = Config((L, L, 2P+1))

    warmup!(C, M, 1)
    mc(C, M, Obs, Grad, 1, Sto)

    sgd_params = VITA2D.SGDParams()

    E = sgd(L=L, 
              P=P, 
              hX = 1.0,
              θ = θ,
              SGD=SGDParams(), 
              path=false)

    E = sgd(L=L, 
            P=P, 
            hX = 3.0,
            θ = θ,
            SGD=sgd_params, 
            print=true)
end
