mutable struct SGDParams
    lr::Float64
    lr₀::Float64
    N_sgd::Int
    N_mc::Int
    N_mc0::Int
end

function SGDParams(;lr=0.1, N_sgd=1, N_mc=10^3)
    SGDParams(lr, lr, N_sgd, N_mc, N_mc)
end

function sgd(;L=2, 
              P=1, 
              hX = 1.0,
              θ = rand(3),
              SGD=SGDParams(), 
              path=false, #"../data/2d-tfim/", 
              print=false)

    num_chain = nworkers() 

    θ[3:3:3P] .= zeros(P) # just for hZ=0
    h = (ZZ = 1.0, 
         X = hX, 
         Z = 0.0)

    M = Model(L, θ, h)
    Obs = [Thermo(P) for i in 1:num_chain]
    Grad = [Gradient(Obs[i], P) for i in 1:num_chain]
    Sto = [StochasticMatrix(Obs[i], P) for i in 1:num_chain]
    C = [Config(size(M)) for i in 1:num_chain]

    E, Eσ, mag, Mσ, grad = pmc(C, M, Obs, Grad, SGD.N_mc, Sto, num_chain)

    for i in 1:SGD.N_sgd
        M = Model(L, θ, h)
        Obs = [Thermo(P) for i in 1:num_chain]
        @time E, Eσ, mag, Mσ, grad = pmc(C, M, Obs, Grad, SGD.N_mc, Sto, num_chain)

        normalize!(grad)
        θ.+= SGD.lr * grad
        θ.= abs.(θ)
        θ[3:3:3P] .= zeros(P) # just for hZ=0

        SGD.lr = SGD.lr₀ * 0.4512 ^ floor(i / 10) 
        SGD.N_mc += 10*floor(1/SGD.lr) + 100

        if print == true
            println("\nIteration :", i)
            # println("θ is :", θ[θ .> 0])
            println("N is :", SGD.N_mc)
            println("Δ is :", SGD.lr)
            println("E is :", E/L^2)
            println("M is :", mag/L^2)
            println("gradient is :", grad)
        end

        if path != false
            params = θ[θ .> 0]
            result = vcat(sum(params), params, E/L^2, Eσ/L^2, mag/L^2, Mσ/L^2)'
            open(path, "a") do io
                writedlm(io, result)
            end  
        end
    end

    return E
end
