using Distributed
using Statistics
using LinearAlgebra
using DelimitedFiles
using Dates
import Base.ImmutableDict

using Random

@everywhere include("../src/VMC.jl")

function nsgd(L, P, N, N_sgd; hX=3.0)

    outfile = "../data/2d-tfim-" * string(Dates.today())
    mkpath(outfile)

    θ = 0.1 * ones(3P) .+ P*rand(3P)
    θ[3:3:end] .= zero(P)
    h = Base.ImmutableDict(:ZZ => 1.0, :X => hX, :Z => 0.0)

    lr₀ = 0.1
    lr = lr₀
    num_chain = nworkers() 

    println("Initial parameters are :", θ)
    println("Threads are ", num_chain)

    M = VMC.Model(L, θ, h)
    Obs = [VMC.Thermo(P) for i in 1:num_chain]
    Grad = [VMC.Gradient(Obs[i], P) for i in 1:num_chain]
    Sto = [VMC.StochasticMatrix(Obs[i], P) for i in 1:num_chain]
    C = [VMC.Config(size(M)) for i in 1:num_chain]

    Es = zeros(20)
    norms = zeros(20)
    grad_list = zeros(3P, 20)

    @time E, grad, mag, σ², sgrad = VMC.pmc(C, M, Obs, Grad, 10^4, Sto, num_chain)

    T = 0 

    for i in 1:N_sgd
        M = VMC.Model(L, θ, h)
        Obs = [VMC.Thermo(P) for i in 1:num_chain]
        @time E, grad, mag, σ², sgrad = VMC.pmc(C, M, Obs, Grad, N, Sto, num_chain)

        normalize!(grad)
        θ.+= lr * grad
        θ.= abs.(θ)

        popfirst!(Es)
        popfirst!(norms)
        grad_list .= circshift(grad_list, (0, 1))

        push!(Es, E)
        push!(norms, norm(grad))
        grad_list[:, end] .= grad

        T0 = 6.222652893066406e-7 * N * L^2
        # println("Expected time is ", T0, " seconds")
        T += T0 
        # println("total time is ", T/60/60, " hours")

        N = Int(floor(10^3 * 2^floor(i / 20)))
        lr = lr₀  * 0.4^floor(i / 20) 

        # println("\nIteration :", i)
        # println("θ is :", θ)
        # println("N is :", N)
        # println("Δ is :", lr)
        # println("E is :", mean(Es)/L^2)
        # println("σ² is :", σ²/L^2)
        # println("M is :", mag/L^2)
        # println("gradient is :", grad)

        if N > 50 && mod(N, 5) ≡ 0
            res = vcat(L, h[:X], sum(θ), θ, mean(Es)/L^2, mag/L^2, std(Es)/L^2)'

            FILE = open(outfile * "/L-$L-p-$P-h-$hX.txt", "a")
            writedlm(FILE, res) 
            close(FILE)
        end
    end

    return E
end

# nsgd(2, 1, 1, 10)
# @time nsgd(12, 1, 10^3, 160) # 20 minutes

# 200 is 20min
# 125 is 3 hours
# 160 is 7.3 hours
