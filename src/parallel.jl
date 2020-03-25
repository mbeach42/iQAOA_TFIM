function pmc(Cs::Vector{Config{D,S}}, M::Model, Obs::Vector{Thermo}, Grad::Vector{Gradient}, N::Int, Sto::Vector{StochasticMatrix}, num_chains::Int) where {D,S}

    Es = pmap(i -> mc(Cs[i], M, Obs[i], Grad[i], N, Sto[i]), 1:num_chains)

    E = mean([Es[i][1] for i = 1:size(Es,1)])
    Eσ = std([Es[i][1] for i = 1:size(Es,1)])

    mag = mean([Es[i][3] for i = 1:size(Es,1)])
    Mσ = std([Es[i][3] for i = 1:size(Es,1)])

    grad = mean([Es[i][2][:total] for i = 1:size(Es,1)])
    # sgrad = mean([Es[i][4] for i = 1:size(Es,1)])

    return E, Eσ, mag, Mσ, grad
end
