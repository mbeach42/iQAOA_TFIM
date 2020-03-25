function warmup!(spins::Config, M::Model, N::Int)
    for _ in 1:N
        metro!(spins, M)
    end
end

function mc(spins::Config, M::Model, Obs::Thermo, Grad::Gradient, N::Int, Sto::StochasticMatrix)
    for _ in 1:N
        for _ in 1:5
            metro!(spins, M)
        end
        update_thermo!(Obs, spins.config, M, N, Grad) 
    end
    update_grad!(Grad, Obs, M, N) 
    # update_smatrix!(Sto, M, Obs) 
    return Obs.E.E, Grad.∇E, Obs.mag, Sto.Sinv*Grad.∇E[:total]
end
