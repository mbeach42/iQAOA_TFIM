function compute_obs(O::Thermo, S::Array{Int, 3}, M::Model)
    L = M.dims[1]
    P = length(O.D.∂E[:α])
    m = P + 1
    E, Z, X, ZZ, mag = 0.0, 0.0, 0.0, 0.0, 0.0
    for var in [:α, :β, :γ]
        O.D.tmp_∂E[var] .= zeros(P)
    end
    O.D.tmp_X_∂X .= zeros(P)

    for j in 1:L, i in 1:L 
        if i < L && j < L
            zz = S[i, j, m] * S[i+1, j, m]
            zz += S[i, j, m] * S[i, j+1, m]
        elseif j == L && i < L
            zz = S[i, j, m] * S[i+1, j, m]
        elseif i == L && j < L
            zz = S[i, j, m] * S[i, j+1, m]
        else
            zz = 0
        end
        x = exp(-2.0 * M.J[:X][m] * S[i, j, m] * S[i, j, m+1]) # SLOW LINE 3
        # z = S[i, j, m] 
        e = zz + M.h[:X]*x #+ M.h[:Z]*z 
        mag += S[i, j, m]

        # Z += z 
        ZZ += zz 
        X += x 
        E += e 

        for p in 1:P
            p̃ = 2P - p + 2
            if i < L && j < L
                O.D.tmp_∂E[:α][p] += S[i, j, p] * S[i+1, j, p] + S[i, j, p̃] * S[i+1, j, p̃]
                O.D.tmp_∂E[:α][p] += S[i, j, p] * S[i, j+1, p] + S[i, j, p̃] * S[i, j+1, p̃]
            elseif j == L && i < L
                O.D.tmp_∂E[:α][p] += S[i, j, p] * S[i+1, j, p] + S[i, j, p̃] * S[i+1, j, p̃]
            elseif i == L && j < L
                O.D.tmp_∂E[:α][p] += S[i, j, p] * S[i, j+1, p] + S[i, j, p̃] * S[i, j+1, p̃]
            end
            # O.D.tmp_∂E[:γ][p] += S[i, j, p] + S[i, j+1, p̃]
            O.D.tmp_∂E[:β][p] += (S[i, j, p] * S[i, j, p+1] + S[i, j, p̃] * S[i, j, p̃-1])* M.∂Jt[p]
            if p == P
                O.D.tmp_X_∂X[p] += -2S[i, j, m] * S[i, j, m+1] * M.∂Jt[m] * M.h[:X] * x
            end
        end 
    end

    return E, ZZ, X, Z, mag
end


function update_thermo!(O::Thermo, c::Array{Int, 3}, M::Model, N::Int, Grad::Gradient) 
    P = length(O.D.∂E[:α])
    E, ZZ, X, Z, mag = compute_obs(O, c, M)
    O.E.ZZ += ZZ / N
    # O.E.Z += Z / N
    O.E.X += X / N

    O.E.ZZ² += ZZ^2 / N
    O.E.X² += X^2 / N
    # O.E.Z² += Z^2 / N

    O.E.ZZ_X += ZZ*X / N
    # O.E.ZZ_Z += ZZ*Z / N
    # O.E.X_Z += X*Z / N

    O.E.E += E / N

    O.mag += abs(mag) / N

    for p in 1:P
        for var in [:α, :β]#, :γ]
            O.D.∂E[var][p] += O.D.tmp_∂E[var][p] ./ N
            O.D.E_∂E[var][p] += E * O.D.tmp_∂E[var][p] ./ N
        end
        O.D.E_∂E[:β][p] += O.D.tmp_X_∂X[p] ./ N
    end
end

function update_grad!(grad::Gradient, O::Thermo, M::Model, N::Int)
    N = size(M)[1] * size(M)[2]
    for var in [:α, :β]#, :γ]
        grad.∇E[var] .= (-O.E.E * O.D.∂E[var] .+ O.D.E_∂E[var]) / N
    end
    grad.∇E[:total] .= vec(hcat(grad.∇E[:α], grad.∇E[:β], grad.∇E[:γ])')
end
