struct StochasticMatrix
    S::Matrix{Float64}
    Sinv::Matrix{Float64}
end

function StochasticMatrix(O::Thermo, P::Int)
    S = zeros(3P, 3P)
    Sinv = zeros(3P, 3P)
    StochasticMatrix(S, Sinv)
end

function update_smatrix!(S::StochasticMatrix, M::Model, O::Thermo)
    P = third(size(S.S)[1])
    for j in 1:3:3P, i in 1:3:3P
        S.S[i, j] = O.E.ZZ² - O.E.ZZ^2
        S.S[i+1, j+1] = (O.E.X² - O.E.X^2)#*M.∂Jt[i]^2
        S.S[i+2, j+2] = (O.E.Z² - O.E.Z^2)

        S.S[i, j+1] = (O.E.ZZ_X - O.E.ZZ * O.E.X)#*M.∂Jt[i]
        S.S[i, j+2] = (O.E.ZZ_Z - O.E.ZZ * O.E.Z)
        S.S[i+1, j+2] = (O.E.X_Z - O.E.X * O.E.Z)#*M.∂Jt[i]

        S.S[i+1, j] = (O.E.ZZ_X - O.E.ZZ * O.E.X)#*M.∂Jt[i]
        S.S[i+2, j] = O.E.ZZ_Z - O.E.ZZ * O.E.Z
        S.S[i+2, j+1] = (O.E.X_Z - O.E.X * O.E.Z)#*M.∂Jt[i]
    end
    S.Sinv .= pinv(S.S)
end
