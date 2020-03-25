function metro!(S::Config, M::Model)
    for _ in eachindex(S.config) #length(M)
        mcstep!(S, M)
    end
end

function mcstep!(S::Config, M::Model)
    proposed_site = rand(S.sites)
    dE = energy_singleflip(S, M, proposed_site) # SLOW 1
    if dE < 0 || rand() < exp(-dE)
        S.config[proposed_site] *= -1
    end
end

function energy_singleflip(S::Config, M::Model, site::CartesianIndex{D}) where D
    E = 0.0
    T = length(M.J[:X])
    @views for nn in S.neighbours_table[:, site]
        if 1 ≤ nn[1] ≤ M.dims[1] &&  # OBNC
           1 ≤ nn[2] ≤ M.dims[2] &&
           1 ≤ nn[3] ≤ M.dims[3]

            if nn[D] == site[D] # horizontal couplings
                J = M.J[:ZZ][nn[D]]
            elseif nn[D] == mymod1(site[D] - 1, T) 
                J = M.J[:X][nn[D]]
            elseif nn[D] == mymod1(site[D] + 1, T) 
                J = M.J[:X][site[D]]
            end
            E += ifelse(S.config[nn] == S.config[site], J, -J)  # kinda slow
        end
    end
    # Exteral field
    # E += M.J[:Z][site[D]] * S.config[site]
    return 2.0 * E
end
