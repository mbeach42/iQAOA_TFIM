using VITA2D

L = 12
P = 1
θ = [0.2, 0.2, 0.0]
# θ = rand(3P)

sgd_params = VITA2D.SGDParams(lr=0.2, N_mc=10^3, N_sgd=200, num_avg=10)

E = sgd(L=L, 
        P=P, 
        hX = 3.0,
        θ = θ,
        SGD=sgd_params, 
        print=true)
