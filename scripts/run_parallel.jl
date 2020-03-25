using Distributed
addprocs(exeflags="--project=@.")
using Dates
using VITA2D

println("Loaded modules ...")

TASK_ID = parse(Int, ARGS[1])

L = 12
sgd_params = VITA2D.SGDParams(lr=0.2, N_mc=10^3, N_sgd=2000)

repeats = [i for i in 1:2]
# Ps = [2]
Ps = [5, 10]
hs = [2.0]#3.0, 2.0, 2.5, 4.0, 3.5]


list = Iterators.product(Ps, hs, repeats) |> collect
println("length of array jobs is:\t", length(list))

# println(list)
println(list[TASK_ID])

P, hX, r = list[TASK_ID]
path = "../data/" * string(Dates.today())
path *= "/L-$L-p-$P-h-$hX/"
mkpath(path)
path *= "iteration-$r.txt" 
touch(path)

θ = P * rand(3P)
# if hX == 2.0
    # if repeats == 1
        # θ = [0.613, 1.632, 0.0, 0.3469, 0.20271, 0.0] 
    # else
        # θ = [1.22, 1.632, 0.0, 0.3469, 0.20271, 0.0] 
    # end
# end

# Precompile 
sgd(L=L, 
    P=P, 
    hX = hX,
    θ = θ,
    SGD=SGDParams(), 
    path=false,
    print=true)

sgd(L=L, 
    P=P, 
    hX = hX,
    θ = θ,
    SGD=sgd_params, 
    path=path,
    print=true)
