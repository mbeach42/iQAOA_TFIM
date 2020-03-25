# VITA2D.jl

Start julia from main directory with `julia --project`
then compute the energy like
```julia
using Distributed 
addprocs(exeflags="--project=@.") # these two lines enable MC sampling on all cores

using VITA2D
energy(L=4, P=1, θ=ones(3), Nmc=10^4, hX=3.0)
```

If it doesn't run, then try
`julia --project`
Now press `]` and type `instantiate`
This should initialize the project and all dependancies.
