module va


#include(ST.jl)
include("./VR.jl")
include("./Alpha.jl")

using .VR
using .Alpha

export VRComplex
export AlphaComplex
export dimension, hassimplex, vert, numvertices, simplices, skeleton, numsimplices, gathersimplices

# single-character synonyms
#const d = coboundary
#const δ = coboundary_adj
#const Δ = laplacian
#const χ = euler_characteristic

end #module
