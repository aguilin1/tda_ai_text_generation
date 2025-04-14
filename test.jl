#include("ST.jl")
#include("./VR.jl")
#include("./Alpha.jl")
include("./va.jl")

using .va
#using .VR


dim = 3
simplex = rand(Int, dim+1)
B = va.VR.VRComplex()
insert!(B, [0,1])
insert!(B, [1,2])
insert!(B, [3,1])
insert!(B, [1,2,3])   # 1 triangle
insert!(B, [4,5,6,7]) # 4 triangles and 1 tet


@show("Dimension = ", va.VR.dimension(B), va.VR.numsimplices(B,0), va.VR.numsimplices(B,1), va.VR.numsimplices(B,2), va.VR.numsimplices(B,3))