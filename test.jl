#include("ST.jl")
#include("./VR.jl")
#include("./Alpha.jl")
include("./va.jl")

using .va
#using .VR


dim = 3
simplex = rand(Int, dim+1)
#B = va.VR.VRComplex()
B = va.Alpha.AlphaComplex()
insert!(B, [0], 0.0, 0.0)
insert!(B, [1], 0.0, 0.0)
insert!(B, [2], 0.0, 0.0)
insert!(B, [3], 0.0, 0.0)
insert!(B, [4], 0.0, 0.0)
insert!(B, [0,1], 0.0, 0.0)
insert!(B, [1,2], 1.2, 0.0)
insert!(B, [3,1], 2.3, 1.2)
insert!(B, [1,2,3], 3.0, 3.0)   # 1 triangle
insert!(B, [1,2,3,4], 5.0, 5.0) # 4 triangles and 1 tet


#@show("Dimension = ", va.VR.dimension(B), va.VR.numsimplices(B,0), va.VR.numsimplices(B,1), va.VR.numsimplices(B,2), va.VR.numsimplices(B,3))
@show("Dimension = ", va.Alpha.dimension(B), va.Alpha.numsimplices(B,0), va.Alpha.numsimplices(B,1), va.Alpha.numsimplices(B,2), va.Alpha.numsimplices(B,3))