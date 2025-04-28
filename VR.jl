# This module implements Vietoris-Rips using 2x ball size
# I am using 2x ball size so that Alpha complex can be realized using the same code.
#

module VR

"""
Vietoris-Rips are constructed as filtrations diameters are fed (2x of radius or ball diameter) is provided.
Persistent homology information is generated and saved for display interfacing if array of ball diameters are fed.
"""

#import .ST
include("./ST.jl")

using .STs
using Base: insert!

mutable struct VRComplex
    dimension :: Int
    simplices :: STs.ST
    VRComplex() = new(-1, STs._root())
end

function VRComplex(iter)
    VietorisRipsC = VRComplex()
    foreach(s -> insert!(VietorisRipsC, s), iter)
    return VietorisRipsC
end

# provide a copy function in base for generic, type VRComplex
Base.copy(VietorisRipsC::VRComplex) = VRComplex(simplices(VietorisRipsC))

function Base.insert!(VietorisRipsC::VRComplex, simp, filtration, death=0.0)
    STs.insert!(VietorisRipsC.simplices, simp, filtration, death)
    _dim = length(simp) -1
    if _dim > VietorisRipsC.dimension
        VietorisRipsC.dimension = _dim
    end
    return VietorisRipsC
end

function skeleton(VietorisRipsC::VRComplex, l::Integer)
    # starting at 0 index to also include top level tree.
    return VRComplex(Iterators.flatten(simplices(VietorisRipsC,iter) for iter in 0:l))
end

dimension(VietorisRipsC::VRComplex) = VietorisRipsC.dimension

function vert(VietorisRipsC::VRComplex)
    return STs.get_children(VietorisRipsC.simplices)
end

function numvertices(VietorisRipsC::VRComplex)
    return length(VietorisRipsC.simplices.children)
end

# get all simplices
function simplices end

# get simplices by dimension 0 for vertices, 1 for edges, 2 for triangles, 3 for tet.
function simplices(VietorisRipsC::VRComplex) :: Vector{Vector{Int}}
    faces = Iterators.flatten(STs.getsimplices(VietorisRipsC.simplices, k) for k in 0:dimension(VietorisRipsC))
    return pushfirst!(collect(faces), []) # empty is also needed.
end

function simplices(VietorisRipsC::VRComplex, dim::Integer) :: Vector{Vector{Int}}
    if dim < 0 || dim > dimension(VietorisRipsC)
        return Vector{Int}[]
    else
        return STs.getsimplices(VietorisRipsC.simplices, dim)
    end
end

function hassimplex(VietorisRipsC::VRComplex, simp)
    return STs.hassimplex(VietorisRipsC.simplices, simp)
end

function numsimplices end

function numsimplices(VietorisRipsC::VRComplex) :: Int
    #println(dimension(VietorisRipsC))
    return 1 + sum(Iterators.flatten(STs.numsimplices(VietorisRipsC.simplices, l) for l in 0:dimension(VietorisRipsC)))
end

function numsimplices(VietorisRipsC::VRComplex,dim::Integer) :: Int
    if dim < 0 || dim > dimension(VietorisRipsC)
        return 0
    elseif dim == 0
        return numvertices(VietorisRipsC)
    else
        return STs.numsimplices(VietorisRipsC.simplices, dim)
    end
end

# gathers simplicial complexes of asked dimension. 0 - vertices, 1 - edges, 2- triangles, 3- tetra etc..
function gathersimplices(VietorisRipsC::VRComplex, dim::Integer) :: Vector{Vector{Int}}
    C = vert(VietorisRipsC)     # calls get_children giving ids. - dim 0
    
    # there are no deeper level children at start.
    # we do union find i.e. find cliques.
    tetrahedrons_or_tri = Vector{Vector{Int}}()
    for simplices_0 in C         # scan vertices
        simplices_1 = STs.get_children(VietorisRipsC.simplices, simplices_0)
	#println("Pr - ", simplices_0, "Children - ", simplices_1)
        for child in simplices_1      # dim 2
            simplices_2 = STs.get_children(VietorisRipsC.simplices, child)
	    #println("  Pr - ", child, "Children - ", simplices_2)
            for gchild in simplices_2
                simplices_3 = STs.get_children(VietorisRipsC.simplices, gchild)
	        #println("    Pr - ", gchild, "Children - ", simplices_3)
                for ggchild in simplices_3
                    # check all interconnects and inject a tetrahedron.
                    # 1-2-3-4 chain already exists
                    sv = [simplices_0, child, gchild, ggchild]
		    #println("      deepest - ", ggchild)
                    edge_1_3 = sv[3] in simplices_1  # 1 is connected to 3
                    edge_1_4 = sv[4] in simplices_1  # 1 is connected to 3
                    edge_2_4 = sv[4] in simplices_2  # 2 is connected to 4
                    if dim == 3 && edge_1_3 && edge_2_4 && edge_1_4
                        push!(tetrahedrons_or_tri, [simplices_0, child, gchild, ggchild]) # we will add all triangles 
                    else
			if dim <=3 && edge_1_3 && !([sv[1], sv[2], sv[3]] in tetrahedrons_or_tri) 
                            push!(tetrahedrons_or_tri, [sv[1], sv[2], sv[3]])
                        end
			if dim <=3 && edge_2_4 && !([sv[2], sv[3], sv[4]] in tetrahedrons_or_tri)
                            push!(tetrahedrons_or_tri, [sv[2], sv[3], sv[4]])
                        end
			if dim <= 3 && edge_1_4 && edge_2_4 && !([sv[1], sv[2], sv[4]] in tetrahedrons_or_tri)
                            push!(tetrahedrons_or_tri, [sv[1], sv[2], sv[4]])
                        end
			if dim <= 3 && edge_1_3 && edge_1_4 && !([sv[1], sv[3], sv[4]] in tetrahedrons_or_tri)
                            push!(tetrahedrons_or_tri, [sv[1], sv[3], sv[4]])
                        end
                    end
                end
            end
        end
    end
    return tetrahedrons_or_tri
end

end
