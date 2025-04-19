# We will construct Alpha complex in 2 steps.
# 1. Construct Veroni cells.
# 2. We will not do Delaunay Triangulation as it eats up lot of time.
#=
Parallel computation of alpha complexes for biomolecules
Talha Bin Masood a,∗, Tathagata Ray b , Vijay Natarajan c,
=#

module Alpha

"""
Alpha are constructed as filtrations diameters are fed (2x of radius or ball diameter) is provided.
Key difference between Alpha and Vietoris-Rips is Nerve Theorem (Intersection of Balls is non empty)
A nerve is a simplicial complex constructed from a cover of a topological space. In the case of balls, 
the nerve is formed by adding a k-simplex for every k+1 balls that have a non-empty intersection. 
Persistent homology information is generated and saved for display interfacing if array of ball diameters are fed.
"""

#import .ST
include("./ST.jl")

using .STs
using Base: insert!
using DataStructures

Ball_radius = 5.0    # https://discourse.julialang.org/t/static-structs-with-constant-field/48009

mutable struct AlphaComplex
    dimension :: Int
    simplices :: STs.ST
    AlphaComplex() = new(-1, STs._root())
end

function AlphaComplex(iter)
    AlphaC = AlphaComplex()
    foreach(s -> insert!(AlphaC, s), iter)
    return AlphaC
end

# provide a copy function in base for generic, type AlphaComplex
Base.copy(AlphaC::AlphaComplex) = AlphaComplex(simplices(AlphaC))

function Base.insert!(AlphaC::AlphaComplex, simp, filtration=0.0)
    STs.insert!(AlphaC.simplices, simp, filtration)
    _dim = length(simp) -1
    if _dim > AlphaC.dimension
        AlphaC.dimension = _dim
    end
    return AlphaC
end

# -------------------------------
# Basic structural setup is done.
# -------------------------------
#=
Definition 2 (Alpha complex). A simplex σ d = (pσ0 , pσ1 , · · · , pσd ), 0 ≤ d ≤ 3, belongs to the alpha complex Kα of B if and only
if there exists a point p ∈ R3 such that the following three conditions are satisfied:
AC1: π (p, bσ0 ) = π (p, bσ1 ) = · · · = π (p, bσd ),
AC2: π (p, bσ0 ) ≤ π (p, b i ) for b i ∈ B − σ d , and
AC3: π (p, bσ0 ) ≤ α or equivalently, the Size of σ d is at most α.
=#
### Key Implementation Choice ####
# We are not trying to see whether more than 4 points are in sphere as we are only going to approach 
# tetrahedron (4-d) space. With this constraint in place, we can avoid converting distance matrix through
# MultiDimensionalScaling to get lower dimensional embeddings to get co-ordinates, which is what most
# Alpha complex creation algorithms implemented professionally require as they need co-ordinates to compute 
# intersecting spheres/bounding spheres when number of points go to >4, See Gudhi, Dynosys needing co-ordinates not
# distance matrix. Ripser does not do Alpha etc...

# We will deploy Heron's algorithm as
# 1. s = (d1 + d2 + d3)/2
# 2. k = sqrt(s(s-d1)(s-d2)(s-d3))
# 3. R = Bounding Sphere's radius with points on surface = (d1*d2*d3)/4K. 
# 4. Simplices (Triangles) will only be added if R < Ball radius. -- Key difference from Vietoris Rips.
# 5. 3 balls can intersect each other but still may not intersect all. Key condition for adding Alpha simplex. 
#    Requires union of balls (Nerve Theorem - Intersection of all balls to be not empty).
# This covers a Triangle.
# Going to Tetrahedron -
# Welzl's algorithm requires points.
# Quadratic programming approach to solve for minimum enclosing radius is outlined in https://www.nature.com/articles/s41598-024-63971-3
# Carlsson, E., Carlsson, J. Computing the alpha complex using dual active set quadratic programming. 
# Sci Rep 14, 19824 (2024). https://doi.org/10.1038/s41598-024-63971-3.4
# By restricting dimension to 4, we can create a bounding sphere from 2 existing distances (exclusing the smallest one) 
# and distance to this point, creating a new bigger sphere and use this as criterion to include 4 points.
# We first check whether 4th point is already inside by verifying 2*R < distance from 1st point to 4th point.
#  During triangle complex creation we ignore this 4th point (Veronoi cells based creation also follows similar approach).
# New bigger sphere is needed for tetrahedron with 4 points:
# This means at any column of distance matrix, in Julia distances to other points are laid out in order, we pick 4th smallest distances.
# Then we then lookup all 4 distances from 3 points on sphere to this 4th point and evaluate B_R + Br_4 for overlap (filtration)
# If it is less, we include this point into our tetrahedron.
# This way we do not need to convert distances to co-ordinates, we do not need to create veronai cells, we also do not need to 
# perform Delanuay triangulation and by limiting dimensions to 4, we also do not need to solve for min radius using quadratic
# programming.

# Simplex tree construction remains the same. We do not remove any simplices as we incrementally construct Alpha Complex.

function skeleton(AlphaC::AlphaComplex, l::Integer)
    # starting at 0 index to also include top level tree.
    return AlphaComplex(Iterators.flatten(simplices(AlphaC,iter) for iter in 0:l))
end

dimension(AlphaC::AlphaComplex) = AlphaC.dimension

function vert(AlphaC::AlphaComplex)
    return STs.get_children(AlphaC.simplices)
end

function numvertices(AlphaC::AlphaComplex)
    return length(AlphaC.simplices.children)
end

# get all simplices
function simplices end

# get simplices by dimension 0 for vertices, 1 for edges, 2 for triangles, 3 for tet.
function simplices(AlphaC::AlphaComplex) :: Vector{Vector{Int}}
    faces = Iterators.flatten(STs.getsimplices(AlphaC.simplices, k) for k in 0:dimension(AlphaC))
    return pushfirst!(collect(faces), []) # empty is also needed.
end

function simplices(AlphaC::AlphaComplex, dim::Integer) :: Vector{Vector{Int}}
    if dim < 0 || dim > dimension(AlphaC)
        return Vector{Int}[]
    else
        return STs.getsimplices(AlphaC.simplices, dim)
    end
end

function hassimplex(AlphaC::AlphaComplex, simp)
    return STs.hassimplex(AlphaC.simplices, simp)
end

function numsimplices end

function numsimplices(AlphaC::AlphaComplex) :: Int
    #println(dimension(VietorisRipsC))
    return 1 + sum(Iterators.flatten(STs.numsimplices(AlphaC.simplices, l) for l in 0:dimension(AlphaC)))
end

function numsimplices(AlphaC::AlphaComplex,dim::Integer) :: Int
    if dim < 0 || dim > dimension(AlphaC)
        return 0
    elseif dim == 0
        return numvertices(AlphaC)
    else
        return STs.numsimplices(AlphaC.simplices, dim)
    end
end

function herons_sphere(x::Vector{Float32}) :: Float32
    a = x[1]   # a-b
    b = x[2]   # b-c
    c = x[3]   # a-c
    s = (a + b + c)/2
    K = sqrt(s*(s-a)*(s-b)*(s-c))
    R = (a*b*c)/(4*K)
    return R
end
# This is where we have different Sphere based integration of Vertices.
# How to implement balls of different radii? We will approach it after Balls of equal raddii are tested to be working.
# --- We will create similar strategy as described above, we will integrate 3 smaller atoms and then assign larger Ball to 
# bigger atom as 4th one. In case of 3 vertices, we will divide distance/2 among smaller atoms and then enlarge 3rd Radius by 
# using a factor to multiply distance with... so that we get a Larger Sphere ensuring one ball has bigger radius than other two.
# We really do not want to compute Gram Matrix from Distance Matrix and then compute eigen values to get co-ordinates going to 
# Ritter approximation or Welzl's algorithm.
# gathers simplicial complexes of asked dimension. 0 - vertices, 1 - edges, 2- triangles, 3- tetra etc..
function gathersimplices(AlphaC::AlphaComplex, dim::Integer) :: Vector{Vector{Int}}
    C = vert(AlphaC)     # calls get_children giving ids. - dim 0
    
    # there are no deeper level children at start.
    # we do union find i.e. find cliques.
    tetrahedrons_or_tri = Vector{Vector{Int}}()
    for simplices_0 in C         # scan vertices
        simplices_1 = STs.get_children(AlphaC.simplices, simplices_0)
	#println("Pr - ", simplices_0, "Children - ", simplices_1)
        for child in simplices_1      # dim 2
            simplices_2 = STs.get_children(AlphaC.simplices, child)
	    #println("  Pr - ", child, "Children - ", simplices_2)
            for gchild in simplices_2
                simplices_3 = STs.get_children(AlphaC.simplices, gchild)
	        #println("    Pr - ", gchild, "Children - ", simplices_3)
                for ggchild in simplices_3
                    # check all interconnects and inject a tetrahedron.
                    # 1-2-3-4 chain already exists
                    sv = [simplices_0, child, gchild, ggchild]
		    #println("      deepest - ", ggchild)
                    edge_1_3 = sv[3] in simplices_1  # 1 is connected to 3
                    edge_1_4 = sv[4] in simplices_1  # 1 is connected to 3
                    edge_2_4 = sv[4] in simplices_2  # 2 is connected to 4
                    # We confirmed edges exists which are within 2*Ball Radius distance.
                    if dim == 3 && edge_1_3 && edge_2_4 && edge_1_4
                        # Compute bounding sphere
                        three_distances = Vector{Float32}()
                        # Simplex tree first level children
                        node_1 = STs.find_child(AlphaC.simplices, simplices_0)
                        node_2 = STs.find_child(AlphaC.simplices, child)
                        # Simplex tree starts out with 1 init_depot_path
                        # Only edges are stored so if 1 is connected to 1-2-3-4 we will have 
                        # 1-2, 1-3, 1-4, 2-3, 3-4..., so we will need to pick gchild as child from root node, and then look
                        node_3 = STs.find_child(AlphaC.simplices, gchild)
                        node_4 = STs.find_child(AlphaC.simplices, ggchild)

                        # Now get distances between nodes.
                        # This gives information about distances among points on tetrahedron.
                        # we need to keep track of longest distance to make sure it is less than 
                        # radius of bounding sphere + radius of the ball at either of those points.
                        ## -- This is to make sure point of intersection of 3 balls (Radius of bounding sphere) is on the 
                        ## -- surface of 4th ball. 
                        node_5 = STs.find_child(node_2, gchild) # b-C
                        node_6 = STs.find_child(node_2, ggchild) # b-C
                        node_7 = STs.find_child(node_3, ggchild) # b-C

                        # these are distances from a, i.e. a-b, a-c, a-d, we also need b-c, b-d, c-d
                        # birth carries edge length (except for first where it carries distance to 2., 2 carries distance to 1)
                        # 3.birth is distance of node 3 to node 1 etc..
                        #                   +----------------------------------------------------------------------------------------------------------------
                        #          . b      |
                        #         /         Bounding Sphere of Radius R (selecting farthest 3 points to construct bounding sphere e.g. acd)
                        #      a .----.c    |----------------------------------------------------------------------------------------------------------------
                        #         \         Check for viability of 4th to create tetrahedron iff R + Ball_r4 < filtration radius
                        #          \        | Add Each Triangles based on Radius of Bounding Spheres (1,2,3,4 = 123, 124, 134, 423 bouding Sp < filtration.)
                        #           . d     | Tetra only if largest + 4th is still < filtration
                        #                   | If any one of 4 triangles, fails to get added, we just ignore tetrahedron addition and just add permissible 
                        #                   | triangles.
                        #                   +---------------------------------------------------------------------------------------------------------------- 
                        pq = PriorityQueue(Base.Order.Reverse) # Gives highest distance first.

                        pq["a"] = node_1.birth 
                        pq["e"] = node_5.birth # b-c
                        pq["b"] = node_2.birth 
                        pq["f"] = node_6.birth # b-d
                        pq["c"] = node_3.birth 
                        pq["g"] = node_7.birth # c-d
                        pq["d"] = node_4.birth
                        # All distances should be less than 1.63299*Ball_radius for tetrahedron to be in.
                        # Send top 3 distances forming tetra to Bounding Sphere
                        k,v = dequeue_pair!(pq)
                        push!(three_distances, v)
                        l,v = dequeue_pair!(pq)
                        push!(three_distances, v)
                        m,v = dequeue_pair!(pq)
                        push!(three_distances, v)
                        # This will give minimum bounding sphere, not assuming balls are merely touching so creating 2*Ball_radius sides...
                        R = herons_sphere(three_distances) # find radius of sphere touching 3 points on surface.
                        # The four spheres intersect if the intersection point of first 3 spheres (Sphere of Radius R),
                        # also lies on the fourth sphere. Meaning R + Radius of 4th Ball < max of 6 distances among all points.

                        # We are in this loop because 1-2-3-4 intersect and 1-3, 2-4, 1-4 spheres do intersect i.e. (< 2*Ball_radius).
                        n, v = dequeue_pair!(pq) # next largest distance in tetrahedron

                        # Balls are of equal size:
                        # We already know that all edges are < 2 * Ball_Radius, implying n is also < 2*Ball_radius.
                        # As long as 4th point is less than equidistance points of other three, i.e. from center of bounding sphere (3 balls initersection), 
                        # 4th ball will intersect.
                        ####################################################
                        # Here we will introduce some geometric analysis.
                        ####################################################
                        # 1-D Two balls of radius r will intersect if distance between 2 points is less than 2*r
                        # 2-D Three balls of radius r will intersect each other if distance between each point is at most sqrt(3)*R = 1.73205 * R (equidistance = bounding sphere R)
                        # 3-D Four Balls of radius r will intersect each other if distance between each point is at most (4/sqrt(6))*R = 1.63299 * R (bounding Sphere) -- Max distance.
                        # These relationships come from considering the largest triangles/tetrahedron that can be inscibed in a sphere... Farthest points can be all equal
                        # once we consider worse case, every other case will place spheres into intersecting positions.
                        # 
                        if ((4*R)/sqrt(6) > v)                       # Point of intersection of 3 balls is on 4th Ball
                            push!(tetrahedrons_or_tri, [simplices_0, child, gchild, ggchild]) # we will add all triangles 
                        end
                    else
			            if dim <=3 && edge_1_3 && !([sv[1], sv[2], sv[3]] in tetrahedrons_or_tri) 
                            # for triangles we will use upperbounds from our geometric analysis
                            # All distances should be less than 1.73205*Ball_radius for triangles to be in.
                            node_1 = STs.find_child(AlphaC.simplices, simplices_0)
                            node_2 = STs.find_child(AlphaC.simplices, child) # a-b
                            # Simplex tree starts out with 1 init_depot_path
                            # Only edges are stored so if 1 is connected to 1-2-3-4 we will have 
                            # 1-2, 1-3, 1-4, 2-3, 3-4..., so we will need to pick gchild as child from root node, and then look
                            node_3 = STs.find_child(AlphaC.simplices, gchild) # a-c
                            node_5 = STs.find_child(node_2, gchild) # b-C
                            min_1 = min(node_2.birth, node_3.birth, node_5.birth)
                            if (min_1 < (1.73205 * Ball_radius))
                                push!(tetrahedrons_or_tri, [sv[1], sv[2], sv[3]])
                            end
                        end
			            if dim <=3 && edge_2_4 && !([sv[2], sv[3], sv[4]] in tetrahedrons_or_tri)
                            node_2 = STs.find_child(AlphaC.simplices, child)
                            node_5 = STs.find_child(node_2, gchild) # b-C
                            node_6 = STs.find_child(node_2, ggchild) # b-d
                            node_3 = STs.find_child(AlphaC.simplices, gchild)
                            node_7 = STs.find_child(node_3, ggchild)
                            min_1 = min(node_5.birth, node_6.birth, node_7.birth)
                            if (min_1 < (1.73205 * Ball_radius))
                                push!(tetrahedrons_or_tri, [sv[2], sv[3], sv[4]])
                            end
                        end
			            if dim <= 3 && edge_1_4 && edge_2_4 && !([sv[1], sv[2], sv[4]] in tetrahedrons_or_tri)
                            node_1 = STs.find_child(AlphaC.simplices, simplices_0)
                            node_2 = STs.find_child(AlphaC.simplices, child)   # 1-2
                            node_4 = STs.find_child(AlphaC.simplices, ggchild) # 1-4
                            node_6 = STs.find_child(node_2, ggchild)           # 2-4
                            min_1 = min(node_2.birth, node_4.birth, node_6.birth)
                            if (min_1 < (1.73205 * Ball_radius))
                                push!(tetrahedrons_or_tri, [sv[1], sv[2], sv[4]])
                            end
                        end
			            if dim <= 3 && edge_1_3 && edge_1_4 && !([sv[1], sv[3], sv[4]] in tetrahedrons_or_tri)
                            node_3 = STs.find_child(AlphaC.simplices, gchild) # 1-3
                            node_4 = STs.find_child(AlphaC.simplices, ggchild) # 1-4
                            node_7 = STs.find_child(node_3, ggchild)           # 3-4
                            min_1 = min(node_3.birth, node_4.birth, node_7.birth)
                            if (min_1 < (1.73205 * Ball_radius))
                                push!(tetrahedrons_or_tri, [sv[1], sv[3], sv[4]])
                            end
                        end
                    end
                end
            end
        end
    end
    return tetrahedrons_or_tri
end

end
