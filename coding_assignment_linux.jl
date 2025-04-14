using CombinatorialSpaces
using StaticArrays
using RowEchelon
using SparseArrays
using LinearAlgebra
using Printf, Format
using SmithNormalForm
using LinearSolve
#using Nemo 
using StatsBase 
using Plots
using NearestNeighbors
using DataStructures

using CSV
using DataFrames
using DelimitedFiles
using Distances

using Debugger
#using GLMakie

include("./va.jl")
using .va             # Vinay's module implementing SimplexTree (modeled after Hodge) storing all simplices, VRComplex, AlphaComplex etc...
                      # SimplicialSets also stores edges, vertices etc... I will use SimplexTree essentially for filtration.

# Note of SNF vs RREF #####################
"""
For full matrices, the algorithm is based on the vectorization of MATLAB's RREF function. A typical speed-up range is about 2-4 times of the MATLAB's RREF function. However, the actual speed-up depends on the size of A. The speed-up is quite considerable if the number of columns in A is considerably larger than the number of its rows or when A is not dense.

For sparse matrices, the algorithm ignores the tol value and uses sparse QR to compute the rref form, improving the speed by a few orders of magnitude.

Authors: Armin Ataei, Ashish Myles
Cite As

Armin Ataei (2025). Fast Reduced Row Echelon Form (https://www.mathworks.com/matlabcentral/fileexchange/21583-fast-reduced-row-echelon-form), MATLAB Central File Exchange. Retrieved April 12, 2025.
"""
# We filter 2*radius i.e. at diameter level.
#Ball_radius = 7.0
Ball_radius = 5.0
#Ball_radius = 3.5
#Ball_radius = 3.9
#Ball_radius = 2.5
#Ball_radius = 1.5
distance_threshold = 2*(Ball_radius)
const Point3D = SVector{3,Float64}

#=====================================================================
const     Cow = 1
const     Rabbit = 2
const     Horse = 3
const     Dog = 4
const     Fish = 5
const     Dolphin = 6
const     Oyster = 7
const     Broccoli = 8
const     Fern = 9
const     Onion = 10
const     Apple = 11

function pchains(Ks, p1,p2, p3, dim)
 
  # Get Simplicial complex structure.
  tt1 = 0
  if dim == 3
    tt1 = nsimplices(3, Ks) # no of tereahedrons
  end
  tt = nsimplices(2,Ks)   # no of triangles
  te = nsimplices(1,Ks)   # no of edges
  tv = nsimplices(0,Ks)   # no of vertices

  print("Structure : ", "Tet : ", tt1, "Tri : ", tt, "Edges : ", te, "Vertices : ", tv)

  # first C3
  if dim == 3
     # Get all tetrahedron in simplex. 
    ntetra = ntetrahedra(Ks)
    if ntetra != 0
      for nt in ntetra
        # show p-chains <triangles>
        #@bp
        # Terahedron to triangles:
        # Query kind of objects
        
        # Query all triangles from tetrahedron by index last parameter in a given simplex
        ttet_tri = tetrahedron_triangles(Ks, 1)
        # I can get Trichains for a tetrahedron : In calss this is called 2 chians.
        tri_orientation = ∂(Ks, TetChain([1]))::TriChain
        tetrahedron_triangles_in_vertices = Vector{Int}[]
        form_string = ' '
        # Library naturally stoeres triangles with edge references, however vertices are also
        # available, triangle_vertices is recommended way.
        # https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/blob/main/docs/src/simplicial_sets.md
        for i in eachindex(ttet_tri)
          tri_vert = triangle_vertices(Ks, i)
          push!(tetrahedron_triangles_in_vertices, tri_vert)
        end
        # All paths carrying vertex i
        
        println("\n")
        @printf("∂C3 2-Chains = (triangles in Vertices) ")
        for i in eachindex(ttet_tri)
          @printf("(%d).%s ", tri_orientation[i], "[" *join(tetrahedron_triangles_in_vertices[i], " ")*"]")
          if i != lastindex(ttet_tri) 
            @printf("[+] ")
          end
        end
        println("\n")
      end
    end
  end

  # Let us get some one chains for each triangle, following 1s are multipliers, if we took path twice
  # or triangle needs to be in reverse...for normal case we will just use 1s
  # Confirm to see if this adds up as closed_star output. 
  # tt is number of triangles
  TChains = Array{Int}(undef, tt)
  fill!(TChains, 1)
  println("TCHAINS : ", TChains)
  edge_orientation = ∂(Ks, TriChain(TChains))::EChain
  edges_in_vertices = Vector{Int}[]
  print("\n Edge_orientation ∂C2 1-Chains: ", edge_orientation)
  println("\n")
  @printf("\n∂C2 1-Chains = (Edge oriented sum) \n")
  # Edges are kept in src, dst Format
  for i in 1 : te
    e = edge_vertices(Ks, i)
    push!(edges_in_vertices, e)
  end
  # Now edges weighted sum based on boundary matrix.
  @show("\n Edges : ", edges_in_vertices)
  
  @show("\n Weighted sum E : ")
  for i in 1:te
    @printf("(%d).%s ", edge_orientation[i], "[" *join(edges_in_vertices[i], " ")*"]")
    if i != te 
      @printf("[+] ")
    end
  end
  println("Done with Edges Sum... ∂C2 1-Chains of this simplex.\n")

  # We use closed_star operation
  path1 = closed_star(Ks, p1) #1,2 - gives same path just vertex switched.
  path5 = closed_star(Ks, p2) #5,6
  path8 = closed_star(Ks, p3) #8,9
  triangles_count = ntriangles(Ks)
  for nt in triangles_count
    #tri_orientation_next = ∂(Ks, TriChain([1,-1,1,-1,1,-1,1,-1,1]))::EChain
    @printf("∂C2 1-Chains (Collapsed to vertices) = (connected paths involving vertices) ")
    println(path1[1],"-", path5[1], "-", path8[1])
  end
end

Ks = OrientedDeltaSet3D{Bool}()
add_vertices!(Ks,11)
# 16 edges
# Vertex 2 is src, Vertex 1 is tgt.
add_edge!(Ks, Cow, Rabbit, edge_orientation=true)
add_edge!(Ks, Cow, Horse, edge_orientation=true ) 
add_edge!(Ks, Cow, Dog, edge_orientation=true)
add_edge!(Ks, Rabbit, Horse, edge_orientation=true) 
add_edge!(Ks, Rabbit, Dog, edge_orientation=true)
add_edge!(Ks, Horse, Dog, edge_orientation=true ) 

add_edge!(Ks, Fish, Dolphin, edge_orientation=true) 
add_edge!(Ks, Fish, Oyster, edge_orientation=true) 
add_edge!(Ks, Dolphin, Oyster, edge_orientation=true)

add_edge!(Ks, Broccoli, Fern, edge_orientation=true ) 
add_edge!(Ks, Broccoli, Onion, edge_orientation=true) 
add_edge!(Ks, Broccoli, Apple, edge_orientation=true ) 
add_edge!(Ks, Fern, Onion, edge_orientation=true)
add_edge!(Ks, Fern, Apple, edge_orientation=true) 
add_edge!(Ks, Onion, Apple, edge_orientation=true ) 


# glue_sorted adds minimal edges, glue_triangle may add extra edges to satisfy simplex identities 
# if orientations were not followed right!!

# REALIZING TOPOLOGY by Gluing; Transforming Simplicial Set to Simplicial Complex (geometric meanings).
 
glue_triangle!(Ks, Cow, Rabbit, Horse, tri_orientation=true) #, tri_orientation=true)
glue_triangle!(Ks, Cow, Rabbit, Dog, tri_orientation=true)   #, tri_orientrtion=true ) # edge_orientation=[true,true,true])
glue_triangle!(Ks, Cow, Horse, Dog, tri_orientation=true)    #, tri_orientation=true ) 
glue_triangle!(Ks, Rabbit, Horse, Dog, tri_orientation=true) # tri_orientation=true )# edge_orientation=[true,true,true])

# This will put a filled tetrahedron producing H0 - 3, H1 - 0, H2 - 1
# Just following problem text is sufficient.
# Gluing goes v0-v1-v2, v0-v1-v3, v0-v2-v3, v1-v2-v3 for tetrahedron
#add_tetrahedron!(Ks, Cow, Rabbit, Horse, Dog) # Changed glue to add.

# Isolated Triangle
glue_triangle!(Ks, Fish, Dolphin, Oyster, tri_orientation=true )

# 2 Triangles
glue_triangle!(Ks, Broccoli, Fern, Onion, tri_orientation=true ) # edge_orientation=[true,true,true])
glue_triangle!(Ks, Broccoli, Fern, Apple, tri_orientation=true )
glue_triangle!(Ks, Broccoli, Onion, Apple, tri_orientation=true) # tri_orientation=true)
glue_triangle!(Ks, Fern, Onion, Apple, tri_orientation=true ) # tri_orientation=true)

#orient_component!(Ks, 1, true)
show(Ks)

# Homology computation
# 11 Vertices {C0}
C_0 = size(simplices(0, Ks))[1]
# 15 edges, 1st vector has source (vertex-2), 2nd vector has (vertex-1) 
# {C1}

# Number of Edges
C_1 = edge_vertices(Ks)
C_1 = size(C_1[1])[1]

# Triangles have 1st vector carrying vertices 1 (of all triangles)
# Triangles have 2nd vector carrying vertices 2 (of all triangles)
# Triangles have 3rd vector carrying vertices 3 (of all triangles)
# {C2}

# Number of Triangles
C_2 = ntriangles(Ks)

# Number of Tetrahedra in Simplicial Complex
C_3 = ntetrahedra(Ks)

#@bp
dim = 3
pchains(Ks, 1, 5, 8, dim)
# Boundary map matrix - 1
# Vertex - Edge Matrix {vertex are in column, edges are in row}
# Sign eventually comes out correct as dst, src scheme is used, with [1,-1,1] sign array
# instead of eject first so 2nd element gets 1, 3rd gets -1 etc... 
# In this implementation we can just multiply with sign array SVector [1,-1,1]

# Boundary Matrix Vertex to Edge Matrix.
# The boundary operator on `n`-faces and `n`-chains is implemented by the call
# See https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/blob/main/src/SimplicialSets.jl 
# Line #694.
#```julia
#∂(n, s, ...)
#```
#.        11 x 15 Matrix of 11 vertices and 15 Edges.
#         |<--------- Rank C_p ------------>
#              |<--- Rank Z_p ------------->
#. ^      ----------------------------------    ^
#  B(p-1) |\   |                           |    |
#  |      | \  |                           |    Rank (C(p-1))
#  |      |  \ |                           |    |
#  V      |   \|                           |    |
#         |                                |    |
#         |                                |    |
#         |                                |    |
#         ----------------------------------    V
Delta_1 = ∂(1, Ks)
println("Delta_1 Matrix...")
display(Delta_1)
# Compute SNF (SmithNormForm)
SNF_Delta_1 = smith(Matrix(Delta_1))  # This is from SmithNormalForm
#SNF_Delta_1 = snf(Matrix(Delta_1))     # This is from Nemo
#println("SNF delta_1 Matrix...")
#@show(diagm(SNF_Delta_1))
# Compute RREF
rref_Delta_1 = rref_with_pivots(Matrix(Delta_1))
#println("RREF delta_1 Matrix...")
#display(rref_Delta_1[1])
b = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0;;]
#prob = LinearProblem(SNF_Delta_1, b)
#sol = solve(prob)
#X = rref_Delta_1[1] \ b
X = diagm(SNF_Delta_1) \ b
#println("Display Solution of del_1 Ax = 0")
# solution is extracted via .u
#display(X)
rref_Delta_1_int = rref_Delta_1[1] #round.(Int, rref_Delta_1[1])
#null_vectors_d1 = nullspace(rref_Delta_1_int)   # cycle crearting edges
# -----------------------------------------------------------------------------------------
# RREF(A).C = SNF gives zero fiber shifted by RREF(A) giving solution where C gives Kernel
# F.S is left side
# F.T is right side
# diagm(F) is smith normal matrix.
# F.S*diagm(F)*F.T = original matrix.
# -----------------------------------------------------------------------------------------
null_vectors_d1 = nullspace(rref_Delta_1_int)
#println("Null vectors Edges i.e. Vectors spanning Kernel")
#display(null_vectors_d1)
# Get Pivots (in 2)-first_element - Julia Arrays start at 1, unlike C++ 0 based.
rank_B_p_0 = size(rref_Delta_1[2])[1]  # number of pivots = Image(del1)
rank_Cp_1 = size(Delta_1, 2)           # number of rows
rank_Cp_0 = size(Delta_1, 1)           # number of columns
rank_Zp_1 = rank_Cp_1 - rank_B_p_0     # Kernel(del1) -- [ Hom h_i = ker(del_i) - image(del_(i+1)) ]
#println("Rank B0 : ", rank_B_p_0, " Rank C1 : ", rank_Cp_1, " Rank Z1 : ", rank_Zp_1)
rref_d1 = sparse(rref_Delta_1[1])
println("Rank B0 : ", rank_B_p_0, " Rank C0 : ", rank_Cp_0,  " Rank C1 : ", rank_Cp_1, " Rank Z1 : ", rank_Zp_1)

# 15x9 - Boundary Matrix - Edges - Triangle Matrix
# The boundary operator on `n`-faces and `n`-chains is implemented by the call
# See https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/blob/main/src/SimplicialSets.jl 
# Line #694.
#```julia
#∂(n, s, ...)
#```
#        15 x 9 Matrix 15 Edges and 9 triangles.
#
Delta_2 = ∂(2, Ks)
println("Delta_2 Matrix...")
display(Delta_2)
# Compute RREF
rref_Delta_2 = rref_with_pivots(Matrix(Delta_2))
#display(rref_Delta_2[1])
b = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0;;]
# Compute SNF (SmithNormForm)
SNF_Delta_2 = smith(Matrix(Delta_2))  # This is from SmithNormalForm
#println("SmithNormalForm.. to get nullity of kernel")
#@show(SNF_Delta_2)
#X = rref_Delta_2[1] \ b
X = diagm(SNF_Delta_2) \ b
#println("Display Solution of del_2 Ax = 0")
#display(X)
rref_Delta_2_int = round.(Int, rref_Delta_2[1])
display(rref_Delta_2_int)
null_vectors_d2 = nullspace(rref_Delta_2_int)  # cycle creating triangles
#println("Null vectors Triangles i.e. Vectors spanning Kernel")
#display(null_vectors_d2)
# Get Pivots (in 2)-first_element - Julia Arrays start at 1, unlike C++ 0 based.
rank_B_p_1 = size(rref_Delta_2[2])[1] # Number of Pivots = Image(del_2)
rank_Cp_2 = size(Delta_2, 2)
rank_Zp_2 = rank_Cp_2 - rank_B_p_1    # Kernel (del2)
#println("Rank B1 : ", rank_B_p_1, " Rank C2 : ", rank_Cp_2, " Rank Z2 : ", rank_Zp_2)
rref_d2 = sparse(rref_Delta_2[1])
println("Rank B1 : ", rank_B_p_1, " Rank C2 : ", rank_Cp_2, " Rank Z2 : ", rank_Zp_2)

# Boundary Matrix - Triangles to Tetrahedron Matrix
# The boundary operator on `n`-faces and `n`-chains is implemented by the call

# See https://github.com/AlgebraicJulia/CombinatorialSpaces.jl/blob/main/src/SimplicialSets.jl 
# Line #694.
#```julia
#∂(n, s, ...)
#```
Delta_3 = ∂(3, Ks)
println("Delta_3 Matrix...")
display(Delta_3)
# Compute RREF
rref_Delta_3 = rref_with_pivots(Delta_3)
rank_B_p_2 = size(rref_Delta_3[2])[1] # Number of Pivots = Image(del_3)
rank_Cp_3 = size(Delta_3, 2)
rank_Zp_3 = rank_Cp_3 - rank_B_p_2    # Kernel (del3)
#println("Rank B1 : ", rank_B_p_1, " Rank C2 : ", rank_Cp_2, " Rank Z2 : ", rank_Zp_2)
rref_d3 = sparse(rref_Delta_3[1])
println("Rank B2 : ", rank_B_p_2, " Rank C3 : ", rank_Cp_3, " Rank Z3 : ", rank_Zp_3, " Refresh display [ ", display(rref_d3), " ] ")

# We can now compute all homologies H_0, H_1, H_2
h0 = C_0 - rank_B_p_0
h1 = rank_Zp_1 - rank_B_p_1
h2 = rank_Zp_2 - rank_B_p_2
println("------------------------------------------------")
println("Homologies : h0 ", h0, " h1 ", h1, " h2 ", h2)
println("------------------------------------------------")


#############################################################
## 2nd Simplex ##
#############################################################
K2s = OrientedDeltaSet3D{Bool}()
add_vertices!(K2s,11)

#const     Cow = 1
#const     Rabbit = 2
#const     Horse = 3
#const     Dog = 4
#const     Fish = 5
#const     Dolphin = 6
#const     Oyster = 7
#const     Broccoli = 8
#const     Fern = 9
#const     Onion = 10
#const     Apple = 11


#K1 = SimplicialComplex([[Cow, Rabbit], [Cow, Fish], [Cow, Oyster], [Cow, Oyster], [Cow, Broccoli], [Cow, Onion], [Cow, Apple],
#                        [Rabbit, Fish], [Rabbit, Oyster], [Rabbit, Broccoli], [Rabbit, Onion], [Rabbit, Apple], [Fish, Oyster],
#                        [Fish, Broccoli], [Fish, Onion], [Fish, Apple], [Oyster, Broccoli], [Oyster, Onion], [Oyster, Apple],
#                        [Broccoli, Onion], [Broccoli, Apple], [Onion, Apple], [Horse, Dog], [Horse, Dolphin], [Horse, Fern], [Dog, Dolphin],
#                        [Dog, Fern], [Dolphin, Fern], [Cow, Broccoli, Apple], [Cow, Onion, Apple], [Rabbit, Broccoli, Apple], [Rabbit, Onion, Apple],
#                        [Fish, Broccoli, Apple], [Fish, Onion, Apple], [Oyster, Broccoli, Apple], [Oyster, Onion, Apple]])

add_edge!(K2s, Cow, Rabbit, edge_orientation=true)
add_edge!(K2s, Cow, Fish, edge_orientation=true)
add_edge!(K2s, Cow, Oyster, edge_orientation=true)
#add_edge!(K2s, Cow, Oyster, edge_orientation=true) ## Not adding this extra edge (Available in printout? Matrix gores from 11x27 to 11x28)
add_edge!(K2s, Cow, Broccoli, edge_orientation=true)
add_edge!(K2s, Cow, Onion, edge_orientation=true)
add_edge!(K2s, Cow, Apple, edge_orientation=true)

add_edge!(K2s, Rabbit, Fish, edge_orientation=true)
add_edge!(K2s, Rabbit, Oyster, edge_orientation=true)
add_edge!(K2s, Rabbit, Broccoli, edge_orientation=true)
add_edge!(K2s, Rabbit, Onion, edge_orientation=true)
add_edge!(K2s, Rabbit, Apple, edge_orientation=true)

add_edge!(K2s, Fish, Oyster, edge_orientation=true)
add_edge!(K2s, Fish, Broccoli, edge_orientation=true)
add_edge!(K2s, Fish, Onion, edge_orientation=true)
add_edge!(K2s, Fish, Apple, edge_orientation=true)

add_edge!(K2s, Oyster, Broccoli, edge_orientation=true)
add_edge!(K2s, Oyster, Onion, edge_orientation=true)
add_edge!(K2s, Oyster, Apple, edge_orientation=true)
add_edge!(K2s, Broccoli, Onion, edge_orientation=true)
add_edge!(K2s, Broccoli, Apple, edge_orientation=true)
add_edge!(K2s, Onion, Apple, edge_orientation=true)

add_edge!(K2s, Horse, Dog, edge_orientation=true)
add_edge!(K2s, Horse, Dolphin, edge_orientation=true)

add_edge!(K2s, Horse, Fern, edge_orientation=true)
add_edge!(K2s, Dog, Dolphin, edge_orientation=true)

add_edge!(K2s, Dog, Fern, edge_orientation=true)
add_edge!(K2s, Dolphin, Fern, edge_orientation=true)

# REALIZING TOPOLOGY by Gluing; Transforming Simplicial Set to Simplicial Complex (geometric meanings).
 
glue_triangle!(K2s, Cow, Broccoli, Apple, tri_orientation=true) #, tri_orientation=true)
glue_triangle!(K2s, Cow, Onion, Apple, tri_orientation=true)   #, tri_orientrtion=true ) # edge_orientation=[true,true,true])

glue_triangle!(K2s, Rabbit, Broccoli, Apple, tri_orientation=true)    #, tri_orientation=true ) 
glue_triangle!(K2s, Rabbit, Onion, Apple, tri_orientation=true) # tri_orientation=true )# edge_orientation=[true,true,true])
glue_triangle!(K2s, Fish, Broccoli, Apple, tri_orientation=true)    #, tri_orientation=true ) 
glue_triangle!(K2s, Fish, Onion, Apple, tri_orientation=true) # tri_orientation=true )# edge_orientation=[true,true,true])
glue_triangle!(K2s, Oyster, Broccoli, Apple, tri_orientation=true)    #, tri_orientation=true ) 
glue_triangle!(K2s, Oyster, Onion, Apple, tri_orientation=true) # tri_orientation=true )# edge_orientation=[true,true,true])

show(K2s)

# p-chains
pchains(K2s, 1, 3, 8, 2)
# Vertices to Edge Boundary map
K2s_C_0 = size(simplices(0, K2s))[1]  # Same number of vertices...
K2s_Delta_1 = ∂(1, K2s)
display(K2s_Delta_1)
show(IOContext(stdout, :limit => true, :displaysize => (100, 100)), "text/plain", K2s_Delta_1)
K2s_rref_Delta_1 = rref_with_pivots(Matrix(K2s_Delta_1))
#display(K2s_rref_Delta_1[1])
# for 11x27 matrix 11 vertices , 27 edges b = 27x1 column matrix 
b = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0;;]
#display(b)
X = K2s_rref_Delta_1[1] \ b
println("K2s Display Solution of del_1 Ax = 0")
#display(X)
K2s_rref_Delta_1_int = round.(Int, K2s_rref_Delta_1[1])
K2s_null_vectors_d1 = nullspace(K2s_rref_Delta_1_int)   # cycle creating edges
#println("Null vectors Edges i.e. Vectors spanning Kernel")
##display(K2s_null_vectors_d1)
# Get Pivots (in 2)-first_element - Julia Arrays start at 1, unlike C++ 0 based.
K2s_rank_B_p_0 = size(K2s_rref_Delta_1[2])[1]  # number of pivots
K2s_rank_Cp_1 = size(K2s_Delta_1, 2)           # number of rows
K2s_rank_Cp_0 = size(K2s_Delta_1, 1)           # number of columns
K2s_rank_Zp_1 = K2s_rank_Cp_1 - K2s_rank_B_p_0
#println("Rank B0 : ", rank_B_p_0, " Rank C1 : ", rank_Cp_1, " Rank Z1 : ", rank_Zp_1)
K2s_rref_d1 = sparse(K2s_rref_Delta_1[1])
println("Rank B0 : ", K2s_rank_B_p_0, " Rank C0 : ", K2s_rank_Cp_0,  " Rank C1 : ", K2s_rank_Cp_1, " Rank Z1 : ", K2s_rank_Zp_1)
outfile = "K2s_Delta_1.txt"
open(outfile, "w") do f
    for i in eachrow(K2s_Delta_1)
      println(f, i)
    end
end # the file f is automatically closed after this block finishes

# No longer printing rref matrix.
#outfile = "K2s_rref_d1.txt"
#open(outfile, "w") do f
#  for i in eachrow(K2s_rref_d1)
#    println(f, i)
#  end
#end # the file f is automatically closed after this block finishes

K2s_Delta_2 = ∂(2, K2s)
outfile = "K2s_Delta_2.txt"
open(outfile, "w") do f
    for i in eachrow(K2s_Delta_2)
      println(f, i)
    end
end # the file f is automatically closed after this block finishes
# Compute RREF
K2s_rref_Delta_2 = rref_with_pivots(Matrix(K2s_Delta_2))
#display(K2s_rref_Delta_2[1])
# 27 edges x 8 triangles Matrix
b = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0;;]
X = K2s_rref_Delta_2[1] \ b
#println("K2s Display Solution of del_2 Ax = 0")
#display(X)
K2s_rref_Delta_2_int = round.(Int, K2s_rref_Delta_2[1])
K2s_null_vectors_d2 = nullspace(K2s_rref_Delta_2_int)  # cycle creating triangles
#println("Null vectors Triangles i.e. Vectors spanning Kernel")
#display(K2s_null_vectors_d2)
# Get Pivots (in 2)-first_element - Julia Arrays start at 1, unlike C++ 0 based.
K2s_rank_B_p_1 = size(K2s_rref_Delta_2[2])[1] # Number of Pivots
K2s_rank_Cp_2 = size(K2s_Delta_2, 2)
K2s_rank_Zp_2 = K2s_rank_Cp_2 - K2s_rank_B_p_1
#println("Rank B1 : ", rank_B_p_1, " Rank C2 : ", rank_Cp_2, " Rank Z2 : ", rank_Zp_2)
K2s_rref_d2 = sparse(K2s_rref_Delta_2[1])
println("Rank B1 : ", K2s_rank_B_p_1, " Rank C2 : ", K2s_rank_Cp_2, " Rank Z2 : ", K2s_rank_Zp_2)
#show(IOContext(stdout, :limit => true), "text/plain", K2s_rref_d2)

# no longer printing rref matrix
#outfile = "K2s_rref_d2.txt"
#open(outfile, "w") do f
#  for i in eachrow(K2s_rref_d2)
#    println(f, i)
#  end
#end # the file f is automatically closed after this block finishes

K2s_rank_B_p_2 = 0
# Computing Homologies
kh0 = K2s_C_0 - K2s_rank_B_p_0
kh1 = K2s_rank_Zp_1 - K2s_rank_B_p_1
kh2 = K2s_rank_Zp_2 - K2s_rank_B_p_2
println("2nd Simples Homologies : h0 ", kh0, " h1 ", kh1, " h2 ", kh2)
==========================================================#
## Coding Assignment 3

# Load datasets into arrays.

# read dataframe
#tab_1_matrix = readdlm("/users/vaw1/Downloads/Topology_3/CDHWdata_1.csv", ',') 
#tab_2_matrix = readdlm("/users/vaw1/Downloads/Topology_3/CDHWdata_2.csv", ',') 
#tab_3_matrix = readdlm("/users/vaw1/Downloads/Topology_3/CDHWdata_3.csv", ',')
#tab_4_matrix = readdlm("/users/vaw1/Downloads/Topology_3/CDHWdata_4.csv", ',')
#tab_5_matrix = readdlm("/users/vaw1/Downloads/Topology_3/CDHWdata_5.csv", ',') 
tab_5_matrix = readdlm("./CDHWdata_3.csv", ',') 

#tab_1_matrix_f = Float64.(tab_1_matrix[2:end, 2:end])
#tab_2_matrix_f = Float64.(tab_2_matrix[2:end, 2:end])
#tab_3_matrix_f = Float64.(tab_3_matrix[2:end, 2:end])
#tab_4_matrix_f = Float64.(tab_4_matrix[2:end, 2:end])
tab_5_matrix_f = Float64.(tab_5_matrix[2:end, 2:end])

@show(tab_5_matrix_f[1,:])
# distance function
# Using columnwise distance functions which are much faster that computing distances in 
# a loop
# In the output, R is a matrix of size (m, n), such that R[i,j] is the distance between X[:,i] and Y[:,j]. 
# Computing distances for all pairs using pairwise function is often remarkably faster than evaluting for each pair individually.
# We transpose this Matrix to get each sample as column and then 
tab_5_matrix_t = tab_5_matrix_f'
#tab_2_matrix_t = tab_2_matrix_f'
#@show(tab_5_matrix_t[:,1], size(tab_5_matrix_t), size(tab_2_matrix_t))
# Julia prefers columns vectors as vectors.
# We now have 93 samples organized in columns and dimension of each vector (15 - x,y,z,u,v,w...) 
# are rows.
# Following function can compute columnwise distances for all columns witin a matrix.
# Col1-col2, col1-col3, col1-col3.. = col1
# col2-col1, col2-col2, col2-col3.. = col2
R_5 = pairwise(Euclidean(), tab_5_matrix_t, dims=2)
#R_2 = pairwise(Euclidean(), tab_2_matrix_t, dims=2)
n = size(R_5)
println(n)
#m = size(R_2)
#println(n, m)
println(R_5[1:5,1:5]) # see entries in each column as distances col1-col1 = 0, col1-col2 = col2-col1, col3-col1 = col1-col3 etc..
# See column rank (1st point distance wrt to other 92 points)
println("Coliumn-Ranks : ", R_5[:,1])

rank_R_5 = rank(R_5)  # Gives number of linearly independent rows.
println("Rank of R_5 = ", rank_R_5)
# order of distances of points within each column, (transform given matrix using (anonymous) function f)
# mapslice works over rows comparing rank so Adjoint operator (R_5' vs R_5) is used.
R_5_transpose = R_5'   # All point distances are in horizontal format.
println("Coliumn-Ranks In Row Format: ", R_5_transpose[1:1,:])
ordinal_rank_R_5 = StatsBase.mapslices(r -> ordinalrank(r, rev=false), R_5_transpose, dims=2)
display(ordinal_rank_R_5[1:1,:]) # 1st row ranks first point diffed against all.

# Build Skeleton, while sparsifying
# we will simply number these vecors as 1..93 for labeling purposes
vertices_csv = collect(1:size(R_5_transpose,2))
vertices_csv_3 = collect(2:size(R_5_transpose,2))
# construct skeleton ball radius = 2*r as I will use the same construction for Alpha
B = va.VR.VRComplex()

# insert all edges falling under this threshold.
K5s = OrientedDeltaSet3D{Bool}()
add_vertices!(K5s,93) # 93 samples
#construct graph first
for (ind,v) in pairs(R_5_transpose)
  i,j = Tuple(ind)
  if i <= j
    continue
  else
    if v < distance_threshold
      # add edges
      # println(ind,v)
      # add_edge works dst, src
      add_edge!(K5s, j, i, edge_orientation=true)
      # put it in VRComplex
      insert!(B,[i,j])
    end
  end
end

# Get all neighbors using simplex-tree of dimension l
dim_3_simplicial_complexes = va.VR.gathersimplices(B, 3)
# inject all these triangles and tetrahedrons into our simplicial complex for this filtration diameter. (1/2 of diameter = B_r = Ball radius)
removed_duplicates = unique(dim_3_simplicial_complexes)
# ignore dense simplices.
process_this_simplex=false
simplex_key = Dict{Vector{Int}, Int}()
simplex_has_tri = Dict{Vector{Int}, Int}()
for simplices in dim_3_simplicial_complexes
  # manage existing triangles in multi-clique structures.
  # We need to have a strategy for reduction
  # we get 300,000+ triangles from objects such as 
  # 1-2-3-4, 1-2-3-5, 1-2-3-6, 1-2-3-7, 1-2-3-8, 1-2-3-9, 1-2-3-10....
  # These objects also have 4-5, 5-6, 6-7, 7-8 etc connected.
  # 
  # -----------------------------------------------
  # we will condense these objects to A POINT.
  # -----------------------------------------------
  # run a loop comparing first 3 vertices and condense
  if !haskey(simplex_key, [simplices[1], simplices[2], simplices[3]])
    # new 3 rooted simplex. (i.e. first 3 entries in simplex tree are not the same)
    simplex_key[[simplices[1], simplices[2], simplices[3]]] = 1
    global process_this_simplex=true
  else
    global process_this_simplex=false
  end
  if process_this_simplex == false
    # only process simplices which might have any homological features.
    # ignore 1-2-3-4, 1-2-3-5, 1-2-3-6...1-2-3-80, as these also have 4-5-6 edges connected 
    # making it a glued tetrahedron cluster. reduced triangles from 96000+ to 16000...
    continue
  end
  if size(simplices)[1] == 5 || size(simplices)[1] == 4 
    # Gluing goes v0-v1-v2, v0-v1-v3, v0-v2-v3, v1-v2-v3 for tetrahedron
    # Julia arrays are indexed at 1.
    # glue_triangle checks for existence of edges.
    # we get stream of higher order structures 1-2-3-[24..62], we do not need to add existing triangles.
    println("Processing - ", simplices)
    if size(simplices)[1] == 4
        if !haskey(simplex_has_tri, [simplices[1], simplices[2], simplices[3]])
          #println("123", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[3]]] = 1
        else
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=false) #, tri_orientation=true)
	  delete!(simplex_has_tri, [simplices[1], simplices[2], simplices[3]])
	end
	  
        if !haskey(simplex_has_tri, [simplices[1], simplices[2], simplices[4]])
          #println("124", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[4]]] = 1
        else
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[4], tri_orientation=true) #, tri_orientation=true)
	  delete!(simplex_has_tri, [simplices[1], simplices[2], simplices[4]])
        end
        if !haskey(simplex_has_tri, [simplices[1], simplices[3], simplices[4]])
          #println("134", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[3], simplices[4], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[3], simplices[4]]] = 1
        else
          glue_triangle!(K5s, simplices[1], simplices[3], simplices[4], tri_orientation=false) #, tri_orientation=true)
	  delete!(simplex_has_tri, [simplices[1], simplices[3], simplices[4]])
        end
        if !haskey(simplex_has_tri, [simplices[2], simplices[3], simplices[4]])
          #println("234", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[2], simplices[3], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[2], simplices[3], simplices[4]]] = 1
        else
          glue_triangle!(K5s, simplices[2], simplices[3], simplices[4], tri_orientation=true) #, tri_orientation=true)
	  delete!(simplex_has_tri, [simplices[2], simplices[3], simplices[4]])
        end
    end
    # I was trying 5D it got very expensive so code is here nut it is not getting called
    ### 2nd tetra ### 234 shared face -ve
    if size(simplices)[1] == 5
        if !haskey(simplex_has_tri, [simplices[2], simplices[3], simplices[4]])
          #println("123", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[3]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[2], simplices[3], simplices[5]])
          #println("124", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[4]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[2], simplices[4], simplices[5]])
          #println("134", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[3], simplices[4], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[3], simplices[4]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[3], simplices[4], simplices[5]])
          #println("234", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[2], simplices[3], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[2], simplices[3], simplices[4]]] = 1
        end
    end
    ### 3rd tetra ### 134 shared face -ve
    if size(simplices)[1] == 5
        if !haskey(simplex_has_tri, [simplices[1], simplices[3], simplices[4]])
          #println("123", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[3]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[1], simplices[3], simplices[5]])
          #println("124", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[2], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[2], simplices[4]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[1], simplices[4], simplices[5]])
          #println("134", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[1], simplices[3], simplices[4], tri_orientation=true) #, tri_orientation=true)
          simplex_has_tri[[simplices[1], simplices[3], simplices[4]]] = 1
        end
        if !haskey(simplex_has_tri, [simplices[3], simplices[4], simplices[5]])
          #println("234", simplices[1], simplices[2], simplices[3]) 
          glue_triangle!(K5s, simplices[2], simplices[3], simplices[4], tri_orientation=false) #, tri_orientation=true)
          simplex_has_tri[[simplices[2], simplices[3], simplices[4]]] = 1
        end
    end

    # glue_tetrahedron checks for existence of triangles.
    if size(simplices)[1] == 5
        glue_tetrahedron!(K5s, simplices[1], simplices[2], simplices[3], simplices[4], tet_orientation=true) 
        glue_tetrahedron!(K5s, simplices[2], simplices[3], simplices[4], simplices[5], tet_orientation=false) 
        glue_tetrahedron!(K5s, simplices[1], simplices[3], simplices[4], simplices[5], tet_orientation=false) 
    end
    if size(simplices)[1] == 4
	if !haskey(simplex_has_tri, [simplices[1], simplices[2], simplices[3], simplices[4]])
          glue_tetrahedron!(K5s, simplices[1], simplices[2], simplices[3], simplices[4], tet_orientation=true)
	else
          glue_tetrahedron!(K5s, simplices[1], simplices[2], simplices[3], simplices[4], tet_orientation=false)
	  # many structure in this simplex collapse due to orientation.
	  # delete key in case it gets inserted 3rd time + 
	  delete!(simplex_has_tri, [simplices[1], simplices[2], simplices[3], simplices[4]])
          println("-ve orientation", simplices)
	end
	  
    end

    # My simplex tree.
    insert!(B,simplices, Ball_radius)
  else
    if !haskey(simplex_has_tri, [simplices[1], simplices[2], simplices[3]])
      glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=true) #, tri_orientation=true)
      simplex_has_tri[[simplices[1], simplices[2], simplices[3]]] = 1
    else
      glue_triangle!(K5s, simplices[1], simplices[2], simplices[3], tri_orientation=false) #, tri_orientation=true)
      # many structure in this simplex collapse due to orientation.
      # delete key in case it gets inserted 3rd time + 
      delete!(simplex_has_tri, [simplices[1], simplices[2], simplices[3]])
      println("-ve orientation", simplices)
    end
    insert!(B,simplices, Ball_radius)
  end
end

############## Simplicial Complex for filtration 1 done ####################
# Compute Homologies
############################################################################

######   ∂1   ###################
K5s_C_0 = size(simplices(0, K5s))[1]   # number of vertices.
println("Computing 1st boundary matrix...Vertices ", K5s_C_0, "Edges - ", size(simplices(1, K5s))[1], "Triangles - ", size(simplices(2, K5s))[1], "Tetrahedra - ", size(simplices(3, K5s))[1])
K5s_Delta_1 = ∂(1, K5s)
K5s_rref_Delta_1 = rref_with_pivots!(Matrix(K5s_Delta_1))
K5s_rank_B_p_0 = size(K5s_rref_Delta_1[2])[1] # number of pivots
K5s_rank_Cp_1  = size(K5s_Delta_1, 2)         # number of columns
K5s_rank_Cp_0  = size(K5s_Delta_1, 1)         # number of rows
K5s_rank_Zp_1  = K5s_rank_Cp_1 - K5s_rank_B_p_0
println("Rank B0 : ", K5s_rank_B_p_0, " Rank Cp1 : ", K5s_rank_Cp_1, " Rank Cp0 : ", K5s_rank_Cp_0, "Rank Z1 : ", K5s_rank_Zp_1)

######   ∂2   ###################
println("Computing 2nd boundary matrix...")
K5s_Delta_2 = ∂(2, K5s)
K5s_rref_Delta_2 = rref_with_pivots!(Matrix(K5s_Delta_2))
K5s_rank_B_p_1 = size(K5s_rref_Delta_2[2])[1] # Number of Pivots
K5s_rank_Cp_2 = size(K5s_Delta_2, 2)          # Number of columns
K5s_rank_Zp_2 = K5s_rank_Cp_2 - K5s_rank_B_p_1
println("Rank B1 : ", K5s_rank_B_p_1, " Rank Cp2 : ", K5s_rank_Cp_2, " Rank Cp2 : ", K5s_rank_Cp_2, "Rank Z2 : ", K5s_rank_Zp_2)

######   ∂3   ###################
K5s_Delta_3 = ∂(3, K5s)
K5s_rref_Delta_3 = rref_with_pivots!(Matrix(K5s_Delta_3))
K5s_rank_B_p_2 = size(K5s_rref_Delta_3[2])[1] # Number of Pivots

# Computing Homologies
kh0 = K5s_C_0 - K5s_rank_B_p_0
kh1 = K5s_rank_Zp_1 - K5s_rank_B_p_1
kh2 = K5s_rank_Zp_2 - K5s_rank_B_p_2
println("Filtration r = ", Ball_radius, "Simples Homologies : h0 ", kh0, " h1 ", kh1, " h2 ", kh2)

#B = va.VR.VRComplex([vertices_csv]) #va.RipsComplex(ordinal_rank_R_5, R_5_transpose, max_length = 6.6, ε=5)

#GLMakie.AbstractPlotting.inline!(false)
#fig, ax, ob = wireframe(K5s; color=(:blue,1), transparency=true, linewidth=3)
#display(fig)
@show("Dimension = ", va.VR.dimension(B), va.VR.numsimplices(B,0), va.VR.numsimplices(B,1), va.VR.numsimplices(B,2), va.VR.numsimplices(B,3))
#query skeleton to get structures of cliques (connected vertices)
#C = va.VR.skeleton(B, 3) # 3 dimensional simplices (i.e. triangles)
#if !(isempty(C))
#   @show(va.VR.numsimplices(C), va.VR.numvertices(B), va.VR.numsimplices(B))   # numvertices are 1-simplices
#end
#k = va.VR.simplices(B, 1) # gets all edges from simplex tree.
println("Done")
# construct simplical complex, increasing r values.
# First the easiest - The Vietoris–Rips complex is used to capture a notion of proximity
# in a metric space.

# Sparsification using SimplexTree and distance Matrix
# build a SimplexTree bringing complexity of simplicial complex creation down from 2^N (unbounded simplex) to O(N) by zeroing out data that is below threshold
# SimplexTree organizes distance data into a flat array with sorted branches for any given node.
# Paper - 

# Data needs to be in n_dim x n_points.
# leafsize allows for sparsification (when should one stop splitting) - cost of distance function.
# 
#ct = KDTree(tab_5_matrix_t, Euclidean(); 25, reorder=false)

# assemble edges
#ct_edges = sparse_ctree(ct, 1.0f0, -1, strategy=APriori())



# Do ordering to setup points in increasing order.

# Rank tell distances i.e. in matrix 
# one point
# r_51 = Euclidean()(tab_5_matrix_t[:,1], tab_5_matrix_t[:,1])
# display(r_51)
# Using Wasserstein distance.
#=
You are going to examine some data and find persistent homologies, and do some simple analysis.

Your code must do all the following:

1) Read in the data – files CDHWdata_1.csv through CDHWdata_5.csv have been uploaded onto the class Canvas site. If you cannot download them, email me this week (or next) and I will send you a copy. These files are comma separated value files, have a text header row, and then each following row is an identifier number and the 15-dimensional feature vector of the data.

Each file has something like 93 items.

2) Use a Euclidean metric (without any scaling or taking out correlations).

3) Rank order the metric distances from point to point, so you can filter on thresholds from below the minimum space between any two points to above the maximum space between any two points.

4) For each value of the threshold (it your code runs fast enough step the threshold between every value the metric distance takes), construct a simplicial complex. It can be any complex construction, but make sure it at least includes 0, 1, 2, and 3 dimensional simplices. I have not examined what happens in dimensions higher than 4 dimensional simplices, but feel free to find out and share with all of us. If your code is slow, try to have enough instances through the filtration to be able to find interesting persistent homologies.

5) For each step in the filtration (threshold), find the simplicial complex, with added simplices at each step, find the homologies with the algorithm like we have disused in class, you might check that the sum of the Betti numbers (at each step) is consistent with the Euler characteristic (a good check to see your code works properly). You need to at least have H0, H1, and H2, but if you can go farther share what you find.

6) Construct some representation of the persistence diagram, and output this. If possible, output it in such a way as can be run through something to graph it, or alternately, graph it in your code.

7) Determine which homologies are persistent and figure out what you can from them. Then provide an analysis of the data. Do this for at least two of the files, and if you have time, do more.
=#
