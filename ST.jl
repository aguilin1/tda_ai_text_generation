# This module builds a SimplexTree by gluing
# skeleton.
# This module allows for Realization to happen lazily.
"""
Based on updated paper by - Jean-Daniel Boissonnat, Karthik C. S., Sébastien Tavenas
Introducing Maximal Simplex Tree (MxST) and the Simplex Array List (SAL).
"""
# Modeled after https://arxiv.org/abs/1503.07444 + Gluing that I started with :)
# simplified Hasse diagram with prefix tree(removing trie) like structure saving some connections.


# added realize (after vertices and edges are added, calling expansion will add triangles, tetrahedrons..)
# similar to geometric realization but on simplicial sets (gluing behavior)

# Modeled after Hodge SimplexTree implementation, while changing types/structure, making it persistent and keeping 
# of vertex, edge, triangles, tetra's birth times. There are also performance related changes.
# gluing happens as add_.. is called in insert. (Not in Hodge SimplexTree, Hodge also does not track filtration.)
# Also looked into DataStructures.IntDisjointSets : for my purpose this tree representation is easier to debug
# -- another issue : Julia implementation does not implement path halving optimization yet.
# -- from wiki The docs say that “path compression” is used for the find operation, which is good enough for 
#    (optimal) O(α(n)) complexity, but requires two traversals of a path from node to root. In contrast, 
#    the Wikipedia entry notes that path-halving or path-splitting can be used to remove one of the traversals and 
#    still retain the optimal complexity (and is fast in practice).
#    SRC https://discourse.julialang.org/t/complexity-implementation-of-disjoint-set-union-find-in-datastructures-jl/119050
# as I can easily track which triangles are part of which tetrahedron (path to root is a triangle)
#
"""
Jean-Daniel Boissonnat, Clément Maria. The Simplex Tree: An Efficient Data Structure for General
Simplicial Complexes. [Research Report] RR-7993, 2012, pp.20. <hal-00707901v1>
&
https://github.com/iagoleal/Hodge.jl/blob/master/src/SimplexTrees.jl
+ Gluing + Filtration.
"""
module STs

abstract type ST end

# Node - abstract type, subtype of tree ST
mutable struct _node <: ST
    id       :: Int             # Just a label
    children :: Array{_node, 1}
    parent   :: ST
    birth    :: Float32
    death    :: Float32
    # create new node, julia does not have classes like Python.
    # Julia dispatches over type.
    # Create nodes with same birth and death=-birth, to create complete objects... 
    # edit death as linking to children happens.
    # Objects that do not die, death value there will have -value.
    _node(id, parent, filtration, death) = new(id, [], parent, filtration, death)
end

# Root - abstract type, subtype of Tree ST
struct _root <: ST
    children :: Array{_node, 1}
    _root() = new([])
end

# check whether a node is empty

# check whether a node is last one
# Check whether underlying array carrying children is empty
isleaf(s::ST) = isempty(s.children)
# check whether node has children of id == 'someid'

function haschild(_node::ST, val::Integer)
    # Using Base.in generic function extended to these types.
    # map calls first to all children of _node.children
    return val in map(x -> x.id, _node.children)
end
# check for Simplex

function get_children(_node::ST)
    return map(x -> x.id, _node.children)
end

function get_children(_node::ST, val::Integer)
    # instead of calling find_child and then get children with label 
    id = -5     # for debugging.
    node = find_child(_node::ST, val::Integer)
    return get_children(node)
end

function find_child(_node::ST, val::Integer)
    id = -5     # for debugging
    id = findfirst(x -> x.id == val, _node.children)
    if id === nothing
        return nothing   # nothing has special meaning in Julia
    else 
        return _node.children[id]
    end
end

# insertion will only work as we insert vertex to edges to triangles to tet etc..
# not straight tet
function insert!(s::ST, x, filtration, death)
    # this sort is why we do not need to remeber order
    # this sort will be sort after distance metric. <-- fed by column sort order of mapslices output.
    return insert_ord!(s, sort(collect(x)), filtration, death)
end

function insert_ord!(s::ST, x, filtration, death)
    for (i,v) in enumerate(x)
        if haschild(s,v)
            # do nothing
        else
            # Gluing happens before insertion.
            push!(s.children, _node(v,s, filtration, death))
	        # As soon as link is established. vertex becomes an edge so we set death
            sibling = find_child(s,v)
            simplex = simplex_node(sibling)
	        if length(simplex) > 1
                # 0 is root node
                for k in simplex
                    if k == v || k == 0
                        continue
                    else
                        killedchild = find_child(s.parent, k)
                        if !isnothing(killedchild)
                            # only set new death, if it was not set
                            if killedchild.death == 0.0
                                killedchild.death=filtration
                            end
                        end
                    end
                end
            end
        end
        # only look deeper by dropping i items
        insert_ord!(find_child(s,v), Iterators.drop(x,i), filtration, death)
    end
    return s
end

# insert cluster of nodes with right order
function insert_ord_nodes!(s::ST, simp, filtration, death)
    node = s # tree
    for v in simp
        if haschild(node, v)
            node = find_child(node, v)
        else
            local_node = _node(v,node, filtration, death)
            push!(node.children,local_node)
	        # as higher order objects connect, lower order objects die
            # get path to parent and set death for all upper ones.
            sibling = find_child(s,v)
            simplex = simplex_node(v)
	        if length(simplex) > 1
                for k in simplex
                    # 0 is root node
                    if k == v || k == 0
                        continue
                    else
                        killedchild = find_child(s.parent, k)
                        if !isnothing(killedchild)
                            if killedchild.death == 0.0
                                killedchild.death=filtration
                            end
                        end
                    end
                end
            end
            node = local_node
        end
    end
    return s # julia passes args by sharing memory
end

function fold_leaves(_node; nomore, concat)
    if isleaf(_node)
        return nomore(_node)
    else
        return concat(map(x -> fold_leaves(x; nomore=nomore, concat=concat), _node.children))
    end
end

function fold_tree(_node, depth; nomore, concat)
    if depth == 0
        return nomore(_node)
    else
        return concat(map(x -> fold_tree(x, depth-1; nomore=nomore, concat=concat), _node.children))
    end
end

function numsimplices(s::ST, dim::Integer)
    if dim < 0
        return 0
    else
        return fold_tree(s, dim+1; nomore = x -> 1, concat = sum)
    end
end

isemptyface(s::ST) = typeof(s) == _root

# Simplex.
function simplex_node(_node::ST)
    simp = Array{Int,1}()
    while !isemptyface(_node)
        pushfirst!(simp, _node.id)
        _node = _node.parent
    end
    return simp
end

function getsimplices(s::ST, dim::Integer)
    if dim < 0
        return Vector{int}[]
    else
        # if vcat with splatted is too slow use reduce.
        return fold_tree(s, dim+1, nomore = x -> Vector{Int}[simplex_node(x)], concat = x -> Vector{Vector{Int}}(vcat(x...)))
    end
end

function children_data(s::ST)
    return map(x -> x.id, _node.children)
end

# As described in paper, everything is laid out in an array.
function hassimplex(s::ST, simp)
    if isempty(simp)
        return true
    elseif !haschild(s, simp[1])
        return false
    else
        # blast find_child to all remaining elements.
        return hassimplex(find_child(s, simp[1]), simp[2:end])
    end
end

end #module
