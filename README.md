# tda_ai_text_generation

This repository has implementation of two simplicial complexes.
1. Vietoris-Rips - Maintianing minimal reduction of cancelling alternating triangles as they all arrive in sequence as Vietoris Rips just adds everything, without checking whether Balls really intersect as we go to 3 or 4 dimensions. It only checks for intersection of 2 balls (i.e. presence of an edge).
2. Alpha Complex - Using Heron's algorithm and Geometric Analysis of equal balls interseecting for largest possible tetrahedron, we avoid having to solve for quadratic radius*radius constraint (active set quadratic programming), we also do not create Veronoi cells or perform Delaunay Triangulation as we limit our dimensions to maximum 4. We also incrementally create Alpha complex, while taking care of empty circumsphere property. 
