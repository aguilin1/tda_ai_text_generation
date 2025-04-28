# tda_ai_text_generation

This repository has implementation of two simplicial complexes.
1. Vietoris-Rips - Maintianing minimal reduction of cancelling alternating triangles as they all arrive in sequence as Vietoris Rips just adds everything, without checking whether Balls really intersect as we go to 3 or 4 dimensions. It only checks for intersection of 2 balls (i.e. presence of an edge).
2. Alpha Complex - Using Heron's algorithm and Geometric Analysis of equal balls interseecting for largest possible tetrahedron, we avoid having to solve for quadratic radius*radius constraint (active set quadratic programming), we also do not create Veronoi cells or perform Delaunay Triangulation as we limit our dimensions to maximum 4. We also incrementally create Alpha complex, while taking care of empty circumsphere property. 

Why not just use Ripser?
If you look at png files describing homologies for CDHWdata_3.csv, you will notice a point (in between 2 blue lines, representing H1 homology) at 5 (ball radius = 2.5), using Vietoris-Rips. For same dataset, just using Alpha complex shows stable H1 homology from 3.5 to 10. This means, Vietoris Rips only approximates underlying homology, does not follow nerve theorem and generates super high dimensional objects for no reason... draw 15 points on a plane, Vietoris-Rips will generate a 14 dimensional object for you at some filtration, leading you to perform 0->C14->C13->C12->C11->C10->C9->C8->C7->C6->C5->C4->C3->C2->C1->0, where as Alpha complex will show this planar structure correctly as you will not find an intersection of 4 balls in 3D space for any 4 points on a plane.

Ripser is so good, that it is almost impossible to resist its use to get some persistence diagrams... When we need to actually study structure, Alpha complex is the way to go.
