using Plots
using PersistenceDiagrams
# Just harcoding persistence data for now.
# CDHWdata_5.csv is saved as persistence_5
# Only printing Homology wrt to filtration, 
# objectes are tracked for their appearance, merger in simplex tree
# these intervals are not yer reduced by edge collapsing, (Filtration=5, diame = 10, produces 1819 edges)
# or collapsing a filled complex to a point.

#=
# Outcomes for CDHWdata_5.txt
# diam = 2*r is used for filtration.
# r = 1.5 :V = 93, :E = 0,    :Tri = 0,     :Tet = 0
# r = 2.5 :V = 93, :E = 80,   :Tri = 4,     :Tet = 0
# r = 3.5 :V = 93, :E = 566,  :Tri = 1308,  :Tet = 34
# r = 3.9 :V = 93, :E = 852,  :Tri = 3097,  :Tet = 56
# r = 5.0 :V = 93, :E = 1819, :Tri = 16969, :Tet = 494
h0 = [93, 31, 1, 1, 1]
h1 = [0, 4, 3, 0, 0]
h2 = [0, 0, 803, 2281, 14748]
x = [3, 5, 7, 7.8, 10]
# due to x also representing birth and y death. all point must land above y=x line (death after birth)
diagram = PersistenceDiagram([(0,500), (250, 390), (350, 500)])
barcode(diagram)
png("persistence_5.png")
=#

# Outcomes for CDHWdata_3.txt
# diam = 2*r is used for filtration.
# r = 1.5 :V = 93, :E = 0,    :Tri = 0,     :Tet = 0
# r = 2.5 :V = 93, :E = 84,   :Tri = 20,    :Tet = 0
# r = 3.5 :V = 93, :E = 542,  :Tri = 1130,  :Tet = 25
# r = 3.9 :V = 93, :E = 853,  :Tri = 3146,  :Tet = 64
# r = 5.0 :V = 93, :E = 1861, :Tri = 18198, :Tet = 609
h0 = [93, 31, 1, 1, 1]
h1 = [0, 5, 0, 0, 0]
h2 = [0, 0, 660, 2321, 15770]
x = [3, 5, 7, 7.8, 10]
# due to x also representing birth and y death. all point must land above y=x line (death after birth)
diagram = PersistenceDiagram([(0,500), (250, 390), (350, 500)])
barcode(diagram)
png("persistence_3.png")
