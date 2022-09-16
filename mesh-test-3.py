import gmsh # ellpsoid with holes
import pygmsh
import meshio

Hw = 500
Hi = 400
Hc = Hw - (910/1024) * Hi 
Wx = 1000
xf = Wx/4

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 50
    ocean = geom.add_rectangle([0.0, 0.0, 0.0], Wx, Hw)
    ice = geom.add_rectangle([xf, Hc, 0.0], (Wx-xf), Hi)
    domain=geom.boolean_fragments(ocean,ice)
    for i,entity in enumerate(domain):
        geom.add_physical(entity,f'{i}')
    mesh = geom.generate_mesh()
    import gmsh
    gmsh.write("mesh.msh")
    gmsh.write("mesh.vtk")

mesh_from_file = meshio.read("mesh.msh")
