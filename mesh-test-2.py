import pygmsh
import gmsh
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
rectangle = model.add_rectangle(0.0, 1.0, 0.0, 1.0, 0)
rectangle2 = model.add_rectangle(1.0, 2.0, -1.0, 2.0, 0,mesh_size=0.1)
geometry.generate_mesh(dim=2)
gmsh.write("mesh.vtk")
gmsh.clear()
geometry.__exit__()
