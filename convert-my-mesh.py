import gmsh
import pygmsh
import meshio
import numpy
from dolfin import *
msh = meshio.read("mesh.msh")

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

line_mesh = create_mesh(msh, "line", prune_z=True)
meshio.write("mf.xdmf", line_mesh)

triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)


mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
