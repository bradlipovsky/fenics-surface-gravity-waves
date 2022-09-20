import numpy as np
import gmsh # ellpsoid with holes
import meshio

Hw = 500
Hi = 400
Hc = Hw - (910/1024) * Hi 
Wx = 1000
xf = Wx/4

gmsh.initialize()
water_rect   = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, xf, Hw, tag=3)
water_rect_2 = gmsh.model.occ.addRectangle(xf,0.0,0.0,(Wx-xf),Hc, tag=1)
ice_rect     = gmsh.model.occ.addRectangle(xf, Hc, 0.0, (Wx-xf), Hi, tag=2)
gmsh.model.occ.fuse([(2,1)],[(2,3)],tag=4)
gmsh.model.occ.synchronize()

water, ice_lower, ice_upper = None, None, None
# Label the surfaces (ice and water)
for surface in gmsh.model.getEntities(dim=2):
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if com[0] < Wx/2:
        water = surface[1] 
    else:
        ice = surface[1]

# Add physical surfaces
gmsh.model.addPhysicalGroup(2,[water],0)
gmsh.model.addPhysicalGroup(2,[ice],1)

# Label boundaries
ice_interface = []
for edge in gmsh.model.getEntities(dim=1):
    com = gmsh.model.occ.getCenterOfMass(edge[0], edge[1])
    if np.isclose(com[1], Hw):
        water_surface = [edge[1]]
    elif np.isclose(com[0],xf):
        ice_interface.append(edge[1])
    elif np.isclose(com[1],Hc):
        ice_interface.append(edge[1])

# Add physical entities
print(water_surface)
print(ice_interface)
gmsh.model.addPhysicalGroup(1,water_surface,2)
gmsh.model.addPhysicalGroup(1,ice_interface,3)

gmsh.option.setNumber("Mesh.MeshSizeMax", 5)

gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
gmsh.finalize()

'''
Convert the mesh
'''
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points

    out_mesh = meshio.Mesh(points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]})

    return out_mesh

mesh_from_file = meshio.read("mesh.msh")

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("facet_mesh.xdmf", line_mesh)
