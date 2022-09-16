import pygmsh


Wx = 100000
xf = Wx/4
Hw = 500
ice_thickness = 400
Hi = Hw + ((1024-910)/1024) * ice_thickness
Hc = Hw - (910/1024) * ice_thickness
resolution = 100

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

outer_points= [model.add_point((0, 0, 0), mesh_size=resolution),
          model.add_point((Wx, 0, 0), mesh_size=resolution),
          model.add_point((Wx, Hc, 0), mesh_size=resolution),
          model.add_point((xf, Hc, 0), mesh_size=resolution),
          model.add_point((xf,Hw,0), mesh_size=resolution),
          model.add_point((0, Hw, 0), mesh_size=resolution)]

ice_points= [model.add_point((xf, Hc, 0), mesh_size=resolution),
          model.add_point((Wx, Hc, 0), mesh_size=resolution),
          model.add_point((Wx, Hi, 0), mesh_size=resolution),
          model.add_point((xf, Hi, 0), mesh_size=resolution)]

outer_lines= [model.add_line(outer_points[i], outer_points[i+1])
                 for i in range(-1, len(outer_points)-1)]

ice_lines= [model.add_line(ice_points[i], ice_points[i+1])
                 for i in range(-1, len(ice_points)-1)]

outer_loop = model.add_curve_loop(outer_lines)
plane_surface = model.add_plane_surface(outer_loop)
ice_loop = model.add_curve_loop(ice_lines)
ice_surface = model.add_plane_surface(ice_loop)



# Call gmsh kernel before add physical entities
model.synchronize()

#volume_marker = 6
#model.add_physical([plane_surface], "Volume")
#model.add_physical([channel_lines[0]], "Inflow")
#model.add_physical([channel_lines[2]], "Outflow")
#model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
#model.add_physical(ice.curve_loop.curves, "Obstacle")

geometry.generate_mesh(dim=2)

import gmsh
gmsh.write("mesh.vtk")
gmsh.clear()
geometry.__exit__()
