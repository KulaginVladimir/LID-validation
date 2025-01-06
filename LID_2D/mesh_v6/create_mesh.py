import gmsh
import sys

gmsh.initialize()
# gmsh.option.setNumber("Mesh.Algorithm", 8)

R = 2e-3
L_Cu = 6e-3
L_W = 1e-6

gmsh.model.geo.addPoint(0, 0, 0, 1e-4, 1)  # left bottom Cu node
gmsh.model.geo.addPoint(R, 0, 0, 1e-4, 2)  # right bottom Cu node
gmsh.model.geo.addPoint(R, L_Cu, 0, 5e-7, 3)  # right top Cu node
gmsh.model.geo.addPoint(R, L_Cu + L_W, 0, 5e-7, 4)  # right top W node
gmsh.model.geo.addPoint(0, L_Cu + L_W, 0, 5e-8, 5)  # left top W node
gmsh.model.geo.addPoint(0, L_Cu, 0, 5e-7, 6)  # left top Cu node

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 1, 6)
gmsh.model.geo.addLine(3, 6, 7)

gmsh.model.geo.addCurveLoop([1, 2, 7, 6], 1)
gmsh.model.geo.addCurveLoop([3, 4, 5, -7], 2)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)

gmsh.model.geo.synchronize()

bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl, "CurvesList", [4])
gmsh.model.mesh.field.setNumbers(bl, "PointsList", [4, 5])
gmsh.model.mesh.field.setNumber(bl, "Size", 5e-8)
gmsh.model.mesh.field.setNumber(bl, "Ratio", 1.2)
gmsh.model.mesh.field.setNumber(bl, "Quads", 0)
gmsh.model.mesh.field.setNumber(bl, "Thickness", 5e-7)
gmsh.model.mesh.field.setNumber(bl, "SizeFar", 1e-6)

gmsh.model.mesh.field.setAsBoundaryLayer(bl)

gmsh.model.addPhysicalGroup(2, [1], 1)  # Cu domain
gmsh.model.addPhysicalGroup(2, [2], 2)  # W domain

gmsh.model.addPhysicalGroup(1, [1], 3)  # bottom Cu boundary
gmsh.model.addPhysicalGroup(1, [2], 4)  # right Cu boundary
gmsh.model.addPhysicalGroup(1, [3], 5)  # right W boundary
gmsh.model.addPhysicalGroup(1, [4], 6)  # top W boundary

# gmsh.option.setNumber("Mesh.Smoothing", 50)

gmsh.model.mesh.generate(2)
gmsh.write("WCu.msh")

# Launch the GUI to see the results:
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
