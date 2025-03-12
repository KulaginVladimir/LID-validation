import meshio
import numpy as np
import festim as F

# Convert mesh to XDMF
filename = "WCu.msh"
msh = meshio.read(filename)

# Initialize lists to store cells and their corresponding data
triangle_cells_list = []
line_cells_list = []
triangle_data_list = []
line_data_list = []

# Extract cell data for all types
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells_list.append(cell.data)
    elif cell.type == "line":
        line_cells_list.append(cell.data)

# Extract physical tags
for key, data in msh.cell_data_dict["gmsh:physical"].items():
    if key == "triangle":
        triangle_data_list.append(data)
    elif key == "line":
        line_data_list.append(data)

# Concatenate all tetrahedral cells and their data
line_cells = np.concatenate(line_cells_list)
line_data = np.concatenate(line_data_list)

# Concatenate all triangular cells and their data
triangle_cells = np.concatenate(triangle_cells_list)
triangle_data = np.concatenate(triangle_data_list)

# Create the triangular mesh for the surface
triangle_mesh = meshio.Mesh(
    points=msh.points,
    cells=[("triangle", triangle_cells)],
    cell_data={"f": [triangle_data]},
)

# Create the line mesh for boundaries
line_mesh = meshio.Mesh(
    points=msh.points,
    cells=[("line", line_cells)],
    cell_data={"f": [line_data]},
)

# Print unique surface and volume IDs
print("Surface IDs: ", np.unique(line_data))
print("Volume IDs: ", np.unique(triangle_data))

# Write the mesh files
meshio.write("mesh.xdmf", triangle_mesh)
meshio.write("mf.xdmf", line_mesh)
