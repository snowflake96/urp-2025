import meshio
import sys

if len(sys.argv) != 2:
    print("Usage: python inspect_msh.py your_mesh.msh")
    sys.exit(1)

mesh = meshio.read(sys.argv[1])
print(f"Points: {mesh.points.shape[0]}")
print("Cell types and counts:")
for block in mesh.cells:
    print(f"  {block.type:12s} : {len(block.data)}")